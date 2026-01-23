########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math, gc, importlib
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy
if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

try:
    print('RWKV_MY_TESTING', os.environ["RWKV_MY_TESTING"])
except:
    os.environ["RWKV_MY_TESTING"] = ''

def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop
if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method


########################################################################################################
# CUDA Kernel
########################################################################################################

from torch.utils.cpp_extension import load

from torch.nn.utils import skip_init

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE"])

if 'x070' in os.environ["RWKV_MY_TESTING"]:
    CHUNK_LEN = 16

    flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
    load(name="wind_backstepping", sources=[f'cuda/wkv7_cuda.cu', 'cuda/wkv7_op.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)

    class WindBackstepping(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w,q,k,v,z,b):
            B,T,H,C = w.shape 
            assert T%CHUNK_LEN == 0 # if T%CHUNK_LEN != 0: pad your input to T%CHUNK_LEN == 0, or change CHUNK_LEN (will be slower)
            assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
            assert all(i.is_contiguous() for i in [w,q,k,v,z,b])
            y = torch.empty_like(v)
            s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
            sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
            torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)
            ctx.save_for_backward(w,q,k,v,z,b,s,sa)
            return y
        @staticmethod
        def backward(ctx, dy):
            assert all(i.dtype==torch.bfloat16 for i in [dy])
            assert all(i.is_contiguous() for i in [dy])
            w,q,k,v,z,b,s,sa = ctx.saved_tensors
            dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
            torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
            return dw,dq,dk,dv,dz,db

    def RUN_CUDA_RWKV7g(q,w,k,v,a,b):
        B,T,HC = q.shape
        q,w,k,v,a,b = [i.view(B,T,HC//64,64) for i in [q,w,k,v,a,b]]
        return WindBackstepping.apply(w,q,k,v,a,b).view(B,T,HC)

########################################################################################################

class RWKV_Tmix_x070(MyModule):

    def __init__(self, args, layer_id):
        '''
        The parameters are not initialized in the constructor.
        '''

        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.my_testing = args.my_testing

        self.head_size = args.head_size
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        H = self.n_head
        N = self.head_size
        C = args.n_embd

        # the parameters
        self.x_r = nn.Parameter(torch.empty(1, 1, C))
        self.x_w = nn.Parameter(torch.empty(1, 1, C))
        self.x_k = nn.Parameter(torch.empty(1, 1, C))
        self.x_v = nn.Parameter(torch.empty(1, 1, C))
        self.x_a = nn.Parameter(torch.empty(1, 1, C))
        self.x_g = nn.Parameter(torch.empty(1, 1, C))

        self.D_DECAY_LORA = max(32, int(round(  (2.5*(C**0.5))  /32)*32)) # suggestion
        self.w1 = nn.Parameter(torch.empty(C, self.D_DECAY_LORA))
        self.w2 = nn.Parameter(torch.empty(self.D_DECAY_LORA, C))
        self.w0 = nn.Parameter(torch.empty(1, 1, C))

        self.D_AAA_LORA = max(32, int(round(  (2.5*(C**0.5))  /32)*32)) # suggestion
        self.a1 = nn.Parameter(torch.empty(C, self.D_AAA_LORA))
        self.a2 = nn.Parameter(torch.empty(self.D_AAA_LORA, C))
        self.a0 = nn.Parameter(torch.empty(1, 1, C))

        self.D_MV_LORA = max(32, int(round(  (1.7*(C**0.5))  /32)*32)) # suggestion
        self.v1 = nn.Parameter(torch.empty(C, self.D_MV_LORA))
        self.v2 = nn.Parameter(torch.empty(self.D_MV_LORA, C))
        self.v0 = nn.Parameter(torch.empty(1, 1, C))

        # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
        self.D_GATE_LORA = max(32, int(round(  (5*(C**0.5))  /32)*32)) # suggestion
        self.g1 = nn.Parameter(torch.empty(C, self.D_GATE_LORA))
        self.g2 = nn.Parameter(torch.empty(self.D_GATE_LORA, C))

        self.k_k = nn.Parameter(torch.empty(1, 1, C))
        self.k_a = nn.Parameter(torch.empty(1, 1, C))
        self.r_k = nn.Parameter(torch.empty(H, N))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = skip_init(nn.Linear, C, C, bias=False)
        self.key = skip_init(nn.Linear, C, C, bias=False)
        self.value = skip_init(nn.Linear, C, C, bias=False)
        self.output = skip_init(nn.Linear, C, C, bias=False)
        self.ln_x = skip_init(nn.GroupNorm, H, C, eps=64e-5) # !!! notice eps value !!!


    def reset_parameters(self):
        '''
        Initialize the parameters of the model based on the layer_id.
        '''

        device = self.x_r.device

        H = self.n_head
        N = self.head_size
        C = self.args.n_embd

        with torch.device(device):
            with torch.no_grad():
                ratio_0_to_1 = self.layer_id / (self.args.n_layer - 1)  # 0 to 1
                ratio_1_to_almost0 = 1.0 - (self.layer_id / self.args.n_layer)  # 1 to ~0
                ddd = torch.ones(1, 1, C)
                for i in range(C):
                    ddd[0, 0, i] = i / C

                self.x_r.data = 1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0)
                self.x_w.data = 1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0)
                self.x_k.data = 1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0)
                self.x_v.data = 1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0)
                self.x_a.data = 1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0)
                self.x_g.data = 1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0)

                def ortho_init(x, scale) -> torch.Tensor:
                    with torch.no_grad():
                        shape = x.shape
                        if len(shape) == 2:
                            gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                            nn.init.orthogonal_(x, gain=gain * scale)
                        elif len(shape) == 3:
                            gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                            for i in range(shape[0]):
                                nn.init.orthogonal_(x[i], gain=gain * scale)
                        else:
                            raise ValueError(f"Unsupported shape {shape} for orthogonal initialization")
                        return x

                www = torch.zeros(C)
                zigzag = torch.zeros(C)
                linear = torch.zeros(C)

                for n in range(C):
                    linear[n] = n / (C-1) - 0.5
                    zigzag[n] = ((n % N) - ((N-1) / 2)) / ((N-1) / 2)
                    zigzag[n] = zigzag[n] * abs(zigzag[n])
                    www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)

                self.w1.data = torch.zeros(C, self.D_DECAY_LORA)
                self.w2.data = ortho_init(torch.zeros(self.D_DECAY_LORA, C), 0.1)
                self.w0.data = www.reshape(1,1,C) + 0.5 + zigzag*2.5 # !!! 0.5 comes from F.softplus !!!

                self.a1.data = torch.zeros(C, self.D_AAA_LORA)
                self.a2.data = ortho_init(torch.zeros(self.D_AAA_LORA, C), 0.1)
                self.a0.data = torch.zeros(1, 1, C) - 0.19 + zigzag*0.3 + linear*0.4

                self.v1.data = torch.zeros(C, self.D_MV_LORA)
                self.v2.data = ortho_init(torch.zeros(self.D_MV_LORA, C), 0.1)
                self.v0.data = torch.zeros(1, 1, C)+0.73 - linear*0.4

                self.g1.data = torch.zeros(C, self.D_GATE_LORA)
                self.g2.data = ortho_init(torch.zeros(self.D_GATE_LORA, C), 0.1)

                self.k_k.data = torch.zeros(1, 1, C)+0.71 - linear*0.1
                self.k_a.data = torch.zeros(1, 1, C)+1.02
                self.r_k.data = torch.zeros(H, N)-0.04

                # self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
                nn.init.orthogonal_(self.receptance.weight.data, gain=1.0)  # type: ignore
                nn.init.orthogonal_(self.key.weight.data, gain=0.1)   # type: ignore
                nn.init.orthogonal_(self.value.weight.data, gain=1.0)  # type: ignore
                self.output.weight.data = torch.zeros(C, C)

                layer_scale = (1+self.layer_id) / self.args.n_layer
                self.ln_x.weight.data = torch.ones(C) * (layer_scale ** 0.7)
                self.ln_x.bias.data = torch.zeros_like(self.ln_x.bias.data) # type: ignore


    @MyFunction
    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)

        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first
    
########################################################################################################

class RWKV_CMix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        C = args.n_embd

        # the parameters
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_k = nn.Parameter(torch.empty(1, 1, C))
        self.key = skip_init(nn.Linear, C, C * 4, bias=False)
        self.value = skip_init(nn.Linear, C * 4, C, bias=False)


    def reset_parameters(self):

        device = self.key.weight.device

        C = self.args.n_embd

        with torch.no_grad():
            with torch.device(device):
                ratio_1_to_almost0 = 1.0 - (self.layer_id / self.args.n_layer)  # 1 to ~0
                ddd = torch.ones(1, 1, C)
                for i in range(C):
                    ddd[0, 0, i] = i / C
                self.x_k.data = 1.0 - torch.pow(ddd, ratio_1_to_almost0**4)

                nn.init.orthogonal_(self.key.weight.data, gain=1.0)  # type: ignore
                self.value.weight.data = torch.zeros(C, C * 4)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)


########################################################################################################
# The RWKV Model with our blocks
########################################################################################################

class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = skip_init(nn.LayerNorm, args.n_embd)
        self.ln2 = skip_init(nn.LayerNorm, args.n_embd)

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)

    def reset_parameters(self):

        C = self.args.n_embd

        device = self.ln1.weight.device

        with torch.device(device):
            ln1 = nn.LayerNorm(C)
            self.ln1.weight.data = ln1.weight.data
            self.ln1.bias.data = ln1.bias.data

            ln2 = nn.LayerNorm(C)
            self.ln2.weight.data = ln2.weight.data
            self.ln2.bias.data = ln2.bias.data

            self.att.reset_parameters()
            self.ffn.reset_parameters()
        
    def forward(self, x, v_first):
        x_attn, v_first = self.att(self.ln1(x), v_first)
        x = x + x_attn

        x = x + self.ffn(self.ln2(x))
        return x, v_first


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)


class RWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if not hasattr(args, 'dim_att'):
            args.dim_att = args.n_embd
        if not hasattr(args, 'dim_ffn'):
            args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size            
        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        C = args.n_embd

        self.emb = skip_init(nn.Embedding, args.vocab_size, C)

        self.ln_in = skip_init(nn.LayerNorm, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = skip_init(nn.LayerNorm, C)
        self.head = skip_init(nn.Linear, C, args.vocab_size, bias=False)

    def reset_parameters(self):
        # embedding layer
        scale_emb = -1e-4
        nn.init.uniform_(self.emb.weight, a=scale_emb, b=-scale_emb)    # type: ignore

        # ln_in
        ln_in = nn.LayerNorm(self.args.n_embd)
        self.ln_in.weight.data = ln_in.weight.data
        self.ln_in.bias.data = ln_in.bias.data

        for block in self.blocks:
            block.reset_parameters()

        # ln_out
        ln_out = nn.LayerNorm(self.args.n_embd)
        self.ln_out.weight.data = ln_out.weight.data
        self.ln_out.bias.data = ln_out.bias.data


        # head layer
        if self.args.vocab_size > self.args.n_embd:
            scale = 0.5 * math.sqrt(self.args.vocab_size / self.args.n_embd)
        else:
            scale = 0.5
        nn.init.orthogonal_(self.head.weight, gain=scale)  # type: ignore


    def configure_optimizers(self):
        args = self.args
        
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        for n, p in self.named_parameters():
            if ("att.w0" in n):
                lr_2x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0) and (".weight" in n):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))

        # print the learning rate groups for debugging
        if self.trainer.is_global_zero:
            print('decay', lr_decay, '\n')
            print('1x', lr_1x, '\n')
            print('2x', lr_2x, '\n')

        param_dict = {n: p for n, p in self.named_parameters()}
        
        optim_groups = [
            {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
            {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
        ]

        if args.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
        
        else:
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=False, weight_decay=0, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def forward(self, idx):
        args = self.args
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)

        x = self.ln_in(x)

        v_first = torch.empty_like(x)
        for block in self.blocks:
            if args.grad_cp == 1:
                x, v_first = deepspeed.checkpointing.checkpoint(block, x, v_first)
            else:
                x, v_first = block(x, v_first)

        x = self.ln_out(x)
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        idx, targets = batch
        logits = self(idx)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return L2Wrap.apply(loss, logits)

    def training_step_end(self, batch_parts):
        all = self.all_gather(batch_parts)
        if self.trainer.is_global_zero:
            self.trainer.my_loss_all = all


    def generate_init_weight(self):
        print(
            f"""
############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################
"""
        )
        self.reset_parameters()

        m = {}
        n_params = 0

        for n in self.state_dict():
            p = self.state_dict()[n]
            
            m[n] = p.cpu()
            if os.environ["RWKV_FLOAT_MODE"] == "fp16":
                m[n] = m[n].half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                m[n] = m[n].bfloat16()

            n_params += m[n].numel()

        print('model params', n_params)
        gc.collect()
        torch.cuda.empty_cache()
        return m
