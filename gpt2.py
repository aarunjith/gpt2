import inspect
import time
from tqdm.auto import tqdm
import math
from transformers import GPT2LMHeadModel, pipeline, set_seed
import torch.optim as optim
import tiktoken
import torch.nn as nn
import torch
import torch.nn.functional as F
from dataclasses import dataclass

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# device = torch.device("cpu")


@dataclass
class GPTConfig:
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    vocab_size: int = 50257
    block_size: int = 1024

@dataclass
class LRSchedulerConfig:
    max_lr: float = 3e-4
    min_lr: float = max_lr * 0.1
    warmup_steps: int = 10
    max_steps: int = 50

def get_lr(step, config):
    if step < config.warmup_steps:
        return config.max_lr*(step+1)/config.warmup_steps
    if step > config.max_steps:
        return config.min_lr
    decay_ratio = (step - config.warmup_steps)/(config.max_steps - config.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.max_lr - config.min_lr)

class DataLoader:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        with open("input.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        self.input_tokens = torch.tensor(enc.encode(text)).to(device)
        print(f"Loaded {self.input_tokens.shape[0]} tokens")
        print(f"1 Epoch is {self.input_tokens.shape[0] // (self.B * self.T)} batches")
        self.current_idx = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.input_tokens[self.current_idx : self.current_idx + B * T + 1]
        x = buf[:-1].view(B, T).to(device)
        y = buf[1:].view(B, T).to(device)
        self.current_idx += B * T
        if self.current_idx + (B * T + 1) > len(self.input_tokens):
            self.current_idx = 0
        return x, y


def get_batch(batch_size, config):
    ix = torch.randint(len(input_tensor) - config.block_size, (batch_size,))
    x = torch.stack([input_tensor[i : i + config.block_size] for i in ix])
    y = torch.stack([input_tensor[i + 1 : i + config.block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.c_proj.NANOGPT_SCALE = 1
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        '''
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        '''
        # Flash Attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, 4 * embed_dim)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * embed_dim, embed_dim)
        self.c_proj.NANOGPT_SCALE = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config.n_embed)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embed),
                wpe=nn.Embedding(config.block_size, config.n_embed),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embed),
            )
        )
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        # the last layer should be the same as the embedding layer !!!
        self.transformer["wte"].weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.01
            if hasattr(module, "NANOGPT_SCALE"):
                std = (
                    2 * config.n_layer**-0.5
                )  # 2 is coming from the 2 residual pathways
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=std
            )  # std should depend upon the embedding dim
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        device = input_ids.device
        b, t = input_ids.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward input with length {t}, block size is only {self.config.block_size}"
        tok_emb = self.transformer["wte"](input_ids)
        pos_emb = self.transformer["wpe"](torch.arange(0, t, device=device))
        x = tok_emb + pos_emb
        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, top_k=50):
        for _ in range(max_new_tokens):
            logits, _ = self(input_ids[:, -self.config.block_size :])
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            top_k_probs, top_k_indices = probs.topk(top_k, dim=-1)
            next_id = torch.multinomial(top_k_probs, num_samples=1)
            next_id = torch.gather(top_k_indices, dim=-1, index=next_id)
            input_ids = torch.cat((input_ids, next_id), dim=-1)
        return input_ids

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
        print(f"Loading pretrained model {model_type}")
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embed=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embed=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embed=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embed=1600),
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024
        # Create config object by setting attributes directly instead of using kwargs
        config = GPTConfig()
        for key, value in config_args.items():
            setattr(config, key, value)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_hf_keys = [k for k in sd_hf.keys() if not k.endswith(".attn.bias")]
        sd_hf_keys = [k for k in sd_hf_keys if not k.endswith(".attn.masked_bias")]
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        for k in sd_keys:
            if any(k.endswith(suffix) for suffix in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].T)
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    # Implement Weight Decay
    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn,p in self.named_parameters()}
        param_dict = {pn: p for pn,p in param_dict.items() if p.requires_grad}
        # Weight decay only 2D params or above, dont decay batchnorm/ bias etc
        decay_params = [p for n,p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n,p in param_dict.items() if p.dim() < 2]
        optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f'Decayed Parameters : {num_decay_params}')
        print(f'Not Decayed Parameters : {num_nodecay_params}')
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f'Fused AdamW : {use_fused}')
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
        return optimizer

# model = GPT.from_pretrained("gpt2")
config = GPTConfig(vocab_size=50304)
model = GPT(config).to(device)
model = torch.compile(model)

total_batch_size = 524288
micro_batch_size = 16
assert total_batch_size % (micro_batch_size * config.block_size) == 0, "Ensure Total Batch Size is divisible by B * T"
grad_accum_steps = total_batch_size // (micro_batch_size * config.block_size)
print(f'Desired batch size : {total_batch_size}')
print(f'Gradient Accumulation Steps: {grad_accum_steps}')
train_loader = DataLoader(B=micro_batch_size, T=config.block_size)
optimizer = model.configure_optimizers(0.1, 6e-4, device)
lr_config = LRSchedulerConfig(max_lr=6e-4)


for step in range(lr_config.max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for _ in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss /= grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step, lr_config)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tps = train_loader.B * train_loader.T/(t1 - t0)
    print(f"Step {step} | LR : {lr:.6f} | Loss: {loss_accum:.6f} | norm: {norm.item():.4f} | dt: {dt}ms | TPS: {tps:.4f} tok/sec")

'''
num_return_sequences = 5
max_new_tokens = 30
enc = tiktoken.get_encoding("gpt2")
model.eval()
context = "Hello, I'm a language model,"
input_tokens = enc.encode(context)
input_tensor = torch.tensor([input_tokens]).repeat(num_return_sequences, 1).to(device)
output_tokens = model.generate(input_tensor, max_new_tokens)
output_texts = [
    enc.decode(output_tokens[i].tolist()) for i in range(num_return_sequences)
]
for text in output_texts:
    print(">", text)'''
