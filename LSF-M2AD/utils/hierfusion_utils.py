import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Reduce

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def temperature_softmax(logits, temperature=1.0, dim=-1):
    scaled_logits = logits / temperature
    return F.softmax(scaled_logits, dim=dim)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GELU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class SELU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.selu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0., snn: bool = False):
        super().__init__()
        activation = SELU() if snn else GELU()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            activation,
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim, eps=1e-4) if exists(context_dim) else None
    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)
            if torch.isnan(normed_context).any():
                print("NaN detected in normed_context")
        return self.fn(x, **kwargs)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.LeakyReLU(negative_slope=1e-2)
        )
        self.attn_weights = None
    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)
        attn = temperature_softmax(sim, temperature=0.5, dim=-1)
        self.attn_weights = attn
        attn = self.dropout(attn)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class LanguageFusionLayer(nn.Module):
    def __init__(self, query_dim, lang_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(lang_dim, inner_dim * 2, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, context):
        h = self.heads
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return x + self.to_out(out)

class PreNorm_hyper(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.BatchNorm3d(128)
        self.norm3 = nn.BatchNorm3d(128)
        self.fn = fn
    def forward(self, h_latent, h_mri, h_pet):
        h_latent = self.norm1(h_latent)
        h_mri = self.norm2(h_mri) if h_mri is not None else None
        h_pet = self.norm3(h_pet) if h_pet is not None else None
        return self.fn(h_latent, h_mri, h_pet)

class HyperLearningLayer(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k_mri = nn.Conv3d(128, inner_dim, kernel_size=1, bias=False)
        self.to_k_pet = nn.Conv3d(128, inner_dim, kernel_size=1, bias=False)
        self.to_v_mri = nn.Conv3d(128, inner_dim, kernel_size=1, bias=False)
        self.to_v_pet = nn.Conv3d(128, inner_dim, kernel_size=1, bias=False)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=True),
            nn.Dropout(dropout)
        )
    def _process_modality(self, q, features, to_k, to_v):
        if features is None or torch.all(features == 0):
            return torch.zeros_like(q)
        k = to_k(features)
        v = to_v(features)
        k = self.global_pool(k).flatten(2).permute(0, 2, 1)
        v = self.global_pool(v).flatten(2).permute(0, 2, 1)
        h = self.heads
        q_heads = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k_heads = rearrange(k, 'b m (h d) -> b h m d', h=h)
        v_heads = rearrange(v, 'b m (h d) -> b h m d', h=h)
        dots = einsum('b h i d, b h j d -> b h i j', q_heads, k_heads) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v_heads)
        return rearrange(out, 'b h n d -> b n (h d)')
    def forward(self, h_latent, h_mri, h_pet):
        q = self.to_q(h_latent)
        out_mri = self._process_modality(q, h_mri, self.to_k_mri, self.to_v_mri)
        out_pet = self._process_modality(q, h_pet, self.to_k_pet, self.to_v_pet)
        return h_latent + self.to_out(out_mri + out_pet)
