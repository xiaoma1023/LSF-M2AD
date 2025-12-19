import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from LSF_M2AD.utils.resnet3D18 import ResNet3D18
from LSF_M2AD.utils.lsfm_utils import TripleFeatureFusion, Patcher, WaveletEncoder, ARFFC
from LSF_M2AD.utils.hierfusion_utils import (
    Mlp, FeedForward, PreNorm, Attention, LanguageFusionLayer, 
    PreNorm_hyper, HyperLearningLayer, Reduce
)

# --- Main Models ---
class LSFM(nn.Module):
    def __init__(self, mri_enc, pet_enc, kl_weight=1.0):
        super().__init__()
        self.mri_raw_enc, self.pet_raw_enc = mri_enc, pet_enc
        self.kl_weight, self.dwt = kl_weight, Patcher()
        self.m2p_low_enc, self.m2p_high_enc = WaveletEncoder(1), WaveletEncoder(7)
        self.p2m_low_enc, self.p2m_high_enc = WaveletEncoder(1), WaveletEncoder(7)
        self.m2p_fusion, self.p2m_fusion = TripleFeatureFusion(), TripleFeatureFusion()
        self.m2p_mu, self.m2p_logvar = nn.Conv3d(128, 128, 1), nn.Conv3d(128, 128, 1)
        self.p2m_mu, self.p2m_logvar = nn.Conv3d(128, 128, 1), nn.Conv3d(128, 128, 1)
        self.m2p_convert, self.p2m_convert = ARFFC(128, 128), ARFFC(128, 128)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def compute_kl_div(self, mu, logvar):
        mu, logvar = torch.clamp(mu, -10, 10), torch.clamp(logvar, -10, 10)
        kl = 0.5 * (mu.pow(2) + torch.exp(logvar) - 1.0 - logvar)
        return kl.mean()

    def forward(self, mri, pet, pairs):
        device, b = mri.device, mri.shape[0]
        mri_feats, pet_feats = torch.zeros((b, 128, 7, 8, 7), device=device), torch.zeros((b, 128, 7, 8, 7), device=device)
        total_kl, total_dao, imp_cnt = 0, 0, 0
        
        # Both Present
        both = torch.where(pairs[:, 0] & pairs[:, 1])[0]
        if len(both) > 0:
            mri_feats[both], pet_feats[both] = self.mri_raw_enc(mri[both]), self.pet_raw_enc(pet[both])

        # MRI -> PET
        m2p = torch.where(pairs[:, 0] & ~pairs[:, 1])[0]
        if len(m2p) > 0:
            m_s, p_gt = mri[m2p], pet[m2p]
            f_raw, dwt_out = self.mri_raw_enc(m_s), self.dwt(m_s)
            p = self.m2p_fusion(self.m2p_low_enc(dwt_out[:, :1]), self.m2p_high_enc(dwt_out[:, 1:]), f_raw)
            mu, logvar = self.m2p_mu(p), self.m2p_logvar(p)
            f_imp = self.m2p_convert(self.reparameterize(mu, logvar))
            mri_feats[m2p], pet_feats[m2p] = f_raw, f_imp
            total_kl += self.compute_kl_div(mu, logvar) * len(m2p)
            with torch.no_grad(): f_gt = self.pet_raw_enc(p_gt)
            total_dao += F.mse_loss(f_imp, f_gt, reduction='sum') / f_imp[0].numel()
            imp_cnt += len(m2p)

        # PET -> MRI
        p2m = torch.where(~pairs[:, 0] & pairs[:, 1])[0]
        if len(p2m) > 0:
            p_s, m_gt = pet[p2m], mri[p2m]
            f_raw, dwt_out = self.pet_raw_enc(p_s), self.dwt(p_s)
            p = self.p2m_fusion(self.p2m_low_enc(dwt_out[:, :1]), self.p2m_high_enc(dwt_out[:, 1:]), f_raw)
            mu, logvar = self.p2m_mu(p), self.p2m_logvar(p)
            f_imp = self.p2m_convert(self.reparameterize(mu, logvar))
            mri_feats[p2m], pet_feats[p2m] = f_imp, f_raw
            total_kl += self.compute_kl_div(mu, logvar) * len(p2m)
            with torch.no_grad(): f_gt = self.mri_raw_enc(m_gt)
            total_dao += F.mse_loss(f_imp, f_gt, reduction='sum') / f_imp[0].numel()
            imp_cnt += len(p2m)

        loss = self.kl_weight * (total_kl / max(1, imp_cnt)) + (total_dao / max(1, imp_cnt))
        return loss, mri_feats, pet_feats

class HierFusion(nn.Module):
    def __init__(self, mri_enc, pet_enc, out_dims, depth=3, l_c=128, l_d=128, l_heads=8, latent_dim_head=64, attn_dropout=0., ff_dropout=0., self_per_cross_attn=1, final_classifier_head=True, snn=True):
        super().__init__()
        self.resnet_mri, self.resnet_pet, self.depth, self.self_per_cross_attn = mri_enc, pet_enc, depth, self_per_cross_attn
        self._lang_dim, self.latents = 512, nn.Parameter(torch.randn(l_c, l_d))
        self.fusion_layers = nn.ModuleList([])
        for _ in range(depth):
            hyper = PreNorm_hyper(l_d, HyperLearningLayer(l_d, 16, latent_dim_head, attn_dropout))
            lang = PreNorm(l_d, LanguageFusionLayer(l_d, 512, l_heads, latent_dim_head, attn_dropout), 512)
            lat_blocks = nn.ModuleList([nn.ModuleList([PreNorm(l_d, Attention(l_d, heads=l_heads, dim_head=latent_dim_head, dropout=attn_dropout)), PreNorm(l_d, FeedForward(l_d, dropout=ff_dropout, snn=snn))]) for _ in range(self_per_cross_attn)])
            self.fusion_layers.append(nn.ModuleList([hyper, lang, lat_blocks]))
        self.to_logits = nn.Sequential(Reduce('b n d -> b d', 'mean'), nn.LayerNorm(l_d), nn.Linear(l_d, out_dims)) if final_classifier_head else nn.Identity()
        self.inverseNet = Mlp(512, 2048, 512, drop=0.1)

    def forward(self, tensors, pairs, report=None, mri_features=None, pet_features=None):
        b, device = pairs.shape[0], pairs.device
        with torch.cuda.amp.autocast():
            if mri_features is None: mri_features = self.resnet_mri(tensors[0]) if tensors[0] is not None else None
            if pet_features is None: pet_features = self.resnet_pet(tensors[1]) if tensors[1] is not None else None
            lang_context = None
            if report is not None:
                if not hasattr(self, 'text_encoder'):
                    from LSF_M2AD.utils.bioclip.load_text import TextEncoder
                    self.text_encoder = TextEncoder()
                batch_text_features = torch.zeros((b, 512), device=device)
                for i in range(b):
                    if pairs[i, 0]: batch_text_features[i] = self.text_encoder.encode(report[i]).to(device).squeeze(0)
                report_proj = self.inverseNet(batch_text_features)
                lang_context = torch.where(pairs[:, 0:1].bool() == False, torch.zeros_like(report_proj), report_proj)
            x = repeat(self.latents, 'n d -> b n d', b=b)
            for hyper, lang, lat_blocks in self.fusion_layers:
                x = hyper(x, mri_features, pet_features)
                if lang_context is not None: x = lang(x, lang_context.unsqueeze(1))
                for attn, ff in lat_blocks: x, x = attn(x) + x, ff(x) + x
            return self.to_logits(x)

class LSF_M2AD(nn.Module):
    """
    Unified LSF-M2AD Model sharing two raw ResNet3D18 encoders.
    """
    def __init__(self, out_dims, **kwargs):
        super().__init__()
        self.mri_enc = ResNet3D18(in_channels=1)
        self.pet_enc = ResNet3D18(in_channels=1)
        self.lsfm = LSFM(self.mri_enc, self.pet_enc)
        self.hier_fusion = HierFusion(self.mri_enc, self.pet_enc, out_dims, **kwargs)

    def forward(self, tensors, pairs, report=None):
        lsfm_loss, mri_f, pet_f = self.lsfm(tensors[0], tensors[1], pairs)
        logits = self.hier_fusion(tensors, pairs, report, mri_f, pet_f)
        return lsfm_loss, logits

