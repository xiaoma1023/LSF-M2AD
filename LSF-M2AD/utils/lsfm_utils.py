import torch
from torch import nn
import torch.nn.functional as F
from LSF_M2AD.utils.resnet3D18 import BasicBlock

class TripleFeatureFusion(nn.Module):
    def __init__(self, in_channels=128):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Conv3d(in_channels * 3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 3, kernel_size=3, padding=1),
            nn.Softmax(dim=1)
        )
    def forward(self, p_low, p_high, p_raw):
        concat_features = torch.cat([p_low, p_high, p_raw], dim=1)
        weights = self.weight_net(concat_features)
        return weights[:, 0:1] * p_low + weights[:, 1:2] * p_high + weights[:, 2:3] * p_raw

class Patcher(nn.Module):
    def __init__(self, rescale=True):
        super().__init__()
        self.register_buffer("wavelets", torch.tensor([0.7071067811865476, 0.7071067811865476]), persistent=False)
        self.register_buffer("_arange", torch.arange(2), persistent=False)
        self.rescale = rescale
    def forward(self, x):
        dtype, g = x.dtype, x.shape[1]
        h = self.wavelets
        hl = h.flip(0).reshape(1, 1, -1).repeat(g, 1, 1).to(dtype=dtype)
        hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1).to(dtype=dtype)
        xl = F.conv3d(x, hl.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1))
        xh = F.conv3d(x, hh.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1))
        xll = F.conv3d(xl, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xlh = F.conv3d(xl, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xhl = F.conv3d(xh, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xhh = F.conv3d(xh, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xlll = F.conv3d(xll, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xllh = F.conv3d(xll, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xlhl = F.conv3d(xlh, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xlhh = F.conv3d(xlh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhll = F.conv3d(xhl, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhlh = F.conv3d(xhl, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhhl = F.conv3d(xhh, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhhh = F.conv3d(xhh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        out = torch.cat([xlll, xllh, xlhl, xlhh, xhll, xhlh, xhhl, xhhh], dim=1)
        if self.rescale: out = out * (2 * torch.sqrt(torch.tensor(2.0)))
        return out

class WaveletEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 16, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 128, 2, stride=2)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv3d(self.in_planes, planes * block.expansion, 1, stride, bias=False), nn.BatchNorm3d(planes * block.expansion))
        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks): layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))

class ARFFC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, 1)
        self.conv3x3 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.conv5x5 = nn.Conv3d(in_channels, out_channels, 5, padding=2)
        self.spatial_attn = nn.Conv3d(2, 1, 7, padding=3, bias=False)
        self.sigmoid, self.relu = nn.Sigmoid(), nn.ReLU()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.spatial_attn(torch.cat([avg_out, max_out], dim=1)))
        c1 = self.relu(self.conv1x1(x)) * attn
        c3 = self.relu(self.conv3x3(x)) * attn
        c5 = self.relu(self.conv5x5(x)) * attn
        return c1 + c3 + c5

