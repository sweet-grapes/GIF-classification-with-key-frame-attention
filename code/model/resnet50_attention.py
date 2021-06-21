from .resnets import resnet18, resnet34, resnet50, resnet101, resnet152
import torch.nn as nn
import torch.nn.functional as F


class SAttention(nn.Module):
    """
    sa = SAttention(2048)
    print(sa.forward(torch.randn(2*32, 2048, 32, 32), torch.randn(2*32, 73)).shape)
    """
    def __init__(self, num_channels, num_frames=32, reduction=16):
        super(SAttention, self).__init__()
        self.num_frames = num_frames
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(num_channels // reduction, 1, bias=False)
        )

    def forward(self, x, z):
        b, c, h, w = x.size()               # (B*T, C, H, W)
        x = self.avg_pool(x).view(b, c)     # (B*T, C)
        y = self.fc(x)                      # (B*T, 1)
        y = y.view(-1, self.num_frames)     # (B, T)
        att = F.softmax(y, dim=1)
        # T帧分类结果聚合
        z = z.view(-1, self.num_frames, z.size(-1)) * att.unsqueeze(-1)
        z = z.sum(1)                        # (B, T, C)
        return z, y


class DCEHAAModel(nn.Module):
    def __init__(self, backbone='resnet50', num_classes=73, num_frames=32):
        super(DCEHAAModel, self).__init__()
        if backbone == 'resnet50':
            self.backbone = resnet50(pretrained=True, num_classes=num_classes)
        else:
            raise NotImplementedError
        self.ts_attn = SAttention(2048, num_frames=num_frames)
        self.bn_frozen_layers = []
        self.fixed_layers = [self.backbone.conv1, self.backbone.bn1]
        self._fix_running_stats(self.backbone, fix_params=True)

    def train(self, mode=True):
        super().train(mode)
        for l in self.fixed_layers:
            for p in l.parameters():
                p.requires_grad = False
        for bn_layer in self.bn_frozen_layers:
            bn_layer.eval()

    def _fix_running_stats(self, layer, fix_params=False):
        if isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            self.bn_frozen_layers.append(layer)
            if fix_params and not layer in self.fixed_layers:
                self.fixed_layers.append(layer)
        elif isinstance(layer, list):
            for m in layer:
                self._fix_running_stats(m, fix_params)
        else:
            for m in layer.children():
                self._fix_running_stats(m, fix_params)

    def forward(self, x):
        # (B, T, 3, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        frame_preds, feat = self.backbone.forward_feats(x)
        # T帧分类结果聚合
        pred, atte = self.ts_attn(feat, frame_preds)
        return pred, atte, frame_preds
