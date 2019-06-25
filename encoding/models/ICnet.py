import torch 
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseNet
from ..nn import PyramidPooling 

def get_interp_size(x, s_factor=1, z_factor=1): 
    r"""
    Parameters
    ----------
    s_factor: shrink factor 
    z_factor: zoom factor
    """
    h, w = x.size[2:]

    ih = (h-1)/s_factor + 1 
    iw = (w-1)/s_factor + 1

    ih = ih + (ih-1)*(z_factor-1)
    iw = iw + (iw-1)*(z_factor-1)

    return (int(ih), int(iw))

class BottleNeck(nn.Module): 
    """ResNet Bottleneck
    """
    def __init__(self, in_channels, mid_channels, out_channels, stride, dilation=1): 
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=stride, 
                               padding=0, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels) 
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, 
                               padding=1, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, 
                               padding=0, dilation=dilation, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                      stride=stride, padding=0, dilation=1, bias=False), 
                                      nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU() 

    def forward(self, x): 
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.residual(x)+self.bn3(self.conv3(out)))

        return out 

class CascadeFeatureFusion(nn.Module): 
    """
    Parameter: 
    ----------
    n_classes: n classes of dataset 
    low_inchannels: channels of input feature map from low resolution branch 
    high_inchannels: channels of input feature map from high resolution branch
    out_channels: channels of output feature map  
    """
    def __init__(self, n_classes, low_inchannels, high_inchannels, out_channels, norm_layer=nn.BatchNorm2d, 
                 up_kwargs): 
        super().__init__() 
        self._up_kwargs = up_kwargs

        self.low_dilated_conv = nn.Sequential( 
            nn.Conv2d(low_inchannels, out_channels, kernel_size=3, stride=1,
                      padding=2, dilation=2, bias=False),
            norm_layer(out_channels)
        )
        self.high_conv = nn.Sequential( 
            nn.Conv2d(high_inchannels, out_channels, kernel_size=1, stride=1,
                      padding=0, bias=False),
            norm_layer(out_channels)
        )

        self.low_classifier = nn.Conv2d(low_inchannels, n_classes, kernel_size=1, 
                                        stride=1, padding=0)
    
    def forward(self, x_low, x_high): 
        x = F.interpolate(x, size=get_interp_size(x_low, s_factor=2), **self._up_kwargs)

        low_feature = self.low_dilated_conv(x) 
        high_feature = self.high_conv(x)
        fused = F.relu(low_feature+high_feature) 

        low_logits = self.low_classifier(x) 

        return fused, low_logits

class LowResolutionBranch(nn.Module): 
    r""" 
    Parameters
    ----------
    norm_layer: nn.BatchNorm2d or SynBatchNorm2d
    """
    def __init__(self, blocks, norm_layer=nn.BatchNorm2d, up_kwargs):
        super().__init__()
        self._up_kwargs = up_kwargs
        
        self.conv1_1 = nn.Sequential( 
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False), 
            norm_layer(32), 
            nn.ReLU()
        )
        self.conv1_2 = nn.Sequential( 
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False), 
            norm_layer(32), 
            nn.ReLU()
        )
        self.conv1_3 = nn.Sequential( 
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False), 
            norm_layer(64), 
            nn.ReLU()
        )

        self.res2 = self._make_layers(blocks[0], 64, 32, 128, 1, 1, norm_layer)
        self.res3_stride = self._make_layers(blocks[1], 128, 64, 256, 2, 1, norm_layer, keep='conv')
    
    def _make_layers(self, block, in_channels, mid_channels, out_channels, stride, 
                     dilation=1, norm_layer=nn.BatchNorm2d, keep='all'): 
        layers = []
        if keep in ['all', 'conv']:
            layers.append(BottleNeck(in_channels, mid_channels, out_channels, stride, 
                                    dilation, norm_layer))
        if keep in ['all', 'residual']:
            for _ in range(1, block): 
                layers.append(BottleNeck(out_channels, mid_channels, out_channels, stride, 
                                        dilation, norm_layer))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.interpolate(x, size=get_interp_size(x, s_factor=2), **self._up_kwargs)

        out = self.conv1_3(self.conv1_2(self.conv1_1(x)))
        out = F.max_pool2d(out, 3, 2, 1)
        out = self.res2(out)
        out = self.res3_stride(out)
        
        return out

class MidResolutionBranch(nn.Module): 
    r""" 
    Parameters
    ----------
    norm_layer: nn.BatchNorm2d or SynBatchNorm2d
    """
    def __init__(self, blocks, norm_layer=nn.BatchNorm2d, up_kwargs): 
        super().__init__()
        self._up_kwargs = up_kwargs

        self.res3_identity = self._make_layers(blocks[1], 128, 64, 256, 2, 1, keep='residual')
        self.res4 = self._make_layers(blocks[2], 256, 128, 512, 1, 2)
        self.res5 = self._make_layers(blocks[3], 512, 256, 1024, 1, 4)
        self.pyramid_pooling = PyramidPooling(1024, norm_layer, **self._up_kwargs)
        self.shrink = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1, stride=1, 
                                              padding=0, bias=False), 
                                    norm_layer(256), 
                                    nn.ReLU())

    def _make_layers(self, block, in_channels, mid_channels, out_channels, stride, 
                     dilation=1, norm_layer=nn.BatchNorm2d, keep='all'): 
        layers = []
        if keep in ['all', 'conv']:
            layers.append(BottleNeck(in_channels, mid_channels, out_channels, stride, 
                                    dilation, norm_layer))
        if keep in ['all', 'residual']:
            for _ in range(1, block): 
                layers.append(BottleNeck(out_channels, mid_channels, out_channels, stride, 
                                        dilation, norm_layer))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.interpolate(x, size=get_interp_size(x, s_factor=2), **self._up_kwargs)

        out = self.res3_identity(x)
        out = self.res4(out)
        out = self.res5(out)
        out = self.pyramid_pooling(out)
        out = self.shrink(out)
        
        return out

class HighResolutionBranch(nn.Module): 
    r""" 
    Parameters
    ----------
    norm_layer: nn.BatchNorm2d or SynBatchNorm2d
    """
    def __init__(self, norm_layer, up_kwargs):
        super().__init__() 

        self.conv1_1 = nn.Sequential( 
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False), 
            norm_layer(32), 
            nn.ReLU()
        )
        self.conv1_2 = nn.Sequential( 
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False), 
            norm_layer(32), 
            nn.ReLU()
        )
        self.conv1_3 = nn.Sequential( 
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), 
            norm_layer(64), 
            nn.ReLU()
        )

    def forward(self, x): 
        out = self.conv1_3(self.conv1_2(self.conv1_1(x)))

        return out 

class ICnet(nn.Module): 
    r"""
    
    """
    def __init__(self, n_classes, blocks=[3,4,6,3], up_kwargs): 
        super().__init__()
        self._up_kwargs = up_kwargs

        # produce 1/16 feature maps
        self.branch1 = LowResolutionBranch(blocks, norm_layer, **self._up_kwargs)
        # produce 1/8 feature maps 
        self.branch2 = MidResolutionBranch(blocks, norm_layer, **self._up_kwargs)
        # produce 1/4 feature maps
        self.branch3 =  HighResolutionBranch(blocks, norm_layer, **self._up_kwargs)

        # fuse low branch and mid branch 
        self.cff_lm = CascadeFeatureFusion(n_classes, 256, 256, 128, norm_layer, **self._up_kwargs)
        # fuse mid branch and high branch
        self.cff_mh = CascadeFeatureFusion(n_classes, 128, 64, 128, norm_layer, **self._up_kwargs)
        # classification 
        self.classifier = nn.Conv2d(128, n_classes, kernel_size=1, stride=1, 
                                    padding=0)
    
    def forward(self, x): 
        x_branch1 = self.branch1(x)
        x_branch2 = self.branch2(x_branch1)

        x_branch3 = self.branch3(x) 

        x_branch12, low_logits = self.cff_lm(x_branch1, x_branch2)
        x_branch23, mid_logits = self.cff_mh(x_branch12, x_branch3)

        x_branch23 = F.interpolate(x_branch23, size=get_interp_size(x_branch23, z_factor=2), 
                                   mode='bilinear', align_corners=True)
        high_logits = self.classifier(x_branch23)

        return (high_logits, mid_logits, low_logits)