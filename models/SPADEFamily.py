
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()

        param_free_norm_type = 'instance'
        ks = 3

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, cs):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        cs = F.interpolate(cs, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(cs)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, style_nc, weight):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        self.W = weight
        # create conv layers
        self.conv_s1 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_e1 = nn.Conv2d(fmiddle, fmiddle, kernel_size=3, padding=1)
        self.conv_e2 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)

        self.conv_s2 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_p1 = nn.Conv2d(fmiddle, fmiddle, kernel_size=3, padding=1)
        self.conv_p2 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)

        # define normalization layers
        label_nc = 1
        self.norm_s1 = SPADE(fin, style_nc)
        self.norm_e1 = SPADE(fmiddle, label_nc)
        self.norm_e2 = SPADE(fmiddle, label_nc)
        label_nc = 18
        self.norm_s2 = SPADE(fin, style_nc)
        self.norm_p1 = SPADE(fmiddle, label_nc)
        self.norm_p2 = SPADE(fmiddle, label_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, style, content_edge, content_pose):

        ex = self.conv_s1(self.actvn(self.norm_s1(x, style)))
        ex = self.conv_e1(self.actvn(self.norm_e1(ex, content_edge)))
        ex = self.conv_e2(self.actvn(self.norm_e2(ex, content_edge)))

        px = self.conv_s2(self.actvn(self.norm_s2(x, style)))
        px = self.conv_p1(self.actvn(self.norm_p1(px, content_pose)))
        px = self.conv_p2(self.actvn(self.norm_p2(px, content_pose)))

        if self.W > 0:
            out = ex * self.W + px * (1 - self.W)
        else:
            out = ex + px
        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
