import torch.nn as nn
import functools
import torch
import functools
import torch.nn.functional as F
from torch.autograd import Variable
from .vgg import VGG
import os
from models.networks import ResnetGenerator
import torchvision.models.vgg as models
from models.SPADEFamily import SPADEResnetBlock
from util.util import multi_gpus_load_dict

class CSGenerator(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, opt, input_dim, dim, style_dim, n_downsample, n_res, mlp_dim, activ='relu', pad_type='reflect'):
        super(CSGenerator, self).__init__()
        self.opt = opt
        nf = 64

        # content encoder
        input_dim = 18+18+1
        output_nc = 1
        self.edge_generator = EdgeContentEncoder(input_dim, output_nc)

        # style encoder
        input_dim = 3
        self.enc_style = SPADEStyleEncoder(input_dim, dim, norm='none', activ=activ, pad_type=pad_type)
        self.sh, self.sw = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 8 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(18, 8 * nf, 3, padding=1)

        # fin, fout, style_nc
        self.head_0 = SPADEResnetBlock(8 * nf, 8 * nf, style_dim, opt.content_weight)

        self.G_middle_0 = SPADEResnetBlock(8 * nf, 8 * nf, style_dim, opt.content_weight)
        self.G_middle_1 = SPADEResnetBlock(8 * nf, 8 * nf, style_dim, opt.content_weight)

        self.up_0 = SPADEResnetBlock(8 * nf, 8 * nf, style_dim, opt.content_weight)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, style_dim, opt.content_weight)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, style_dim, opt.content_weight)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, style_dim, opt.content_weight)

        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    def forward(self, input_P1, input_EP1, input_BP2, input_BEP2, input_EP2):
        fake_EP2 = self.edge_generator(input_EP1, input_BP2, input_BEP2)
        # reconstruct an image
        style = self.enc_style(input_P1)

        # finally content_edge = fake_EP2
        content_edge = fake_EP2
        content_pose = input_BP2

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            z = torch.randn(style.size(0), self.opt.z_dim,
                            dtype=torch.float32, device=style.get_device())
            x = self.fc(z)
            x = x.view(-1, 8 * self.opt.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(content_pose, size=(self.sh, self.sw))
            x = self.fc(x)
        x = self.head_0(x, style, content_edge, content_pose)

        x = F.interpolate(x, size=(self.compute_up_size(x)), mode='bilinear', align_corners=False)
        x = self.G_middle_0(x, style, content_edge, content_pose)
        x = self.G_middle_1(x, style, content_edge, content_pose)

        x = F.interpolate(x, size=(self.compute_up_size(x)), mode='bilinear', align_corners=False)
        x = self.up_0(x, style, content_edge, content_pose)
        x = F.interpolate(x, size=(self.compute_up_size(x)), mode='bilinear', align_corners=False)
        x = self.up_1(x, style, content_edge, content_pose)
        x = F.interpolate(x, size=(self.compute_up_size(x)), mode='bilinear', align_corners=False)
        x = self.up_2(x, style, content_edge, content_pose)
        x = F.interpolate(x, size=(self.compute_up_size(x)), mode='bilinear', align_corners=False)
        x = self.up_3(x, style, content_edge, content_pose)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)
        return fake_EP2, x

    def compute_latent_vector_size(self, opt):

        sh = opt.fineSize // (2**opt.G_n_upampling)
        sw = round(sh / (256.0/176.0))

        return sh, sw

    def compute_up_size(self, x):

        sh = x.size(2) * 2
        sw = round(sh / (256.0/176.0))

        return (sh, sw)

    def decode(self, content, style):
        # decode content and style codes to an image
        adain_params = self.mlp(style)

        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params

class SPADEStyleEncoder(nn.Module):
    def __init__(self, input_dim, dim, norm, activ, pad_type):
        super(SPADEStyleEncoder, self).__init__()

        self.head_0 = ResidualConv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)

        self.down_0 = ResidualConv2dBlock(dim * 1, dim * 2, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        self.down_1 = ResidualConv2dBlock(dim * 2, dim * 4, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        self.down_2 = ResidualConv2dBlock(dim * 4, dim * 8, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        self.down_3 = ResidualConv2dBlock(dim * 8, dim * 8, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        self.down_4 = ResidualConv2dBlock(dim * 8, dim * 8, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)

        self.model = []
        self.model += [self.head_0]
        self.model += [self.down_0]
        self.model += [self.down_1]
        self.model += [self.down_2]
        self.model += [self.down_3]
        self.model += [self.down_4]

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        out = self.model(x)
        return out


class EdgeContentEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EdgeContentEncoder, self).__init__()
        # self.vgg = models.vgg19(pretrained=True).features
        self.model_edge = ResnetGenerator(input_dim, output_dim)
        self.model_edge = multi_gpus_load_dict(self.model_edge, './checkpoints/scagan_pctnet/latest_netG.pth')

        for param in self.model_edge.parameters():
            param.requires_grad_(False)

    def forward(self, input_EP1, input_BP2, input_BEP2):
        return self.model_edge(input_EP1, input_BP2, input_BEP2)


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResidualConv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(ResidualConv2dBlock, self).__init__()
        mid_dim = max(input_dim, output_dim)
        self.conv1 = Conv2dBlock(input_dim, mid_dim, kernel_size, stride, padding, norm=norm, activation=activation, pad_type=pad_type)
        self.skip = Conv2dBlock(input_dim, output_dim, kernel_size, stride, padding, norm=norm, activation=activation, pad_type=pad_type)
        self.conv2 = Conv2dBlock(mid_dim, output_dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)

    def forward(self, x):
        skip = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + skip
        return x

class SKResidualConv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(SKResidualConv2dBlock, self).__init__()
        mid_dim = max(input_dim, output_dim)
        self.conv1 = Conv2dBlock(input_dim, mid_dim, kernel_size, stride, padding, norm=norm, activation=activation, pad_type=pad_type)
        self.skip = Conv2dBlock(input_dim, output_dim, kernel_size, stride, padding, norm=norm, activation=activation, pad_type=pad_type)
        self.conv2 = Conv2dBlock(mid_dim, output_dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)

        self.gconv1 = nn.Sequential(
                nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1, groups=8),
                # nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=False)
            )

        self.gconv_skip = nn.Sequential(
                nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1, groups=8),
                # nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=False)
            )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        dim = int(output_dim / 2)
        self.fc = nn.Linear(output_dim, dim)
        self.fcs = nn.ModuleList([])

        for i in range(2):
            self.fcs.append(
                nn.Linear(dim, output_dim)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        skip = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        # x = x + skip

        x = self.gconv1(x).unsqueeze_(dim=1)
        skip = self.gconv_skip(skip).unsqueeze_(dim=1)
        g_conv = torch.cat([x, skip],dim=1)

        fea_u = torch.sum(g_conv, dim=1)
        # fea_s = self.gap(fea_u).squeeze_()
        fea_s = fea_u.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)

        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (g_conv * attention_vectors).sum(dim=1)
        return fea_v


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out