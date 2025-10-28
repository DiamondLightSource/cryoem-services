"""
Created on Thu Jun 29 07:58:32 2023

@author: jaehoon cha
@email: jaehoon.cha@stfc.ac.uk
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from escnn import gspaces, nn as enn


def stretch(X, alpha, gamma, beta, moving_mag, moving_min, eps, momentum, training):
    """
    the code is based on the batch normalization in
    http://preview.d2l.ai/d2l-en/master/chapter_convolutional-modern/batch-norm.html
    """
    if not training:
        X_hat = (X - moving_min) / moving_mag
    else:
        assert len(X.shape) in (2, 4)
        min_ = X.min(dim=0)[0]
        max_ = X.max(dim=0)[0]

        mag_ = max_ - min_
        X_hat = (X - min_) / mag_
        moving_mag = momentum * moving_mag + (1.0 - momentum) * mag_
        moving_min = momentum * moving_min + (1.0 - momentum) * min_
    Y = (X_hat * gamma * alpha) + beta
    return Y, moving_mag.data, moving_min.data


class Stretch(nn.Module):
    """
    the code is based on the batch normalization in
    http://preview.d2l.ai/d2l-en/master/chapter_convolutional-modern/batch-norm.html
    """

    def __init__(self, num_features, num_dims, alpha):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.alpha = alpha
        self.gamma = nn.Parameter(0.01 * torch.ones(shape))
        self.beta = nn.Parameter(np.pi * torch.ones(shape))
        self.register_buffer("moving_mag", 1.0 * torch.ones(shape))
        self.register_buffer("moving_min", np.pi * torch.ones(shape))

    def forward(self, X):
        if self.moving_mag.device != X.device:
            self.moving_mag = self.moving_mag.to(X.device)
            self.moving_min = self.moving_min.to(X.device)
        Y, self.moving_mag, self.moving_min = stretch(
            X,
            self.alpha,
            self.gamma,
            self.beta,
            self.moving_mag,
            self.moving_min,
            eps=1e-5,
            momentum=0.99,
            training=self.training,
        )
        return Y


class Conv_BN_Relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding
            )
        )
        self.layers.append(nn.BatchNorm2d(out_channels))
        self.layers.append(nn.ReLU())

    def forward(self, x):
        for idx in range(len(self.layers)):
            x = self.layers[idx](x)
        return x


class ConvT_BN(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, output_padding
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            )
        )
        self.layers.append(nn.BatchNorm2d(out_channels))

    def forward(self, x):
        for idx in range(len(self.layers)):
            x = self.layers[idx](x)
        return x


class ConvT_BN_Relu(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, output_padding
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            )
        )
        self.layers.append(nn.BatchNorm2d(out_channels))
        self.layers.append(nn.ReLU())

    def forward(self, x):
        for idx in range(len(self.layers)):
            x = self.layers[idx](x)
        return x


class Up_Block(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, output_padding
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.CBR0 = ConvT_BN_Relu(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.CBR1 = Conv_BN_Relu(
            out_channels, out_channels, kernel_size, stride=1, padding="same"
        )

        self.short_up = ConvT_BN(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

    def forward(self, x):
        residual = x
        out = self.CBR0(x)
        out = self.CBR1(out)
        if (self.in_channels != self.out_channels) or (self.stride > 1):
            residual = self.short_up(x)
        y = out + residual
        return y


class Equi_layer(nn.Module):
    def __init__(self, r2_act, hidden_in, hidden_out, h, k_size=3, p_size=1, equi=True):
        super().__init__()
        self.hidden_in = hidden_in
        self.hidden_out = hidden_out

        self.in_type = enn.FieldType(r2_act, self.hidden_in * [r2_act.trivial_repr])
        self.out_type = enn.FieldType(r2_act, self.hidden_out * [r2_act.regular_repr])
        self.equi = equi
        self.h = h

        self.layer = enn.SequentialModule(
            enn.MaskModule(self.in_type, self.h, margin=1),
            enn.R2Conv(
                self.in_type,
                self.out_type,
                kernel_size=k_size,
                padding=p_size,
                bias=False,
            ),
            enn.InnerBatchNorm(self.out_type),
            enn.ReLU(self.out_type, inplace=True),
        )

    def forward(self, input):
        x = enn.GeometricTensor(input, self.in_type)
        x = self.layer(x)  # B, H * R, M, N

        if self.equi:
            x = x.tensor
            x = x.view(-1, self.hidden_out, 8, self.h, self.h)
            x = x.sum(2)
        return x


class Decoder_ResNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.input_channel, self.hidden_dims = params

        self.blockT0 = Up_Block(self.hidden_dims[4], self.hidden_dims[3], 3, 2, 1, 1)
        self.blockT1 = Up_Block(self.hidden_dims[3], self.hidden_dims[2], 3, 2, 1, 1)
        self.blockT2 = Up_Block(self.hidden_dims[2], self.hidden_dims[1], 3, 2, 1, 1)
        self.blockT3 = Up_Block(self.hidden_dims[1], self.hidden_dims[0], 3, 2, 1, 1)
        self.out_conv = nn.Conv2d(self.input_channel, self.input_channel, 3, padding=1)

    def forward(self, x):
        x = self.blockT0(x)
        x = self.blockT1(x)
        x = self.blockT2(x)
        x = self.blockT3(x)
        x = self.out_conv(x)
        return x


class EIAE(nn.Module):
    def __init__(self, alpha, input_dims, hidden_dims, lat_dim=2):
        super().__init__()
        n_rot = 8
        self.r2_act = gspaces.rot2dOnR2(N=n_rot)

        self.hidden_dims = hidden_dims
        self.c, self.m, self.n = input_dims  # self.input_dim
        self.reduced_dim_m = int(np.ceil(self.m / (2 ** (len(self.hidden_dims) - 1))))
        self.reduced_dim_n = int(np.ceil(self.n / (2 ** (len(self.hidden_dims) - 1))))
        self.fc_hidden = 128
        self.latent_dim = lat_dim
        self.alpha = alpha

        self.share1 = Equi_layer(
            self.r2_act, self.hidden_dims[0], self.hidden_dims[1], 64, 7, 3, True
        )

        self.pool1 = nn.MaxPool2d(2)

        self.share2 = Equi_layer(
            self.r2_act, self.hidden_dims[1], self.hidden_dims[2], 32, 3, 1, True
        )

        self.pool2 = nn.MaxPool2d(2)

        self.share3 = Equi_layer(
            self.r2_act, self.hidden_dims[2], self.hidden_dims[3], 16, 3, 1, True
        )

        self.pool3 = nn.MaxPool2d(2)

        self.share4 = Equi_layer(
            self.r2_act, self.hidden_dims[3], self.hidden_dims[4], 8, 3, 1, False
        )

        out_type = self.share4.out_type
        self.pool4 = enn.SequentialModule(
            enn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        in_type = self.share4.out_type
        # the output type of the third convolution layer are 48 regular feature fields of C8
        out_type = enn.FieldType(
            self.r2_act, self.hidden_dims[4] * 4 * [self.r2_act.regular_repr]
        )
        self.enc_inv1 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ELU(out_type, inplace=True),
        )

        self.spool = nn.AdaptiveAvgPool2d(1)
        self.gpool = enn.GroupPooling(out_type)  # rot_pooling

        self.to_lat = nn.Linear(self.hidden_dims[4] * 4, self.latent_dim)
        self.strecth = Stretch(self.latent_dim, 2, self.alpha)

        self.enc_equi1 = nn.Sequential(
            nn.Conv2d(
                self.hidden_dims[4] * n_rot,
                self.hidden_dims[4],
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(self.hidden_dims[4]),
            nn.LeakyReLU(),
        )

        self.et_fc = nn.Linear(
            self.hidden_dims[4] * self.reduced_dim_m * self.reduced_dim_n,
            self.fc_hidden,
        )
        self.to_et = nn.Linear(self.fc_hidden, 3)

        self.to_dec = nn.Linear(
            in_features=self.latent_dim * 2, out_features=self.fc_hidden
        )
        self.de_fc = nn.Linear(
            self.fc_hidden,
            self.hidden_dims[-1] * self.reduced_dim_m * self.reduced_dim_n,
        )

        self.decoder_seq = Decoder_ResNet((self.hidden_dims[0], self.hidden_dims))

        self.pad = 10

        xgrid = np.linspace(-1, 1, self.m + self.pad * 2)
        ygrid = np.linspace(-1, 1, self.n + self.pad * 2)
        x0, x1 = np.meshgrid(xgrid, ygrid)
        grid = np.stack([x0.ravel(), x1.ravel()], 1)
        # self.grid = grid.reshape(self.m, self.n, 2)
        self.grid = torch.from_numpy(grid).float()  ### make coord

    def sample(self, device, num_samples=100, z=None):
        if z is None:
            z = torch.randn(num_samples, self.latent_dim).to(device)

        c = torch.cat((torch.cos(2 * np.pi * z), torch.sin(2 * np.pi * z)), 0)
        print(c)
        c = c.T.reshape(self.latent_dim * 2, -1).T

        print(c)
        print(c.shape)

        samples = self.decode(c)
        return samples

    def encode(self, x):
        x = self.share1(x)
        x = self.pool1(x)

        x = self.share2(x)
        x = self.pool2(x)

        x = self.share3(x)
        x = self.pool3(x)

        x = self.share4(x)
        x = self.pool4(x)

        x_inv = self.enc_inv1(x)
        x_inv = self.gpool(x_inv)
        x_inv = x_inv.tensor  ###to Tensor
        x_inv = self.spool(x_inv)
        x_inv = torch.flatten(x_inv, start_dim=1)
        x_inv = self.to_lat(x_inv)
        x_inv = self.strecth(x_inv)

        x_equi = x.tensor  ###to Tensor
        x_equi = self.enc_equi1(x_equi)
        x_equi = torch.flatten(x_equi, start_dim=1)

        x_equi = self.et_fc(x_equi)
        x_equi = self.to_et(x_equi)

        return x_inv, x_equi

    def decode(self, x):
        x = nn.LeakyReLU()(self.to_dec(x))
        x = nn.LeakyReLU()(self.de_fc(x))

        x = x.view(-1, self.hidden_dims[-1], self.reduced_dim_m, self.reduced_dim_n)

        x = self.decoder_seq(x)

        return x

    def reparameterize(self, z):
        diff = torch.abs(z - z.unsqueeze(axis=1))
        none_zeros = torch.where(diff == 0.0, torch.tensor([100.0]).to(z.device), diff)
        z_scores, _ = torch.min(none_zeros, axis=1)
        std = torch.normal(mean=0.0, std=1.0 * z_scores).to(z.device)
        s = z + std
        c = torch.cat((torch.cos(2 * np.pi * s), torch.sin(2 * np.pi * s)), 0)
        c = c.T.reshape(self.latent_dim * 2, -1).T
        return c

    def get_recon(self, x, z_equi):
        b = len(x)
        rot_mat = (
            torch.cat(
                (
                    torch.cos(z_equi[:, 0]),
                    -torch.sin(z_equi[:, 0]),
                    torch.sin(z_equi[:, 0]),
                    torch.cos(z_equi[:, 0]),
                ),
                axis=0,
            )
            .view(4, -1)
            .T.view(-1, 2, 2)
        )

        dxy = z_equi[:, 1:]

        recon_pad = F.pad(x, (self.pad, self.pad, self.pad, self.pad), "replicate")

        grid = self.grid.expand(b, recon_pad.size()[-2] * recon_pad.size()[-1], 2).to(
            z_equi.device
        )
        grid = grid - dxy.unsqueeze(1)  # translate coordinates
        grid = torch.bmm(grid, rot_mat)  # rotate coordinates by theta
        grid = grid.view(-1, recon_pad.size()[-2], recon_pad.size()[-1], 2)

        reconstruction = F.grid_sample(recon_pad, grid, padding_mode="border")

        reconstruction = reconstruction[
            :, :, self.pad : -self.pad, self.pad : -self.pad
        ]
        return reconstruction

    def latent(self, input):
        z_inv, z_equi = self.encode(input)

        theta = z_equi[:, 0]
        dxy = z_equi[:, 1:]
        return z_inv, theta, dxy

    def reconstr(self, input: torch.Tensor):
        z_inv, z_equi = self.encode(input)

        c_inv = torch.cat(
            (torch.cos(2 * np.pi * z_inv), torch.sin(2 * np.pi * z_inv)), 0
        )
        c_inv = c_inv.T.reshape(self.latent_dim * 2, -1).T

        inv_recon = self.decode(c_inv)

        reconstr = self.get_recon(inv_recon, z_equi)

        return reconstr

    def forward(self, input: torch.Tensor):
        z_inv, z_equi = self.encode(input)

        c_inv = self.reparameterize(z_inv)
        inv_recon = self.decode(c_inv)

        reconstr = self.get_recon(inv_recon, z_equi)

        return reconstr, c_inv, z_inv, z_equi

    def forward_inv(self, input: torch.Tensor):
        z_inv, z_equi = self.encode(input)

        c_inv = torch.cat(
            (torch.cos(2 * np.pi * z_inv), torch.sin(2 * np.pi * z_inv)), 0
        )
        c_inv = c_inv.T.reshape(self.latent_dim * 2, -1).T

        reconstr = self.decode(c_inv)

        return reconstr

    def forward_lat(self, z):
        dxy, theta, z_inv = z[:, :2], z[:, 2], z[:, 3:]
        z_equi = torch.cat((theta.reshape(-1, 1), dxy), axis=-1)

        c_inv = torch.cat(
            (torch.cos(2 * np.pi * z_inv), torch.sin(2 * np.pi * z_inv)), 0
        )
        c_inv = c_inv.T.reshape(self.latent_dim * 2, -1).T

        inv_recon = self.decode(c_inv)

        reconstr = self.get_recon(inv_recon, z_equi)

        return reconstr
