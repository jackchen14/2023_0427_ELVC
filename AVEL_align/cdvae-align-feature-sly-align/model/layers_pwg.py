"""Residual block module in WaveNet.
This code is modified from https://github.com/r9y9/wavenet_vocoder.
"""
import numpy as np
import math

import torch
import torch.nn.functional as F


class Conv1d(torch.nn.Conv1d):
    """Conv1d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv1d module."""
        super(Conv1d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        """Reset parameters."""
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)


class Conv1d1x1(Conv1d):
    """1x1 Conv1d with customized initialization."""

    def __init__(self, in_channels, out_channels, bias):
        """Initialize 1x1 Conv1d module."""
        super(Conv1d1x1, self).__init__(in_channels, out_channels,
                                        kernel_size=1, padding=0,
                                        dilation=1, bias=bias)


class Conv2d(torch.nn.Conv2d):
    """Conv2d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv2d module."""
        super(Conv2d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        """Reset parameters."""
        self.weight.data.fill_(1. / np.prod(self.kernel_size))
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)


class ResidualBlock(torch.nn.Module):
    """Residual block module in WaveNet."""

    def __init__(self,
                 kernel_size=3,
                 residual_channels=64,
                 gate_channels=128,
                 skip_channels=64,
                 aux_channels=80,
                 dropout=0.0,
                 padding=None,
                 dilation=1,
                 bias=True,
                 use_causal_conv=False
                 ):
        """Initialize ResidualBlock module.
        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            residual_channels (int): Number of channels for residual connection.
            skip_channels (int): Number of channels for skip connection.
            aux_channels (int): Local conditioning channels i.e. auxiliary input dimension.
            dropout (float): Dropout probability.
            padding (int): Padding for convolution layers. If None, proper padding is
                computed depends on dilation and kernel_size.
            dilation (int): Dilation factor.
            bias (bool): Whether to add bias parameter in convolution layers.
            use_causal_conv (bool): Whether to use use_causal_conv or non-use_causal_conv convolution.
        """
        super(ResidualBlock, self).__init__()
        self.dropout = dropout
        if padding is None:
            # no future time stamps available
            if use_causal_conv:
                padding = (kernel_size - 1) * dilation
            else:
                assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
                padding = (kernel_size - 1) // 2 * dilation
        self.use_causal_conv = use_causal_conv

        # dilation conv
        self.conv = Conv1d(residual_channels, gate_channels, kernel_size,
                           padding=padding, dilation=dilation, bias=bias)

        # local conditioning
        if aux_channels > 0:
            self.conv1x1_aux = Conv1d1x1(aux_channels, gate_channels, bias=False)
        else:
            self.conv1x1_aux = None

        # conv output is split into two groups
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = Conv1d1x1(gate_out_channels, residual_channels, bias=bias)
        self.conv1x1_skip = Conv1d1x1(gate_out_channels, skip_channels, bias=bias)

    def forward(self, x, c):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, residual_channels, T).
            c (Tensor): Local conditioning auxiliary tensor (B, aux_channels, T).
        Returns:
            Tensor: Output tensor for residual connection (B, residual_channels, T).
            Tensor: Output tensor for skip connection (B, skip_channels, T).
        """
        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv(x)

        # remove future time steps if use_causal_conv conv
        x = x[:, :, :residual.size(-1)] if self.use_causal_conv else x

        # split into two part for gated activation
        splitdim = 1
        xa, xb = x.split(x.size(splitdim) // 2, dim=splitdim)

        # local conditioning
        if c is not None:
            assert self.conv1x1_aux is not None
            c = self.conv1x1_aux(c)
            ca, cb = c.split(c.size(splitdim) // 2, dim=splitdim)
            xa, xb = xa + ca, xb + cb

        x = torch.tanh(xa) * torch.sigmoid(xb)

        # for skip connection
        s = self.conv1x1_skip(x)

        # for residual connection
        x = (self.conv1x1_out(x) + residual) * math.sqrt(0.5)

        return x, s


class Stretch2d(torch.nn.Module):
    """Stretch2d module."""

    def __init__(self, x_scale, y_scale, mode="nearest"):
        """Initialize Stretch2d module.
        Args:
            x_scale (int): X scaling factor (Time axis in spectrogram).
            y_scale (int): Y scaling factor (Frequency axis in spectrogram).
            mode (str): Interpolation mode.
        """
        super(Stretch2d, self).__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.mode = mode

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, C, F, T).
        Returns:
            Tensor: Interpolated tensor (B, C, F * y_scale, T * x_scale),
        """
        return F.interpolate(
            x, scale_factor=(self.y_scale, self.x_scale), mode=self.mode)


class UpsampleNetwork(torch.nn.Module):
    """Upsampling network module."""

    def __init__(self,
                 upsample_scales,
                 upsample_activation="none",
                 upsample_activation_params={},
                 mode="nearest",
                 freq_axis_kernel_size=1,
                 use_causal_conv=False,
                 ):
        """Initialize upsampling network module.
        Args:
            upsample_scales (list): List of upsampling scales.
            upsample_activation (str): Activation function name.
            upsample_activation_params (dict): Arguments for specified activation function.
            mode (str): Interpolation mode.
            freq_axis_kernel_size (int): Kernel size in the direction of frequency axis.
        """
        super(UpsampleNetwork, self).__init__()
        self.use_causal_conv = use_causal_conv
        self.up_layers = torch.nn.ModuleList()
        for scale in upsample_scales:
            # interpolation layer
            stretch = Stretch2d(scale, 1, mode)
            self.up_layers += [stretch]

            # conv layer
            assert (freq_axis_kernel_size - 1) % 2 == 0, "Not support even number freq axis kernel size."
            freq_axis_padding = (freq_axis_kernel_size - 1) // 2
            kernel_size = (freq_axis_kernel_size, scale * 2 + 1)
            if use_causal_conv:
                padding = (freq_axis_padding, scale * 2)
            else:
                padding = (freq_axis_padding, scale)
            conv = Conv2d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
            self.up_layers += [conv]

            # nonlinear
            if upsample_activation != "none":
                nonlinear = getattr(torch.nn, upsample_activation)(**upsample_activation_params)
                self.up_layers += [nonlinear]

    def forward(self, c):
        """Calculate forward propagation.
        Args:
            c : Input tensor (B, C, T).
        Returns:
            Tensor: Upsampled tensor (B, C, T'), where T' = T * prod(upsample_scales).
        """
        c = c.unsqueeze(1)  # (B, 1, C, T)
        for f in self.up_layers:
            if self.use_causal_conv and isinstance(f, Conv2d):
                c = f(c)[..., :c.size(-1)]
            else:
                c = f(c)
        return c.squeeze(1)  # (B, C, T')


class ConvInUpsampleNetwork(torch.nn.Module):
    """Convolution + upsampling network module."""

    def __init__(self,
                 upsample_scales,
                 upsample_activation="none",
                 upsample_activation_params={},
                 mode="nearest",
                 freq_axis_kernel_size=1,
                 aux_channels=80,
                 aux_context_window=0,
                 use_causal_conv=False
                 ):
        """Initialize convolution + upsampling network module.
        Args:
            upsample_scales (list): List of upsampling scales.
            upsample_activation (str): Activation function name.
            upsample_activation_params (dict): Arguments for specified activation function.
            mode (str): Interpolation mode.
            freq_axis_kernel_size (int): Kernel size in the direction of frequency axis.
            aux_channels (int): Number of channels of pre-convolutional layer.
            aux_context_window (int): Context window size of the pre-convolutional layer.
            use_causal_conv (bool): Whether to use causal structure.
        """
        super(ConvInUpsampleNetwork, self).__init__()
        self.aux_context_window = aux_context_window
        self.use_causal_conv = use_causal_conv and aux_context_window > 0
        # To capture wide-context information in conditional features
        kernel_size = aux_context_window + 1 if use_causal_conv else 2 * aux_context_window + 1
        # NOTE(kan-bayashi): Here do not use padding because the input is already padded
        self.conv_in = Conv1d(aux_channels, aux_channels, kernel_size=kernel_size, bias=False)
        self.upsample = UpsampleNetwork(
            upsample_scales=upsample_scales,
            upsample_activation=upsample_activation,
            upsample_activation_params=upsample_activation_params,
            mode=mode,
            freq_axis_kernel_size=freq_axis_kernel_size,
            use_causal_conv=use_causal_conv,
        )

    def forward(self, c):
        """Calculate forward propagation.
        Args:
            c : Input tensor (B, C, T').
        Returns:
            Tensor: Upsampled tensor (B, C, T),
                where T = (T' - aux_context_window * 2) * prod(upsample_scales).
        Note:
            The length of inputs considers the context window size.
        """
        c_ = self.conv_in(c)
        c = c_[:, :, :-self.aux_context_window] if self.use_causal_conv else c_
        return self.upsample(c)


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        
        # Spectral convergence loss
        sc_loss = torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro") 
        # Log STFT magnitude loss
        mag_loss = F.l1_loss(torch.log(y_mag), torch.log(x_mag), reduction='sum')

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window="hann_window"):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss