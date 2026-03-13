# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
import math

import numpy as np
import torch
from torch import nn


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], 
                        dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(
                        channels, 
                        num_features // channels, 
                        kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)

class RNNEncoder(nn.Module):
    """An RNN encoder.

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        hidden_size (int): The hidden size of the RNN layer.
        num_layers (int): The number of layers in the RNN.
        dropout (float): The dropout probability for the RNN. Only applied if
            `num_layers` > 1. (default: 0.1)
        bidirectional (bool): Whether to use a bidirectional RNN. (default:
            False)
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.rnn = nn.RNN(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        out_features = hidden_size * (2 if bidirectional else 1)
        self.proj = nn.Linear(out_features, num_features)
        self.layer_norm = nn.LayerNorm(num_features)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x, _ = self.rnn(inputs)  # (T, N, hidden_size * num_directions)
        x = self.proj(x)  # (T, N, num_features)
        x = x + inputs  # Skip connection
        return self.layer_norm(x)  # (T, N, num_features)

class LSTMEncoder(nn.Module):
    """An LSTM encoder.

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        hidden_size (int): The hidden size of the LSTM layer.
        num_layers (int): The number of layers in the LSTM.
        dropout (float): The dropout probability for the LSTM. Only applied if
            `num_layers` > 1. (default: 0.1)
        bidirectional (bool): Whether to use a bidirectional LSTM. (default:
            False)
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        out_features = hidden_size * (2 if bidirectional else 1)
        self.proj = nn.Linear(out_features, num_features)
        self.layer_norm = nn.LayerNorm(num_features)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(inputs)  # (T, N, hidden_size * num_directions)
        x = self.proj(x)  # (T, N, num_features)
        x = x + inputs  # Skip connection
        return self.layer_norm(x)  # (T, N, num_features)

class GRUEncoder(nn.Module):
    """A GRU encoder.

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        hidden_size (int): The hidden size of the GRU layer.
        num_layers (int): The number of layers in the GRU.
        dropout (float): The dropout probability for the GRU. Only applied if
            `num_layers` > 1. (default: 0.1)
        bidirectional (bool): Whether to use a bidirectional GRU. (default:
            False)
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        out_features = hidden_size * (2 if bidirectional else 1)
        self.proj = nn.Linear(out_features, num_features)
        self.layer_norm = nn.LayerNorm(num_features)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x, _ = self.gru(inputs)  # (T, N, hidden_size * num_directions)
        x = self.proj(x)  # (T, N, num_features)
        x = x + inputs  # Skip connection
        return self.layer_norm(x)  # (T, N, num_features)

# class PositionalEncoding(nn.Module):
#     """A `torch.nn.Module` that adds sinusoidal positional encodings to an input
#     tensor of shape (T, N, num_features) as per "Attention is All You Need"
#     (https://arxiv.org/abs/1706.03762).

#     Args:
#         d_model (int): The number of expected features in the input (i.e., the
#             last dimension of the input tensor).
#         max_len (int): The maximum length of the input sequences. This
#             determines the size of the positional encoding buffer. (default:
#             5000)
#     """

#     def __init__(self, d_model: int, max_len: int = 5000):
#         super().__init__()
#         self.d_model = d_model

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
#         div_term = torch.exp(
#             torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
#         )
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer("pe", pe)  # (max_len, d_model)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # Expect x shape: (T, N, D)
#         if x.ndim != 3:
#             raise ValueError(f"PositionalEncoding expects a 3D tensor (T,N,D), got {x.ndim}D")
#         if x.size(2) != self.d_model:
#             raise ValueError(
#                 f"PositionalEncoding expected last dim {self.d_model}, got {x.size(2)}"
#             )

#         # Use a (T, 1, D) buffer so it broadcasts cleanly across batch dim (N)
#         pe = self.pe[: x.size(0)].unsqueeze(1).to(x.dtype)
#         return x + pe

class TransformerEncoder(nn.Module):
    """A Transformer encoder.

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        num_heads (int): The number of attention heads in the Transformer.
        num_layers (int): The number of layers in the Transformer.
        dropout (float): The dropout probability for the Transformer. (default:
            0.1)
    """

    def __init__(
        self,
        num_features: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(num_features, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.layer_norm = nn.LayerNorm(d_model)
    
    def position_encoding(self, inputs) -> torch.Tensor:
        num_features = inputs.shape[2]
        length = inputs.shape[0]

        pe = torch.zeros(length, num_features)
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, num_features, 2).float() * 
            (-math.log(10000.0) / num_features)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual shortcut around the encoder.
        """
        # Project each (T, num_features) slice to (T, d_model) and normalize prior to encoding.
        x = self.input_proj(inputs)
        x = self.input_norm(x)

        pe = self.position_encoding(x).unsqueeze(1).to(x.device)  # (T, 1, d_model)
        x = x + pe

        out = self.transformer_encoder(x)  # (T, N, d_model)
        out = out + x

        return self.layer_norm(out)  # (T, N, d_model)

# class EMGFilter(nn.Module):
#     """A `torch.nn.Module` that applies a bandpass FIR filter to EMG signals
#     using depthwise convolution. For an input of shape (T, N, C), the convolution is applied independently over each of the C channels."""

#     def __init__(
#             self, 
#             lowcut, 
#             highcut, 
#             fs, 
#             channels, 
#             taps=401):
#         super().__init__()

#         kernel = self.design_fir_bandpass(lowcut, highcut, fs, taps)
#         kernel = kernel.view(1, 1, taps).repeat(channels, 1, 1)  # (channels, 1, taps)

#         self.conv = nn.Conv1d(
#             in_channels=channels,
#             out_channels=channels,
#             kernel_size=taps,
#             padding=taps // 2,
#             groups=channels,  # depthwise convolution
#             bias=False,
#         )

#         with torch.no_grad():
#             self.conv.weight[:] = kernel

#     def forward(self, inputs: torch.Tensor) -> torch.Tensor:
#         # TNC -> NCT -> NfT -> NfT (after conv) -> NCT -> TNC
#         x = inputs.movedim(0, 1).movedim(2, 1)  # TNC -> NCT -> NfT
#         x = self.conv(x)  # NfT -> NfT
#         x = x.movedim(2, 1).movedim(1, 0)  # NfT -> NCT -> TNC
#         return x

#     def design_fir_bandpass(self, lowcut, highcut, fs, numtaps, device="cpu"):
#         nyq = 0.5 * fs
#         low = lowcut / nyq
#         high = highcut / nyq
        
#         n = np.arange(numtaps)
#         center = (numtaps - 1) / 2

#         def sinc(x):
#             return np.sinc(x)
        
#         h = (2 * high * sinc(2 * high * (n - center)) - 2 * low * sinc(2 * low * (n - center)))
#         window = np.hamming(numtaps)
        
#         h *= window
#         h /= np.sum(h)
#         h = torch.tensor(h, dtype=torch.float32, device=device)
        
#         return h
    
# class EMGFilter(nn.Module):
#     """A `torch.nn.Module` that applies a bandpass FIR filter to EMG signals
#     using depthwise convolution. For an input of shape (T, N, B, C), the
#     convolution is applied independently over each of the B*C channels."""

#     def __init__(
#         self,
#         lowcut,
#         highcut,
#         fs,
#         channels,
#         bands: int = 1,
#         taps: int = 401,
#     ):
#         super().__init__()

#         self.bands = bands
#         self.channels = channels
#         total_channels = bands * channels

#         kernel = self.design_fir_bandpass(lowcut, highcut, fs, taps)
#         kernel = kernel.view(1, 1, taps).repeat(total_channels, 1, 1)

#         self.conv = nn.Conv1d(
#             in_channels=total_channels,
#             out_channels=total_channels,
#             kernel_size=taps,
#             padding=taps // 2,
#             groups=total_channels,
#             bias=False,
#         )

#         with torch.no_grad():
#             self.conv.weight[:] = kernel

#     def forward(self, inputs: torch.Tensor) -> torch.Tensor:
#         # Expect (T, N, B, C)
#         T, N, B, C = inputs.shape
#         assert B == self.bands and C == self.channels

#         #   -> (N, B, C, T) -> (N, B*C, T)
#         x = inputs.permute(1, 2, 3, 0).reshape(N, B * C, T)

#         x = self.conv(x)  # (N, B*C, T)

#         # (N, B*C, T) -> (N, B, C, T) -> (T, N, B, C)
#         x = x.reshape(N, B, C, T).permute(3, 0, 1, 2)
#         assert x.shape == inputs.shape
        
#         return x

#     def design_fir_bandpass(self, lowcut, highcut, fs, numtaps, device="cpu"):
#         nyq = 0.5 * fs
#         low = lowcut / nyq
#         high = highcut / nyq
        
#         n = np.arange(numtaps)
#         center = (numtaps - 1) / 2

#         def sinc(x):
#             return np.sinc(x)
        
#         h = (2 * high * sinc(2 * high * (n - center)) - 2 * low * sinc(2 * low * (n - center)))
#         window = np.hamming(numtaps)
        
#         h *= window
#         h /= np.sum(h)
#         h = torch.tensor(h, dtype=torch.float32, device=device)
        
#         return h
    
# class Biquad(nn.Module):
#     def __init__(self):
#         super().__init__()

#         # learnable coefficients
#         self.b = nn.Parameter(torch.tensor([1.0, 0.0, 0.0]))
#         self.a = nn.Parameter(torch.tensor([0.0, 0.0]))

#     def forward(self, inputs: torch.Tensor) -> torch.Tensor:
#         # Expect (T, N, B, C)
#         if inputs.ndim != 4:
#             raise ValueError(f"Biquad.forward expects a 4D tensor (T, N, B, C), got {inputs.ndim}D")

#         T, N, B, C = inputs.shape

#         # Process each time series independently by collapsing batch/bands/channels.
#         x = inputs.permute(1, 2, 3, 0).reshape(N * B * C, T)
#         y = torch.zeros_like(x)

#         x1 = torch.zeros(N * B * C, device=inputs.device)
#         x2 = torch.zeros(N * B * C, device=inputs.device)

#         y1 = torch.zeros(N * B * C, device=inputs.device)
#         y2 = torch.zeros(N * B * C, device=inputs.device)

#         b0, b1, b2 = self.b
#         a1, a2 = self.a

#         a1_tanh = torch.tanh(a1)
#         a2_tanh = torch.tanh(a2)

#         for t in range(T):
#             xt = x[:, t]

#             yt = (
#                 b0 * xt
#                 + b1 * x1
#                 + b2 * x2
#                 - a1_tanh * y1
#                 - a2_tanh * y2
#             )

#             y[:, t] = yt

#             x2 = x1
#             x1 = xt

#             y2 = y1
#             y1 = yt

#         # Restore original (T, N, B, C) shape
#         y = y.view(N, B, C, T).permute(3, 0, 1, 2)
#         return y
    
# class ButterWorthLayer(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.section1 = Biquad()
#         self.section2 = Biquad()

#     def forward(self, x):

#         x = self.section1(x)
#         x = self.section2(x)

#         return x

class FilterBank(nn.Module):

    def __init__(self, channels, num_freq_bins, num_bands=2):
        super().__init__()

        self.num_freq_bins = num_freq_bins
        self.channels = channels
        self.num_bands = num_bands

        # learnable cutoffs
        self.low = nn.Parameter(torch.rand(channels, num_bands) * 0.4)
        self.high = nn.Parameter(torch.rand(channels, num_bands) * 0.6 + 0.4)

        self.sharpness = 40

    def forward(self, x):
        # x: (T,N,B,C,F)
        device = x.device

        freqs = torch.linspace(
            start=0, 
            end=1, 
            steps=self.num_freq_bins,
            device=device
        )

        outputs = []

        # Apply a separate learned bandpass filter for each input band.
        for b in range(self.num_bands):
            # Select band b: (T, N, C, F)
            x_b = x[:, :, b, :, :]

            low = torch.sigmoid(self.low[:, b]).unsqueeze(-1)
            high = torch.sigmoid(self.high[:, b]).unsqueeze(-1)

            mask = (
                torch.sigmoid(self.sharpness * (freqs - low))
                - torch.sigmoid(self.sharpness * (freqs - high))
            )

            # mask shape: (1, 1, C, F) to broadcast over (T, N, C, F)
            mask = mask.unsqueeze(0).unsqueeze(0)

            outputs.append(x_b * mask)

        # Reassemble bands: (T, N, B, C, F)
        return torch.stack(outputs, dim=2)
