"""Continuous-tensor implementation of polarity routing and disposable channels.
"""
from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn


@dataclass
class DState:
    open: bool = True
    stored: Optional[torch.Tensor] = None


@dataclass
class RoutedPacket:
    positive: Optional[torch.Tensor]
    negative: Optional[torch.Tensor]
    disposable: Optional[torch.Tensor]


def route_packet(payload, polarity, state, reset=False):
    """Transmit one complete packet; simultaneous compatible and D routes copy it.
    """
    if polarity not in (-1, 0, 1):
        raise ValueError("polarity must be -1, 0, or +1")
    if reset:
        state.open = True
    positive = payload if polarity == 1 else None
    negative = payload if polarity == -1 else None
    disposable = payload if state.open else None
    if state.open:
        state.stored = payload
        state.open = False
    return RoutedPacket(positive, negative, disposable)


class AdaptiveDRead(nn.Module):
    """G=sigmoid(Conv1x1([encoder, upsampled_decoder])); return G*encoder.
    """
    def __init__(self, channels):
        super().__init__()
        self.selector = nn.Conv3d(2 * channels, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.selector.weight)
        nn.init.constant_(self.selector.bias, 2.0)

    def forward(self, saved, decoder):
        if saved.shape != decoder.shape:
            raise ValueError(f"D fusion requires matching shapes: {saved.shape}, {decoder.shape}")
        weights = torch.sigmoid(self.selector(torch.cat((saved, decoder), dim=1)))
        return saved * weights


class PolarityDDecoder(nn.Module):
    def __init__(self, original, adaptive):
        super().__init__()
        self.encoder = original.encoder
        self.num_classes = original.num_classes
        self.deep_supervision = original.deep_supervision
        self.stages = original.stages
        self.transpconvs = original.transpconvs
        self.seg_layers = original.seg_layers
        self.readers = nn.ModuleList([
            AdaptiveDRead(self.encoder.output_channels[-(s + 2)]) if adaptive else nn.Identity()
            for s in range(len(self.stages))])
        self.adaptive = adaptive

    def forward(self, bottleneck, states):
        x, outputs = bottleneck, []
        for s in range(len(self.stages)):
            state = states[-(s + 1)]
            packet = route_packet(x, +1, state)
            if packet.disposable is not None or state.stored is None:
                raise RuntimeError("Decoder must meet a closed D channel containing its encoder packet")
            x = self.transpconvs[s](packet.positive)
            skip = self.readers[s](state.stored, x) if self.adaptive else state.stored
            x = self.stages[s](torch.cat((x, skip), dim=1))
            if self.deep_supervision or s == len(self.stages) - 1:
                outputs.append(self.seg_layers[s](x))
        outputs.reverse()
        return outputs if self.deep_supervision else outputs[0]


class PolarityDUNet3D(nn.Module):
    def __init__(self, baseline, adaptive=False):
        super().__init__()
        if baseline.encoder.conv_op is not nn.Conv3d:
            raise ValueError("This implementation explicitly supports 3D PlainConvUNet")
        self.encoder = baseline.encoder
        self.decoder = PolarityDDecoder(baseline.decoder, adaptive)
        self.adaptive = adaptive

    def forward(self, x):

        states = [DState() for _ in range(len(self.encoder.stages) - 1)]
        for i, stage in enumerate(self.encoder.stages):
            x = stage(x)
            if i < len(states):
                x = route_packet(x, -1, states[i]).negative
        return self.decoder(x, states)

    def compute_conv_feature_map_size(self, input_size):
        if self.adaptive:
            raise NotImplementedError("Adaptive memory must be profiled; do not use baseline memory estimates")
        from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
        return (self.encoder.compute_conv_feature_map_size(input_size) +
                UNetDecoder.compute_conv_feature_map_size(self.decoder, input_size))
