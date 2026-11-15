"""PANNs CNN14 (16 kHz) as a frozen feature extractor, plus the shared log-mel front end.

The CNN14 layer definitions below are copied from the PANNs repository
(github.com/qiuqiangkong/audioset_tagging_cnn, pytorch/models.py) so that the
official checkpoint Cnn14_16k_mAP=0.438.pth loads with matching parameter names.
Mixup and the training-time SpecAugment were removed; an optional masking hook
is added for embedding augmentation.

    The MIT License
    Copyright (c) 2018-2020 Qiuqiang Kong
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import LogmelFilterBank, Spectrogram


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if getattr(layer, "bias", None) is not None:
        layer.bias.data.fill_(0.0)


def init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class LogMel(nn.Module):
    """Log-mel (dB) spectrogram with the CNN14 16 kHz settings. Input (B, samples) -> (B, 1, T, n_mels)."""

    def __init__(self, sample_rate, n_fft, hop_length, n_mels, fmin, fmax):
        super().__init__()
        self.spectrogram_extractor = Spectrogram(
            n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window="hann",
            center=True, pad_mode="reflect", freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax,
            ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)

    def forward(self, wav):
        return self.logmel_extractor(self.spectrogram_extractor(wav))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x, pool_size=(2, 2)):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, kernel_size=pool_size)


class Cnn14(nn.Module):
    """CNN14 from PANNs. forward() returns the 2,048-d embedding."""

    def __init__(self, sample_rate=16000, window_size=512, hop_size=160, mel_bins=64,
                 fmin=50, fmax=8000, classes_num=527):
        super().__init__()
        front = LogMel(sample_rate, window_size, hop_size, mel_bins, fmin, fmax)
        # keep the checkpoint's attribute names
        self.spectrogram_extractor = front.spectrogram_extractor
        self.logmel_extractor = front.logmel_extractor
        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, wav, mask_fn=None):
        x = self.logmel_extractor(self.spectrogram_extractor(wav))  # (B, 1, T, mel)
        if mask_fn is not None:  # augmentation of training clips only
            x = mask_fn(x)
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        for i, block in enumerate([self.conv_block1, self.conv_block2, self.conv_block3,
                                   self.conv_block4, self.conv_block5, self.conv_block6]):
            x = block(x, pool_size=(1, 1) if i == 5 else (2, 2))
        x = torch.mean(x, dim=3)
        x = torch.max(x, dim=2)[0] + torch.mean(x, dim=2)
        return F.relu_(self.fc1(x))  # 'embedding' in PANNs (dropout is inactive in eval mode)

    def frame_features(self, wav):
        """Frame-level features for the speech analyzer: output of conv blocks 3 and 4, averaged over
        frequency. Block 3 has 8x time pooling (80 ms per frame at a 10 ms hop); block 4 (160 ms) is
        repeated to the same rate. Returns (B, T_frames, 768)."""
        x = self.logmel_extractor(self.spectrogram_extractor(wav))
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        b3 = self.conv_block3(x)
        b4 = self.conv_block4(b3)
        f3 = b3.mean(dim=3)                                    # (B, 256, T3)
        f4 = b4.mean(dim=3).repeat_interleave(2, dim=2)        # (B, 512, ~T3)
        T = min(f3.shape[2], f4.shape[2])
        return torch.cat([f3[:, :, :T], f4[:, :, :T]], dim=1).transpose(1, 2)


def load_cnn14(checkpoint: str | None, spec_cfg: dict, sample_rate: int, device: str) -> Cnn14:
    model = Cnn14(sample_rate, spec_cfg["n_fft"], spec_cfg["hop_length"], spec_cfg["n_mels"],
                  spec_cfg["fmin"], spec_cfg["fmax"])
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = state.get("model", state)
        model.load_state_dict(state, strict=True)
    else:
        print("WARNING: no CNN14 checkpoint set; using RANDOM weights (pipeline test only).")
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval().to(device)


def spec_mask(x: torch.Tensor, time_masks: int, time_width: int, freq_masks: int, freq_width: int,
              gen: torch.Generator) -> torch.Tensor:
    """SpecAugment-style time and frequency masking on (B, 1, T, F) log-mel; fills with each item's mean."""
    x = x.clone()
    B, _, T, Fq = x.shape
    for b in range(B):
        fill = x[b].mean()
        for _ in range(time_masks):
            w = int(torch.randint(0, time_width + 1, (1,), generator=gen))
            t0 = int(torch.randint(0, max(1, T - w), (1,), generator=gen))
            x[b, :, t0:t0 + w, :] = fill
        for _ in range(freq_masks):
            w = int(torch.randint(0, freq_width + 1, (1,), generator=gen))
            f0 = int(torch.randint(0, max(1, Fq - w), (1,), generator=gen))
            x[b, :, :, f0:f0 + w] = fill
    return x
