from collections.abc import Container, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SGMSE_ezalg_UNet(nn.Module):
    """NCSN++ backbone used in SGMSE, modified to be time-independent."""

    def __init__(
        self,
        num_channels: int = 1,
        base_channels: int = 128,
        num_res_blocks: int = 2,
        channel_mult: Sequence[int] = (1, 1, 2, 2, 2, 2, 2),
        attn_levels: Container[int] = (4,),
    ) -> None:
        """Initialize the SGMSE NCSN++ backbone.

        Args:
            num_channels: Number of input channels.
            base_channels: Base number of channels.
            num_res_blocks: Number of residual blocks per level.
            channel_mult: Channel multiplier for each level.
            attn_levels: Indices of levels at which to apply attention.
        """
        super().__init__()

        channels = [base_channels * mult for mult in channel_mult]

        # Keep ch_fourier and ch_emb calculation if you want the constructor body
        # to stay close to the original structure.
        # But ch_fourier is no longer used because time embedding is removed.
        ch_in, ch_fourier, ch_emb, ch_prog = channels[0], base_channels, 4 * base_channels, 4 * num_channels
        del ch_fourier

        # Removed:
        # self.emb_block = SGMSEEmbeddingBlock(ch_fourier, ch_emb)

        self.encoder = nn.ModuleList()

        ch_skip = []

        for i, ch_out in enumerate(channels):
            if i == 0:
                self.encoder.append(nn.Conv2d(ch_prog, ch_in, 3, 1, 1))
                ch_skip.append(ch_in)
            else:
                self.encoder.append(SGMSE_ezalg_UNetBlock(ch_in, ch_in, ch_prog, ch_emb, "down"))
                ch_skip.append(ch_in)

            for _ in range(num_res_blocks):
                self.encoder.append(SGMSE_ezalg_UNetBlock(ch_in, ch_out, ch_prog, ch_emb, attn=i in attn_levels))
                ch_skip.append(ch_out)
                ch_in = ch_out

        self.decoder = nn.ModuleList()

        for i, ch_out in reversed(list(enumerate(channels))):
            if i == len(channels) - 1:
                self.decoder.append(SGMSE_ezalg_UNetBlock(ch_in, ch_in, ch_prog, ch_emb, attn=True))
                self.decoder.append(SGMSE_ezalg_UNetBlock(ch_in, ch_in, ch_prog, ch_emb))
            else:
                self.decoder.append(SGMSE_ezalg_UNetBlock(ch_in, ch_in, ch_prog, ch_emb, "up"))

            for j in range(num_res_blocks + 1):
                attn = i in attn_levels and j == num_res_blocks
                self.decoder.append(SGMSE_ezalg_UNetBlock(ch_in + ch_skip.pop(), ch_out, ch_prog, ch_emb, attn=attn))
                ch_in = ch_out

        self.out_prog = nn.Sequential(
            sgmse_groupnorm(ch_in),
            nn.SiLU(),
            nn.Conv2d(ch_in, ch_prog, 3, 1, 1),
        )

        self.out_conv = nn.Conv2d(ch_prog, 2 * num_channels, 1)

        self.downsampling_factor = 2 ** (len(channel_mult) - 1)
        self.num_res_blocks = num_res_blocks

    def forward(self, x: Tensor, y: Tensor, t: Tensor | None = None) -> Tensor:
        """Forward pass.

        Args:
            x: Complex-valued noisy speech tensor.
                Shape `(batch_size, num_channels, num_freqs, num_frames)`.
            y: Complex-valued diffused speech tensor.
                Shape `(batch_size, num_channels, num_freqs, num_frames)`.
            t: Ignored. Kept only for compatibility with existing call sites.

        Returns:
            Complex-valued output score.
            Shape `(batch_size, num_channels, num_freqs, num_frames)`.
        """
        del t

        if x.shape != y.shape:
            raise ValueError(f"Input shapes must match. Got {x.shape} and {y.shape}.")

        if x.ndim != 4:
            raise ValueError(f"Input must be 4-dimensional. Got shape {x.shape}.")

        if x.shape[2] % self.downsampling_factor != 0 or x.shape[3] % self.downsampling_factor != 0:
            raise ValueError(
                f"Input size along dimensions 2 and 3 must be divisible by {self.downsampling_factor}. "
                f"Got shape {x.shape}."
            )

        # Removed:
        # emb = self.emb_block(t)

        x = torch.cat([x.real, x.imag, y.real, y.imag], dim=1)

        prog: Tensor | None = x
        skips = []

        for i, block in enumerate(self.encoder):
            if i == 0:
                x = block(x)
            else:
                x, prog = block(x, prog)

            skips.append(x)

        prog = None

        for i, block in enumerate(self.decoder):
            if i == 0 or i % (self.num_res_blocks + 2) == 1:
                x, prog = block(x, prog)
            else:
                x = torch.cat([x, skips.pop()], dim=1)
                x, prog = block(x, prog)

        prog += self.out_prog(x)

        out = self.out_conv(prog)

        return torch.complex(*out.chunk(2, dim=1))
    
class SGMSE_ezalg_UNetBlock(nn.Module):
    """SGMSE UNet block without time embedding."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        prog_ch: int,
        emb_ch: int,
        kind: str | None = None,
        attn: bool = False,
    ) -> None:
        """Initialize the SGMSE UNet block.

        Args:
            in_ch: Number of input channels.
            out_ch: Number of output channels.
            prog_ch: Number of progressive input/output channels.
            emb_ch: Ignored. Kept only so SGMSE_ezalg_UNet constructor calls stay unchanged.
            kind: Whether this block performs upsampling, downsampling, or neither.
            attn: Whether to use attention.
        """
        super().__init__()

        del emb_ch

        assert kind in ("up", "down", None), f"`kind` must be 'up', 'down', or None. Got {kind}."

        self.resample = nn.Identity() if kind is None else SGMSEResample(kind)

        self.act = nn.SiLU()

        # Removed:
        # self.fc = nn.Linear(emb_ch, out_ch)

        self.norm_1 = sgmse_groupnorm(in_ch)
        self.conv_1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)

        self.norm_2 = sgmse_groupnorm(out_ch)
        self.conv_2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)

        self.skip_conv = nn.Identity() if in_ch == out_ch and kind is None else nn.Conv2d(in_ch, out_ch, 1)

        self.attn = SGMSEAttentionBlock(out_ch) if attn else nn.Identity()

        self.prog_conv = (
            None
            if kind is None
            else (
                nn.Sequential(
                    sgmse_groupnorm(in_ch),
                    nn.SiLU(),
                    nn.Conv2d(in_ch, prog_ch, 3, 1, 1),
                )
                if kind == "up"
                else nn.Conv2d(prog_ch, out_ch, 1)
            )
        )

        self.kind = kind

    def forward(self, x: Tensor, prog: Tensor | None) -> tuple[Tensor, Tensor | None]:
        """Forward pass."""
        if self.kind == "up" and self.prog_conv is not None:
            prog = self.prog_conv(x) if prog is None else prog + self.prog_conv(x)
            prog = self.resample(prog)

        h = self.conv_1(self.resample(self.act(self.norm_1(x))))

        # Removed:
        # h += self.fc(emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv_2(self.act(self.norm_2(h)))
        x = self.skip_conv(self.resample(x))
        x = (x + h) / 2**0.5
        x = self.attn(x)
        
        if self.kind == "down" and self.prog_conv is not None:
            prog = self.resample(prog)
            x += self.prog_conv(prog)
        return x, prog

class SGMSEAttentionBlock(nn.Module):
    """SGMSE attention block."""

    def __init__(self, num_channels: int) -> None:
        """Initialize the SGMSE attention block."""
        super().__init__()
        self.norm = sgmse_groupnorm(num_channels)
        self.conv_qkv = nn.Conv2d(num_channels, 3 * num_channels, 1)
        self.conv_out = nn.Conv2d(num_channels, num_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.conv_qkv(h).reshape(B, 3 * C, H * W).chunk(3, dim=1)
        h = torch.einsum("bci,bcj->bij", q, k) * C ** (-0.5)
        h = torch.softmax(h, dim=-1)
        h = torch.einsum("bij,bcj->bci", h, v).reshape(B, C, H, W)
        h = self.conv_out(h)
        return (x + h) / 2**0.5
    
class SGMSEResample(nn.Module):
    """SGMSE 2D resampling block."""

    kernel: Tensor

    def __init__(self, kind: str) -> None:
        """Initialize the SGMSE 2D resampling block."""
        super().__init__()
        assert kind in ("up", "down"), f"`kind` must be 'up' or 'down'. Got {kind}."
        self.kind = kind
        kernel = torch.tensor([1, 3, 3, 1], dtype=torch.float32)
        kernel = kernel.outer(kernel).unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", kernel / kernel.sum())

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        kernel = (4 if self.kind == "up" else 1) * self.kernel.tile([x.shape[1], 1, 1, 1])
        func = F.conv_transpose2d if self.kind == "up" else F.conv2d
        return func(x, kernel, padding=1, groups=x.shape[1], stride=2)


    
def sgmse_groupnorm(num_channels: int) -> nn.GroupNorm:
    """SGMSE group normalization layer."""
    return nn.GroupNorm(min(32, num_channels // 4), num_channels, 1e-6)
