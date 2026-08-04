"""
Boundary Refinement GAN — model architecture.

This is the PyTorch definition used both for training (train_gan.py, run
offline on GPU/Colab) and for export to ONNX for CPU inference in the live
pipeline (gan_refinement.py).

Design constraints driving the architecture choices:
  - Operates on small PATCHES (the output of region_selector.py), not full
    frames — this is what keeps it fast enough to matter for your adaptive
    pipeline's FPS story. Don't scale this up to full-frame without
    re-checking the timing budget.
  - Input: RGB patch (3 channels) + coarse/MiDaS depth patch (1 channel),
    concatenated -> 4 input channels. The RGB gives the network real edge
    / texture information to hallucinate structure from (e.g. where the
    actual glasses frame boundary is), the coarse depth gives it the
    approximate depth values to refine rather than predict from scratch.
  - Output: a single-channel refined depth patch, same spatial size as
    input by default (the generator is fully convolutional, so it isn't
    hard-locked to one patch size, but PATCH_SIZE below is what it's
    designed and should be trained around).
"""

import torch
import torch.nn as nn

PATCH_SIZE = 64  # default training/inference patch size (H=W=PATCH_SIZE)


def _conv_block(in_ch, out_ch, stride=2, use_norm=True):
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1)]
    if use_norm:
        layers.append(nn.InstanceNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def _deconv_block(in_ch, out_ch, use_norm=True, dropout=False):
    layers = [nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)]
    if use_norm:
        layers.append(nn.InstanceNorm2d(out_ch))
    if dropout:
        layers.append(nn.Dropout(0.3))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class BoundaryRefinementGenerator(nn.Module):
    """Small U-Net generator. Input: (B, 4, H, W) = RGB + coarse depth.
    Output: (B, 1, H, W) = refined depth, same spatial size.

    Deliberately small (4 down / 4 up stages, modest channel counts) since
    this runs per-region on CPU in the live pipeline — this is not the
    place to reach for a deep, high-capacity network.
    """

    def __init__(self, in_channels=4, base_channels=32):
        super().__init__()
        c = base_channels

        # Encoder
        self.enc1 = _conv_block(in_channels, c, use_norm=False)   # H/2
        self.enc2 = _conv_block(c, c * 2)                          # H/4
        self.enc3 = _conv_block(c * 2, c * 4)                      # H/8
        self.enc4 = _conv_block(c * 4, c * 8)                      # H/16

        # Bottleneck
        self.bottleneck = _conv_block(c * 8, c * 8, use_norm=False)  # H/32

        # Decoder (with skip connections, standard U-Net / pix2pix style)
        self.dec4 = _deconv_block(c * 8, c * 8, dropout=True)
        self.dec3 = _deconv_block(c * 8 + c * 8, c * 4)
        self.dec2 = _deconv_block(c * 4 + c * 4, c * 2)
        self.dec1 = _deconv_block(c * 2 + c * 2, c)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(c + c, 1, kernel_size=4, stride=2, padding=1),
            # No activation here — depth is regressed as a continuous value,
            # not squashed into [0,1] or [-1,1]. Normalize depth consistently
            # between training pairs and live inference instead (see
            # train_gan.py's preprocessing).
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        d4 = self.dec4(b)
        d3 = self.dec3(torch.cat([d4, e4], dim=1))
        d2 = self.dec2(torch.cat([d3, e3], dim=1))
        d1 = self.dec1(torch.cat([d2, e2], dim=1))
        out = self.final(torch.cat([d1, e1], dim=1))

        return out


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator (pix2pix-style) — classifies overlapping
    patches of the input as real/fake rather than the whole image at once,
    which is what encourages sharp local (boundary) detail rather than
    blurry global averages. Training-only; not used at inference time.

    Input: concatenation of [RGB, depth] = 4 channels, either the real
    (sharp, ground-truth) pair or the (RGB, generator-output) pair.
    """

    def __init__(self, in_channels=4, base_channels=32):
        super().__init__()
        c = base_channels
        self.model = nn.Sequential(
            _conv_block(in_channels, c, use_norm=False),
            _conv_block(c, c * 2),
            _conv_block(c * 2, c * 4),
            nn.Conv2d(c * 4, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    # Structural self-test: confirms tensor shapes flow correctly through
    # both networks before you spend any GPU time training on real data.
    gen = BoundaryRefinementGenerator()
    disc = PatchDiscriminator()

    dummy_input = torch.randn(2, 4, PATCH_SIZE, PATCH_SIZE)  # batch of 2
    refined_depth = gen(dummy_input)
    print(f"Generator: input {tuple(dummy_input.shape)} -> output {tuple(refined_depth.shape)}")
    assert refined_depth.shape == (2, 1, PATCH_SIZE, PATCH_SIZE), "BUG: generator output shape wrong"

    disc_input = torch.cat([dummy_input[:, :3], refined_depth], dim=1)  # RGB + generated depth
    disc_output = disc(disc_input)
    print(f"Discriminator: input {tuple(disc_input.shape)} -> output {tuple(disc_output.shape)}")

    n_params_gen = sum(p.numel() for p in gen.parameters())
    n_params_disc = sum(p.numel() for p in disc.parameters())
    print(f"Generator parameters: {n_params_gen:,}")
    print(f"Discriminator parameters: {n_params_disc:,}")

    print("Shape checks passed.")