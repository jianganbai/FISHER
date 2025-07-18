import torch
import torch.nn as nn
import numpy as np

from functools import partial
from dataclasses import dataclass
from typing import Callable, Dict, Optional
from enum import Enum, auto
from einops import rearrange

from .mae import get_2d_sincos_pos_embed_flexible, PatchEmbed_new


from .base import (
    D2vModalityConfig,
    ModalitySpecificEncoder,
    get_alibi_bias,
)
from .modules import (
    BlockEncoder,
    FixedPositionalEncoder,
)


class Modality(Enum):
    AUDIO = auto()
    IMAGE = auto()
    TEXT = auto()


@dataclass
class D2vImageConfig(D2vModalityConfig):
    type: Modality = Modality.IMAGE

    input_size: int = 224
    in_chans: int = 3
    patch_size: int = 16
    embed_dim: int = 768

    alibi_dims: int = 2
    alibi_distance: str = "manhattan"

    fixed_positions: bool = True

    transformer_decoder: bool = False
    enc_dec_transformer: bool = False
    target_length: int = 1024
    max_length: int = 768  # 64 for 10s
    max_freq: int = 400

    band_width: int = 50
    flatten: str = 'freq'  # 'time', 'freq'


class ImageEncoder(ModalitySpecificEncoder):
    # forward() implemented in models.base.ModalitySpecificEncoder

    modality_cfg: D2vImageConfig

    def __init__(
        self,
        modality_cfg: D2vImageConfig,
        embed_dim: int,
        make_block: Callable[[float, Optional[int], Optional[int]], nn.ModuleList],
        norm_layer: Callable[[int], nn.LayerNorm],
        layer_norm_first: bool,
        alibi_biases: Dict,
        task=None,
    ):
        self.patch_size = modality_cfg.patch_size
        self.band_width = modality_cfg.band_width
        self.W = self.band_width // self.patch_size
        self.H = modality_cfg.target_length // self.patch_size  # 64

        # convert spec to patch embed, using conv1d
        local_encoder = PatchEmbed_new(
            patch_size=modality_cfg.patch_size,  # 16
            in_chans=modality_cfg.in_chans,  # 1
            embed_dim=modality_cfg.embed_dim,  # 768
            stride=modality_cfg.patch_size,  # 16
            flatten=modality_cfg.flatten
        )

        # CNN initialize
        w = local_encoder.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        if modality_cfg.embed_dim != embed_dim:
            local_encoder = nn.Sequential(
                local_encoder,
                nn.Linear(modality_cfg.embed_dim, embed_dim),
            )

        project_features = nn.Identity()

        # note: max_length control the maximum time length of audio -> "64" for 10s, here we define it as 2min, you can change it yourself
        max_length = modality_cfg.max_length
        max_freq = modality_cfg.max_freq
        # max_length=768, self.W=8, embed_dim=768
        pos_embed = nn.Parameter(
            torch.zeros(1, max_length*max_freq, embed_dim), requires_grad=False
        )

        # side_n = int(num_patches ** 0.5)
        # note: we fix the variable length sequence problem here -> support up to 2min audio
        emb = get_2d_sincos_pos_embed_flexible(
            pos_embed.shape[-1],
            (max_length, max_freq),
            cls_token=False,
        )

        pos_embed.data.copy_(torch.from_numpy(emb[:max_length * max_freq, :]).float().unsqueeze(0))
        fixed_positional_encoder = (
            FixedPositionalEncoder(pos_embed) if modality_cfg.fixed_positions else None  # True
        )

        dpr = np.linspace(  # drop_path_rate
            modality_cfg.start_drop_path_rate,
            modality_cfg.end_drop_path_rate,
            modality_cfg.prenet_depth,  # actual: 0
        )

        # actual: only layer norm
        context_encoder = BlockEncoder(
            nn.ModuleList(make_block(dpr[i]) for i in range(modality_cfg.prenet_depth)),
            norm_layer(embed_dim) if not layer_norm_first else None,
            layer_norm_first,
            modality_cfg.prenet_layerdrop,
            modality_cfg.prenet_dropout,
        )

        alibi_bias_fn = partial(
            get_alibi_bias,
            alibi_biases=alibi_biases,
            heads=modality_cfg.num_alibi_heads,
            dims=modality_cfg.alibi_dims,
            distance=modality_cfg.alibi_distance,
        )

        super().__init__(
            modality_cfg=modality_cfg,
            embed_dim=embed_dim,
            local_encoder=local_encoder,  # patch embed
            project_features=project_features,  # nn.Identity()
            fixed_positional_encoder=fixed_positional_encoder,
            relative_positional_encoder=None,
            context_encoder=context_encoder,  # apply mask
            decoder=None,
            get_alibi_bias=alibi_bias_fn,
        )

    def reset_parameters(self):
        super().reset_parameters()

    @torch.no_grad()
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)   audio: (N,1,H,W)   1024/16 = 64   128/16 = 8
        x: (N, L, patch_size**2 *3)
        """
        if self.modality_cfg.in_chans == 1:  # actual: this one
            p = self.modality_cfg.patch_size
            h = imgs.shape[2] // p
            w = imgs.shape[3] // p
            # h,w = self.patch_embed.patch_hw
            x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))

        else:
            p = self.modality_cfg.patch_size
            h = w = imgs.shape[2] // p
            x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
            x = torch.einsum("nchpwq->nhwpqc", x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))

        return x

    @torch.no_grad()
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *C)
        imgs: (N, C, H, W)
        """
        p = self.modality_cfg.patch_size
        h = w = int(x.shape[1] ** 0.5)  # num patch along two axis
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, -1))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], -1, h * p, h * p))
        return imgs

    def convert_padding_mask(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor
    ) -> torch.Tensor:
        '''patchify and serialize padding_mask: [b,t,f] => [b,t_patch,f_patch] => [b,patch_seq]

        Args:
            x (torch.Tensor): input_features
            padding_mask (torch.Tensor): [b,t_patch,f_patch], 1 for padded patch

        Returns:
            torch.Tensor: serialized padding mask. [b,patch_seq]
        '''
        B, T, F = x.shape
        t_extra, f_extra = T % self.patch_size, F % self.patch_size
        padding_mask = padding_mask[:, :-t_extra, :-f_extra]
        padding_mask = rearrange(
            padding_mask,
            'b (tp p) (fp q) -> b tp fp (p q)',
            p=self.patch_size, q=self.patch_size
        )
        padding_mask = padding_mask.all(-1)

        if self.modality_cfg.flatten == 'time':
            padding_mask = padding_mask.transpose(-2, -1).flatten(1)
        else:
            padding_mask = padding_mask.flatten(1)
        return padding_mask
