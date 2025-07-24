import torch
import logging
import numpy as np
import torch.nn as nn

from functools import partial
from einops import rearrange
from typing import Callable, Optional
from dataclasses import dataclass, field, is_dataclass


from .base import (
    D2vModalityConfig,
    ModalitySpecificEncoder,
)

from .modules import AltBlock

from .images import (
    D2vImageConfig,
    ImageEncoder,
)

logger = logging.getLogger(__name__)


@dataclass
class D2vModalitiesConfig:
    image: D2vImageConfig = field(default_factory=lambda *args: D2vImageConfig())


@dataclass
class Data2VecMultiConfig:
    depth: int = 12

    # band split
    band_width: int = 100

    # standard vision Transformer
    start_drop_path_rate: float = 0.0
    end_drop_path_rate: float = 0.0
    num_heads: int = 12
    norm_eps: float = 1e-6
    norm_affine: bool = True
    encoder_dropout: float = 0.0
    post_mlp_drop: float = 0.0
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    dropout_input: float = 0.0
    layerdrop: float = 0.0
    embed_dim: int = 768
    mlp_ratio: float = 4.0
    layer_norm_first: bool = False

    end_of_block_targets: bool = False

    # clone batch for multi-mask strategy
    clone_batch: int = 8
    max_band_per_sample: int = 64

    # normalization for teacher Transformer layer output
    layer_norm_target_layer: bool = False
    batch_norm_target_layer: bool = False
    instance_norm_target_layer: bool = True
    instance_norm_targets: bool = False
    layer_norm_targets: bool = True

    modalities: D2vModalitiesConfig = field(default_factory=lambda *args: D2vModalitiesConfig())


class FISHER(nn.Module):
    def __init__(self, cfg: Data2VecMultiConfig):
        super().__init__()
        self.cfg = cfg

        make_layer_norm = partial(
            nn.LayerNorm, eps=cfg.norm_eps, elementwise_affine=cfg.norm_affine
        )

        def make_block(drop_path, dim=None, heads=None):
            return AltBlock(
                cfg.embed_dim if dim is None else dim,
                cfg.num_heads if heads is None else heads,
                cfg.mlp_ratio,
                qkv_bias=True,
                drop=cfg.encoder_dropout,
                attn_drop=cfg.attention_dropout,
                mlp_drop=cfg.activation_dropout,
                post_mlp_drop=cfg.post_mlp_drop,
                drop_path=drop_path,
                norm_layer=make_layer_norm,
                layer_norm_first=cfg.layer_norm_first,
                ffn_targets=not cfg.end_of_block_targets,
            )

        self.alibi_biases = {}
        self.modality_encoders = nn.ModuleDict()

        mod_cfg = getattr(cfg.modalities, 'image')
        enc = self.make_modality_encoder(
            mod_cfg,
            cfg.embed_dim,
            make_block,
            make_layer_norm,
            cfg.layer_norm_first,
            self.alibi_biases,
        )
        self.modality_encoders['IMAGE'] = enc

        self.dropout_input = nn.Dropout(cfg.dropout_input)

        dpr = np.linspace(cfg.start_drop_path_rate, cfg.end_drop_path_rate, cfg.depth)

        self.blocks = nn.ModuleList([make_block(dpr[i]) for i in range(cfg.depth)])

        self.norm = None
        if cfg.layer_norm_first:
            self.norm = make_layer_norm(cfg.embed_dim)

        # band split
        self.band_width = cfg.band_width
        self.patch_size = cfg.modalities.image.patch_size
        self.num_time_patch = cfg.modalities.image.target_length // self.patch_size
        self.num_band_patch = self.band_width // self.patch_size

    def make_modality_encoder(
        self,
        cfg: D2vModalityConfig,
        embed_dim: int,
        make_block: Callable[[float], nn.ModuleList],
        norm_layer: Callable[[int], nn.LayerNorm],
        layer_norm_first: bool,
        alibi_biases,
        task=None,
    ) -> ModalitySpecificEncoder:
        return ImageEncoder(
            cfg,
            embed_dim,
            make_block,
            norm_layer,
            layer_norm_first,
            alibi_biases,
            task,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str
    ):
        """
        Load a pretrained FISHER model from a checkpoint file.
        """
        def update_dataclass(instance, data_dict):
            if not data_dict:
                return instance

            for field_name, field_value in data_dict.items():
                if hasattr(instance, field_name):
                    current_value = getattr(instance, field_name)
                    if is_dataclass(current_value) and isinstance(field_value, dict):
                        update_dataclass(current_value, field_value)
                    else:
                        setattr(instance, field_name, field_value)
            return instance

        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        cfg = Data2VecMultiConfig()
        update_dataclass(cfg, state_dict['cfg']['model'])
        model = cls(cfg)
        load_info = model.load_state_dict(state_dict['model'], strict=True)
        print(load_info)
        return model

    def state_dict(self, **kwargs):
        state = {
            'cfg': self.cfg,
            'model': super().state_dict(**kwargs)
        }
        return state

    def forward(
        self,
        source: torch.Tensor,
        target=None,
        id=None,
        mode='IMAGE',
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        force_remove_masked=False,
        remove_extra_tokens: bool = True,
        precomputed_mask: Optional[torch.Tensor] = None,
    ):
        # band split
        num_band = source.shape[-1] // self.band_width
        source = torch.stack(source.split(self.band_width, dim=-1)[:num_band])  # drop residual
        source = rearrange(source, 'nb B c t f -> (B nb) c t f')
        clone_batch = self.cfg.max_band_per_sample // num_band

        feature_extractor = self.modality_encoders[mode]  # models.images.ImageEncoder

        # extract (unmasked) features using CNN encoder
        extractor_out = feature_extractor(
            source,
            padding_mask,
            mask,
            remove_masked=not features_only or force_remove_masked,  # train: True; infer: False
            clone_batch=clone_batch if not features_only else 1,
            mask_seeds=None,
            precomputed_mask=precomputed_mask,
        )

        # x in shape (batch_size * clone batch, patch_frame(64) * patch_freqency(8) * unmask_ratio(0.2) + 1(cls_token), 768(feature dimension))
        x = extractor_out["x"]
        # encoder_mask is applied on sub-band level
        encoder_mask = extractor_out["encoder_mask"]  # models.base.MaskInfo, ["x_unmasked", "mask", "ids_restore", "ids_keep"]
        masked_padding_mask = extractor_out["padding_mask"]
        masked_alibi_bias = extractor_out.get("alibi_bias", None)
        alibi_scale = extractor_out.get("alibi_scale", None)

        if self.dropout_input is not None:
            x = self.dropout_input(x)

        # standard Transformer (for student encoder)
        layer_results = []
        for i, blk in enumerate(self.blocks):
            if (
                not self.training
                or self.cfg.layerdrop == 0
                or (np.random.random() > self.cfg.layerdrop)
            ):
                ab = masked_alibi_bias
                if ab is not None and alibi_scale is not None:
                    scale = (
                        alibi_scale[i]
                        if alibi_scale.size(0) > 1
                        else alibi_scale.squeeze(0)
                    )
                    ab = ab * scale.type_as(ab)

                x, lr = blk(
                    x,
                    padding_mask=masked_padding_mask,
                    alibi_bias=ab,
                )
                if features_only:
                    layer_results.append(lr)

        if self.norm is not None:
            x = self.norm(x)

        # extract features for fine-tuning
        if features_only:
            if remove_extra_tokens:
                x = x[:, feature_extractor.modality_cfg.num_extra_tokens :]
                if masked_padding_mask is not None:
                    masked_padding_mask = masked_padding_mask[
                        :, feature_extractor.modality_cfg.num_extra_tokens :
                    ]

            return {
                "x": x,
                "padding_mask": masked_padding_mask,
                "layer_results": layer_results,
                "mask": encoder_mask,
            }

    def extract_features(
        self, source, mode='IMAGE', padding_mask=None, mask=False, remove_extra_tokens=False
    ):
        num_band = source.shape[-1] // self.band_width
        res = self.forward(
            source,
            mode=mode,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            remove_extra_tokens=remove_extra_tokens,
        )
        x = res['x'][:, 0]
        x = rearrange(x, '(B nb) D -> B (nb D)', nb=num_band)
        return x
