import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple
from dataclasses import dataclass, field
from functools import partial
from omegaconf import MISSING, II
from typing import Optional, Callable
from enum import Enum, auto

from .modules import D2vDecoderConfig


logger = logging.getLogger(__name__)


class Modality(Enum):
    AUDIO = auto()
    IMAGE = auto()
    TEXT = auto()


@dataclass
class D2vModalityConfig:
    type: Modality = MISSING
    prenet_depth: int = 0
    prenet_layerdrop: float = 0.0
    prenet_dropout: float = 0.0
    start_drop_path_rate: float = 0.0
    end_drop_path_rate: float = 0.0

    num_extra_tokens: int = 1
    init_extra_token_zero: bool = False

    mask_noise_std: float = 0.01
    mask_prob_min: Optional[float] = None
    mask_prob: float = 0.8
    inverse_mask: bool = True
    mask_prob_adjust: float = 0.07
    keep_masked_pct: float = 0.0
    flexible_mask: bool = False

    mask_length: int = 5
    add_masks: bool = False
    remove_masks: bool = False
    mask_dropout: float = 0.0
    encoder_zero_mask: bool = True

    mask_channel_prob: float = 0.0
    mask_channel_length: int = 64

    ema_local_encoder: bool = True  # used in data2vec_multi
    ema_local_decoder: bool = False
    local_grad_mult: float = 1.0
    flatten: str = 'freq'
    max_length: int = 128
    max_freq: int = 50

    use_alibi_encoder: bool = False
    alibi_scale: float = 1.0
    learned_alibi: bool = False
    alibi_max_pos: Optional[int] = None
    learned_alibi_scale: bool = False
    learned_alibi_scale_per_head: bool = False
    learned_alibi_scale_per_layer: bool = False

    num_alibi_heads: int = II("model.num_heads")
    model_depth: int = II("model.depth")

    decoder: Optional[D2vDecoderConfig] = field(default_factory=lambda *x: D2vDecoderConfig())


MaskSeed = namedtuple("MaskSeed", ["seed", "update", "ids"])
MaskInfo = namedtuple("MaskInfo", ["x_unmasked", "mask", "ids_restore", "ids_keep"])


class ModalitySpecificEncoder(nn.Module):
    def __init__(
        self,
        modality_cfg: D2vModalityConfig,
        embed_dim: int,
        local_encoder: nn.Module,
        project_features: nn.Module,
        fixed_positional_encoder: Optional[nn.Module],
        relative_positional_encoder: Optional[nn.Module],  # None
        context_encoder: nn.Module,
        decoder: Optional[nn.Module],
        get_alibi_bias: Optional[Callable[[int, int, str, str], torch.Tensor]],
    ):
        super().__init__()

        self.modality_cfg = modality_cfg
        self.local_encoder = local_encoder
        self.project_features = project_features
        self.fixed_positional_encoder = fixed_positional_encoder
        self.relative_positional_encoder = relative_positional_encoder
        self.context_encoder = context_encoder

        self.decoder = decoder
        self.get_alibi_bias = get_alibi_bias if modality_cfg.use_alibi_encoder else None

        self.local_grad_mult = self.modality_cfg.local_grad_mult

        self.extra_tokens = None
        if modality_cfg.num_extra_tokens > 0:
            self.extra_tokens = nn.Parameter(
                torch.zeros(1, modality_cfg.num_extra_tokens, embed_dim)
            )
            if not modality_cfg.init_extra_token_zero:
                nn.init.normal_(self.extra_tokens)
            elif self.extra_tokens.size(1) > 1:
                nn.init.normal_(self.extra_tokens[:, 1:])

        self.alibi_scale = None
        if self.get_alibi_bias is not None:
            self.alibi_scale = nn.Parameter(
                torch.full(
                    (
                        (modality_cfg.prenet_depth + modality_cfg.model_depth)
                        if modality_cfg.learned_alibi_scale_per_layer
                        else 1,
                        1,
                        self.modality_cfg.num_alibi_heads
                        if modality_cfg.learned_alibi_scale_per_head
                        else 1,
                        1,
                        1,
                    ),
                    modality_cfg.alibi_scale,
                    dtype=torch.float,
                ),
                requires_grad=modality_cfg.learned_alibi_scale,
            )

        if modality_cfg.learned_alibi and self.get_alibi_bias is not None:
            assert modality_cfg.alibi_max_pos is not None
            alibi_bias = self.get_alibi_bias(
                batch_size=1,
                time_steps=modality_cfg.alibi_max_pos,
                heads=modality_cfg.num_alibi_heads,
                scale=1.0,
                dtype=torch.float,
                device="cpu",
            )
            self.alibi_bias = nn.Parameter(alibi_bias)
            self.get_alibi_bias = partial(
                _learned_alibi_bias, alibi_bias=self.alibi_bias
            )

    def upgrade_state_dict_named(self, state_dict, name):
        k = f"{name}.alibi_scale"
        if k in state_dict and state_dict[k].dim() == 4:
            state_dict[k] = state_dict[k].unsqueeze(0)

        return state_dict

    def convert_padding_mask(self, x, padding_mask):
        return padding_mask

    def local_features(self, features):
        x = self.local_encoder(features)
        x = self.project_features(x)  # nn.Identity()
        return x

    def contextualized_features(
        self,
        x,
        padding_mask,
        mask,  # True
        remove_masked,  # train: True; infer: False
        clone_batch: int = 1,
        mask_seeds: Optional[torch.Tensor] = None,
        precomputed_mask=None,
    ):

        if padding_mask is not None:
            padding_mask = self.convert_padding_mask(x, padding_mask)  # [b,t,f] => [b,seq]

        local_features = x
        if mask and clone_batch == 1:
            local_features = local_features.clone()

        orig_B, orig_T, _ = x.shape
        pre_mask_B = orig_B
        mask_info = None

        x_pos = None
        # x: [B, seq_len, embed_dim]
        if self.fixed_positional_encoder is not None:  # models.modules.FixPositionalEncoder
            x = x + self.fixed_positional_encoder(x, padding_mask)[:, :x.size(1), :]

        if self.relative_positional_encoder is not None:
            x_pos = self.relative_positional_encoder(x)

        masked_padding_mask = padding_mask
        if mask and remove_masked:  # only pass masked to student
            x = mask_info.x_unmasked
            if x_pos is not None:
                x = x + gather_unmasked(x_pos, mask_info)

            # padding_mask: [bs, seq_len]
            # valid: False; padded: True
            if padding_mask is not None and padding_mask.any():
                # retrieve padding_mask for unmasked patch
                masked_padding_mask = gather_unmasked_mask(padding_mask, mask_info)
                if not masked_padding_mask.any():
                    masked_padding_mask = None
            else:
                masked_padding_mask = None

        elif x_pos is not None:
            x = x + x_pos

        alibi_bias = None
        alibi_scale = self.alibi_scale

        if self.get_alibi_bias is not None:
            alibi_bias = self.get_alibi_bias(
                batch_size=pre_mask_B,
                time_steps=orig_T,
                heads=self.modality_cfg.num_alibi_heads,
                dtype=torch.float32,
                device=x.device,
            )

            if alibi_scale is not None:
                alibi_scale = alibi_scale.clamp_min(0)
                if alibi_scale.size(0) == 1:
                    alibi_bias = alibi_bias * alibi_scale.squeeze(0).type_as(alibi_bias)
                    alibi_scale = None

            if clone_batch > 1:
                alibi_bias = alibi_bias.repeat_interleave(clone_batch, 0)

            if mask_info is not None and remove_masked:
                alibi_bias = masked_alibi(alibi_bias, mask_info)

        if self.extra_tokens is not None:
            num = self.extra_tokens.size(1)
            x = torch.cat([self.extra_tokens.expand(x.size(0), -1, -1), x], dim=1)
            if masked_padding_mask is not None:
                # B x T
                masked_padding_mask = F.pad(masked_padding_mask, (num, 0))
            if alibi_bias is not None:
                # B x H x T x T
                alibi_bias = F.pad(alibi_bias, (num, 0, num, 0))

        x = self.context_encoder(
            x,
            masked_padding_mask,
            alibi_bias,
            alibi_scale[: self.modality_cfg.prenet_depth]
            if alibi_scale is not None
            else None,
        )

        return {
            "x": x,
            "local_features": local_features,
            "padding_mask": masked_padding_mask,
            "alibi_bias": alibi_bias,
            "alibi_scale": alibi_scale[self.modality_cfg.prenet_depth :]
            if alibi_scale is not None and alibi_scale.size(0) > 1
            else alibi_scale,
            "encoder_mask": mask_info,
        }

    def forward(
        self,
        features,
        padding_mask,
        mask: bool,
        remove_masked: bool,
        clone_batch: int = 1,
        mask_seeds: Optional[torch.Tensor] = None,
        precomputed_mask=None,
    ):
        x = self.local_features(features)  # patch embed
        # x: [bs, time*freq, embed_dim], e.g. [12, 512, 768]
        out = self.contextualized_features(
            x,
            padding_mask,
            mask,
            remove_masked,
            clone_batch,
            mask_seeds,
            precomputed_mask,
        )  # add mask, discarded masked, context encoder (only layer norm)
        return out

    def reset_parameters(self):
        pass

    def remove_pretraining_modules(self, keep_decoder=False):
        if not keep_decoder:
            self.decoder = None


def get_annealed_rate(start, end, curr_step, total_steps):
    if curr_step >= total_steps:
        return end
    r = end - start
    pct_remaining = 1 - curr_step / total_steps
    return end - r * pct_remaining


def gather_unmasked(x: torch.Tensor, mask_info: MaskInfo) -> torch.Tensor:
    return torch.gather(
        x,
        dim=1,
        index=mask_info.ids_keep,
    )


def gather_unmasked_mask(x: torch.Tensor, mask_info: MaskInfo) -> torch.Tensor:
    return torch.gather(
        x,
        dim=1,
        index=mask_info.ids_keep[..., 0],  # ignore the feature dimension
    )


def get_alibi(
    max_positions: int,
    attention_heads: int,
    dims: int = 1,
    distance: str = "manhattan",
):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        # In the paper, we only train models that have 2^a heads for some
        # a. This function has some good properties that only occur when
        # the input is a power of 2. To maintain that even when the number
        # of heads is not a power of 2, we use this workaround.
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    maxpos = max_positions
    attn_heads = attention_heads
    slopes = torch.Tensor(get_slopes(attn_heads))

    if dims == 1:
        # prepare alibi position linear bias. Note that wav2vec2 is non
        # autoregressive model so we want a symmetric mask with 0 on the
        # diagonal and other wise linear decreasing valuees
        pos_bias = (
            torch.abs(
                torch.arange(maxpos).unsqueeze(0) - torch.arange(maxpos).unsqueeze(1)
            )
            * -1
        )
    elif dims == 2:
        if distance == "manhattan":
            df = lambda x1, y1, x2, y2: abs(x1 - x2) + abs(y1 - y2)
        elif distance == "euclidean":
            df = lambda x1, y1, x2, y2: math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        n = math.sqrt(max_positions)
        assert n.is_integer(), n
        n = int(n)

        pos_bias = torch.zeros((max_positions, max_positions))

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        new_x = i * n + j
                        new_y = k * n + l
                        pos_bias[new_x, new_y] = -df(i, j, k, l)

    else:
        raise Exception(f"unsupported number of alibi dims: {dims}")

    alibi_bias = slopes.unsqueeze(1).unsqueeze(1) * pos_bias.unsqueeze(0).expand(
        attn_heads, -1, -1
    )

    return alibi_bias


def get_alibi_bias(
    alibi_biases,
    batch_size,
    time_steps,
    heads,
    dtype,
    device,
    dims=1,
    distance="manhattan",
):
    cache_key = f"{dims}_{heads}_{distance}"

    buffered = alibi_biases.get(cache_key, None)

    target_size = heads * batch_size
    if (
        buffered is None
        or buffered.size(0) < target_size
        or buffered.size(1) < time_steps
        or buffered.dtype != dtype
        or buffered.device != device
    ):
        bt = max(time_steps, buffered.size(1) if buffered is not None else 0)
        bn = max(target_size, buffered.size(0) if buffered is not None else 0) // heads

        buffered = (
            get_alibi(bt, heads, dims=dims, distance=distance)
            .to(dtype=dtype, device=device)
            .repeat(bn, 1, 1)
        )

        alibi_biases[cache_key] = buffered

    b = buffered[:target_size, :time_steps, :time_steps]
    b = b.view(batch_size, heads, time_steps, time_steps)
    return b


def _learned_alibi_bias(
    alibi_bias,
    batch_size,
    time_steps,
    heads,
    scale,
    dtype,
    device,
):
    assert alibi_bias.size(1) == heads, alibi_bias.shape
    assert alibi_bias.dtype == dtype, alibi_bias.dtype
    assert alibi_bias.device == device, alibi_bias.device

    if alibi_bias.size(-1) < time_steps:
        psz = math.ceil((time_steps - alibi_bias.size(-1)) / 2)
        alibi_bias = F.pad(alibi_bias, (psz, psz, psz, psz), mode="replicate")

    alibi_bias = alibi_bias.expand(batch_size, -1, -1, -1) * scale
    return alibi_bias[..., :time_steps, :time_steps]


def masked_alibi(alibi_bias, mask_info):
    H = alibi_bias.size(1)

    orig_bias = alibi_bias

    index = mask_info.ids_keep.unsqueeze(1)[..., 0].unsqueeze(-1)
    alibi_bias = torch.gather(
        orig_bias,
        dim=-2,
        index=index.expand(-1, H, -1, mask_info.ids_restore.size(1)),
    )
    alibi_bias = torch.gather(
        alibi_bias,
        dim=-1,
        index=index.transpose(-1, -2).expand(-1, H, alibi_bias.size(-2), -1),
    )

    return alibi_bias
