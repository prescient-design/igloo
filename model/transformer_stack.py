# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from .modules import RobertaLMHead, TransformerLayer
from dataset import Alphabet, proteinseq_toks

class LoopTransformer(nn.Module):
    def __init__(
        self,
        num_layers: int = 3,
        embed_dim: int = 1024,
        attention_heads: int = 4,
        token_dropout: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        alphabet = Alphabet(standard_toks=proteinseq_toks)
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.token_dropout = token_dropout
        self.context_dim = 408  # Dimension of the context vector

        self._init_submodules()

    def _init_submodules(self):
        self.embed_scale = 1
        self.embed_tokens = nn.Embedding(
            self.alphabet_size,
            self.embed_dim,
            padding_idx=self.padding_idx,
        )

        self.dihedral_projection = nn.Linear(6, self.embed_dim)

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.embed_dim,
                    4 * self.embed_dim,
                    self.attention_heads,
                    add_bias_kv=False,
                    use_rotary_embeddings=True,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.emb_layer_norm_after = LayerNorm(self.embed_dim)

        self.lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )

        self.dihedral_decoder = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 6),
        )

        self.context_encoder = nn.Linear(self.context_dim, self.embed_dim)

    def forward(self, tokens, dihedral_angles, dihedral_angles_mask=None, context=None, repr_layers=[], need_head_weights=False):
        assert tokens.ndim == 2
        assert dihedral_angles.ndim == 3
        padding_mask = tokens.eq(self.padding_idx)  # B, T
        special_tokens_mask = tokens.eq(self.cls_idx) | tokens.eq(self.eos_idx) | tokens.eq(self.padding_idx)  # B, T

        x_tokens = self.embed_scale * self.embed_tokens(tokens)

        cos_angles = torch.cos(dihedral_angles)  # B x T x 3
        sin_angles = torch.sin(dihedral_angles)  # B x T x 3
        combined_angles = torch.cat((cos_angles.unsqueeze(-1), sin_angles.unsqueeze(-1)), dim=-1)  # B x T x 3 x 2
        dihedral_angles = combined_angles.view(dihedral_angles.shape[0], dihedral_angles.shape[1], -1) # B x T x 6
        x_dihedral = self.dihedral_projection(dihedral_angles) # dihedral_angles: B x T x 6
        x_dihedral[special_tokens_mask] = 0.0  # Set dihedral angles to zero for special tokens

        if self.token_dropout:
            x_tokens.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            if dihedral_angles_mask is not None:
                x_dihedral.masked_fill_(dihedral_angles_mask.unsqueeze(-1), 0.0)
            # x: B x T x C
            # mask_ratio_train = 0.3
            # src_lengths = (~padding_mask).sum(-1)
            # mask_ratio_observed = (tokens == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
            # x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]
        
        x = x_tokens + x_dihedral # B x T x E

        if context is not None:
            x_context = self.context_encoder(context)  # B x 9 x E, 9 context tokens
            x = torch.cat((x_context, x), dim=1)  # B x (9 + T) x E
            padding_mask = torch.cat(
                (torch.zeros((x_context.shape[0], x_context.shape[1]), dtype=padding_mask.dtype, device=padding_mask.device), padding_mask),
                dim=1
            ) # B x (9 + T)
        else:
            padding_mask = padding_mask

        x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        # last hidden representation should have layer norm applied
        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        
        if context is not None:
            # Remove context tokens from the output
            x = x[:, context.shape[1]:, :]
            padding_mask = padding_mask[:, context.shape[1]:]  # Remove context tokens from padding mask

        x_tokens_logits = self.lm_head(x)

        x_dihedral_unit_circle = self.dihedral_decoder(x).reshape(
            x.shape[0], x.shape[1], 3, 2
        ) # B x T x 3 x 2
        x_dihedral_norm = torch.norm(x_dihedral_unit_circle, dim=-1, p=2) # B x T x 3
        dihedral_regularizer = torch.mean(x_dihedral_norm[~padding_mask]**2 - 1.0)

        x_dihedral_angles = torch.zeros(x.shape[0], x.shape[1], 3, device=x.device) # B x T x 3
        mask = ~special_tokens_mask.unsqueeze(-1).expand_as(x_dihedral_angles)
        x_dihedral_sine = x_dihedral_unit_circle[..., 1][mask] / x_dihedral_norm[mask]
        x_dihedral_cosine = x_dihedral_unit_circle[..., 0][mask] / x_dihedral_norm[mask]
        x_dihedral_angles[mask] = torch.arctan2(x_dihedral_sine, x_dihedral_cosine)

        result = {
            "logits": x_tokens_logits,
            "representations": hidden_representations,
            "dihedral_regularizer": dihedral_regularizer,
            "dihedral_angles": x_dihedral_angles,
            "special_tokens_mask": special_tokens_mask,
        }
        return result


if __name__ == "__main__":
    # Example usage
    model = LoopTransformer(num_layers=3, embed_dim=128, attention_heads=4)
    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    tokens = torch.randint(0, 20, (2, 50))  # Batch of 2 sequences of length 50
    dihedral_angles = torch.randn(2, 50, 3)  # Batch of 2 sequences of length 50 with 3 dihedral angles
    output = model(tokens, dihedral_angles, repr_layers=[0, 1, 2])
    print(output["logits"].shape)  # Should print: torch.Size([2, 50, 20])
    print(output["representations"].keys())  # Should print: dict_keys([0, 1, 2])
    print(output["dihedral_regularizer"].item())  # Should print a scalar value
    print(output["dihedral_angles"].shape)  # Should print: torch.Size([2, 50, 3])