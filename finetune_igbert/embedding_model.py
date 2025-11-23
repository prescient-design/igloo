# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

import os
from typing import Optional

import torch
from torch import nn
from collections import Counter
from typing import List, Optional, Tuple, Union

from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert.modeling_bert import (
    BertModel, BertForMaskedLM, MaskedLMOutput, CrossEntropyLoss,
    _prepare_4d_attention_mask_for_sdpa, _prepare_4d_causal_attention_mask_for_sdpa
)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from igloo.vqvae import VQVAE

from transformers import BertModel, BertForMaskedLM

class LoopTokBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config, loop_config, use_quantized_loop_embeddings=False, use_loop_tokens_whole_sequence=False):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.loop_embeddings = VQVAE(**loop_config)
        if self.loop_embeddings.embedding_dim != config.hidden_size:
            self.loop_embeddings_projection = nn.Linear(self.loop_embeddings.embedding_dim, config.hidden_size)
        else:
            self.loop_embeddings_projection = nn.Identity()
        self.use_quantized_loop_embeddings = use_quantized_loop_embeddings
        self.use_loop_tokens_whole_sequence = use_loop_tokens_whole_sequence
        self.hidden_size = config.hidden_size

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.LayerNorm_loops = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

        self.convert_ids_to_tokens = None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        angles: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        loop_embeddings = self.get_loop_embeddings(input_ids, angles)
        loop_embeddings = self.LayerNorm_loops(loop_embeddings)

        embeddings = inputs_embeds + token_type_embeddings + loop_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


    def get_loop_embeddings(self, input_ids, angles):
        # angles [batch_size, 4=(cdr1,2,4,3), 36=(loop_length), 3=(phi,psi,omega)]
        _, seq_length = input_ids.size()
        batch_size, num_loops, loop_length, num_angles = angles.size()
        alphabet = self.loop_embeddings.encoder.alphabet
        max_length = 36
        loops = []
        batch_start_loop_indexes = []
        batch_loop_indexes = []
        loop_special_token_masks = []
        for sequence in input_ids:
            sequence_tokens = self.convert_ids_to_tokens(sequence.tolist())
            curr_loop = []
            start_loop_indexes = []
            loop_indexes = []
            loop_start = False
            tok_count = Counter(sequence_tokens)
            for tok in ['[CDRs]', '[CDRe]']:
                assert tok_count.get(tok, 0) == 4, f"Expected exactly four occurrences of {tok} in the sequence, found {tok_count.get(tok, 0)}"
            for i, tok in enumerate(sequence_tokens):
                if tok == '[CDRs]':
                    loop_start = True
                    start_loop_indexes.append(i)
                    loop_indexes.append(0)
                elif tok == '[CDRe]':
                    loop_start = False
                    loop_indexes.append(0)
                    if curr_loop:
                        loop_special_token_masks.append([1] + [0] * len(curr_loop) + [1]*(max_length - len(curr_loop) - 1))
                        curr_loop = [alphabet.cls_idx] + curr_loop + [alphabet.eos_idx]
                        curr_loop = curr_loop + [alphabet.padding_idx] * (max_length - len(curr_loop))
                        loops.append(curr_loop)
                        curr_loop = []
                elif loop_start:
                    if tok == '[MASK]':
                        curr_loop.append(alphabet.mask_idx)
                    else:
                        curr_loop.append(alphabet.get_idx(tok))
                    loop_indexes.append(1)
                else:
                    loop_indexes.append(0)
                
            batch_start_loop_indexes.append(start_loop_indexes) 
            batch_loop_indexes.append(loop_indexes)
        loops = torch.tensor(loops, dtype=torch.long, device=input_ids.device)
        batch_start_loop_indexes = torch.tensor(batch_start_loop_indexes, dtype=torch.long, device=input_ids.device) # [batch_size, 4]
        loop_special_token_masks = torch.tensor(loop_special_token_masks, dtype=torch.long, device=input_ids.device) # [batch_size, 4, loop_length]
        batch_loop_indexes = torch.tensor(batch_loop_indexes, dtype=torch.long, device=input_ids.device) # [batch_size, seq_length]
        batch = {
            "sequence": loops,
            "angles": angles.view(batch_size*num_loops, loop_length, num_angles),
            "context": None,
        }
        loop_embeddings, loop_embeddings_quantized, _, sequence_embedding = self.loop_embeddings.inference(batch, return_encoded_sequence=True) # [batch_size * 4, loop_embedding_dim]
        if self.use_quantized_loop_embeddings:
            loop_embeddings = loop_embeddings_quantized
        if self.use_loop_tokens_whole_sequence: # add all loop embeddings across the whole sequence, this gives each loop amino acid token dihedral angle information
            sequence_embedding = self.loop_embeddings_projection(sequence_embedding) # [batch_size * 4, loop_length, hidden_size]
            sequence_embedding = sequence_embedding.view(batch_size, -1, loop_length, self.hidden_size) # [batch_size, 4, loop_length, hidden_size]
            loop_embeddings = sequence_embedding[:, :, 0, :] # [batch_size, 4, hidden_size]
        else:        
            loop_embeddings = self.loop_embeddings_projection(loop_embeddings) # [batch_size * 4, hidden_size]
            loop_embeddings = loop_embeddings.view(batch_size, -1, self.hidden_size) # [batch_size, 4, hidden_size]

        # batch_start_loop_indexes [batch_size, 4]
        loop_embeddings_all = torch.zeros(batch_size, seq_length, self.hidden_size, device=input_ids.device)
        batch_indices = torch.arange(batch_start_loop_indexes.size(0), device=loop_embeddings_all.device).unsqueeze(1).expand(-1, batch_start_loop_indexes.size(1))
        loop_embeddings_all[batch_indices, batch_start_loop_indexes] = loop_embeddings

        if self.use_loop_tokens_whole_sequence:
            loop_embeddings_all[batch_loop_indexes==1] = sequence_embedding[loop_special_token_masks.view(batch_size, -1, loop_length) == 0]

        return loop_embeddings_all # [batch_size, seq_length, hidden_size]


class BertLoopModelForMaskedLM(BertForMaskedLM):
    def __init__(self, config, loop_config, use_quantized_loop_embeddings, use_loop_tokens_whole_sequence):
        super().__init__(config)
        self.bert = BertLoopModel(config, loop_config, use_quantized_loop_embeddings, use_loop_tokens_whole_sequence)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        angles: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            angles=angles,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertLoopModel(BertModel):
    _no_split_modules = ["BertEmbeddings", "BertLayer"]

    def __init__(self, config, loop_config, use_quantized_loop_embeddings, use_loop_tokens_whole_sequence, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.embeddings = LoopTokBertEmbeddings(config, loop_config, use_quantized_loop_embeddings, use_loop_tokens_whole_sequence)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        angles: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, target_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            angles=angles,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)

        use_sdpa_attention_masks = (
            self.attn_implementation == "sdpa"
            and self.position_embedding_type == "absolute"
            and head_mask is None
            and not output_attentions
        )

        # Expand the attention mask
        if use_sdpa_attention_masks and attention_mask.dim() == 2:
            # Expand the attention mask for SDPA.
            # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
            if self.config.is_decoder:
                extended_attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    input_shape,
                    embedding_output,
                    past_key_values_length,
                )
            else:
                extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
        else:
            # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
            # ourselves in which case we just need to make it broadcastable to all heads.
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if use_sdpa_attention_masks and encoder_attention_mask.dim() == 2:
                # Expand the attention mask for SDPA.
                # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
                encoder_extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    encoder_attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
