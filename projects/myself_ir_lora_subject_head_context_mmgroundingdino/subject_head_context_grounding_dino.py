"""Grounding DINO with frozen subject-head routing for proposal selection."""

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from projects.myself_ir_lora_addition_mmgroundingdino import (
    IRLoRAAdditionGroundingDINO)


class SubjectHeadClassifier(nn.Module):
    """Score word-start BERT tokens as the referent head."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1))

    def forward(self, hidden: Tensor, candidate_mask: Tensor) -> Tensor:
        logits = self.classifier(hidden).squeeze(-1)
        return logits.masked_fill(~candidate_mask.bool(), -1e4)


@MODELS.register_module()
class SubjectHeadContextGroundingDINO(IRLoRAAdditionGroundingDINO):
    """Select encoder proposals using one frozen predicted head word."""

    def __init__(self,
                 *args,
                 subject_head_classifier_checkpoint: str,
                 subject_head_hidden_dim: int = 256,
                 subject_head_dropout: float = 0.1,
                 context_text_source: str = 'bert',
                 **kwargs) -> None:
        if context_text_source not in {'bert', 'encoder'}:
            raise ValueError('context_text_source must be bert or encoder.')
        super().__init__(*args, **kwargs)
        self.context_text_source = context_text_source
        self.subject_head_classifier_checkpoint = str(
            subject_head_classifier_checkpoint)
        language_dim = \
            self.language_model.language_backbone.body.language_dim
        self.subject_head_classifier = SubjectHeadClassifier(
            language_dim, subject_head_hidden_dim, subject_head_dropout)
        self.subject_head_classifier.requires_grad_(False)
        self.subject_head_classifier.eval()
        self._subject_head_weights_loaded = False

    def init_weights(self) -> None:
        super().init_weights()
        checkpoint = Path(self.subject_head_classifier_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f'Subject-head checkpoint not found: {checkpoint}')
        state_dict = torch.load(checkpoint, map_location='cpu')
        self.subject_head_classifier.load_state_dict(state_dict, strict=True)
        self.subject_head_classifier.requires_grad_(False)
        self.subject_head_classifier.eval()
        self._subject_head_weights_loaded = True

    def train(self, mode: bool = True):
        super().train(mode)
        self.subject_head_classifier.eval()
        return self

    @staticmethod
    def _is_content_piece(token: str) -> bool:
        piece = token[2:] if token.startswith('##') else token
        return any(character.isalnum() for character in piece)

    def _build_subject_and_context_masks(
            self, text_dict: Dict) -> Dict[str, Tensor]:
        hidden = text_dict['hidden']
        input_ids = text_dict['input_ids']
        valid_mask = text_dict['text_token_mask'].bool()
        special_mask = text_dict['special_tokens_mask'].bool()
        offsets = text_dict['offset_mapping']
        batch_size, text_length = valid_mask.shape

        candidate_mask = torch.zeros_like(valid_mask)
        content_mask = torch.zeros_like(valid_mask)
        article_mask = torch.zeros_like(valid_mask)
        tokens_by_image = []
        for image_id in range(batch_size):
            tokens = self.language_model.tokenizer.convert_ids_to_tokens(
                input_ids[image_id].detach().cpu().tolist())
            tokens_by_image.append(tokens)
            for token_id, token in enumerate(tokens[:text_length]):
                is_content = self._is_content_piece(token)
                content_mask[image_id, token_id] = is_content
                is_word_start = not token.startswith('##')
                candidate_mask[image_id, token_id] = (
                    is_content and is_word_start)
                article_mask[image_id, token_id] = (
                    is_word_start and token.lower() in {'a', 'an', 'the'})

        candidate_mask &= valid_mask & ~special_mask
        content_mask &= valid_mask & ~special_mask

        with torch.no_grad():
            head_logits = self.subject_head_classifier(
                hidden, candidate_mask)
            head_probabilities = head_logits.softmax(dim=-1)

        head_mask = torch.zeros_like(valid_mask)
        head_valid = candidate_mask.any(dim=-1)
        head_token_index = torch.full(
            (batch_size, ), -1, dtype=torch.long, device=hidden.device)
        head_confidence = hidden.new_zeros((batch_size, ))
        head_char_span = torch.full(
            (batch_size, 2), -1, dtype=torch.long, device=hidden.device)

        for image_id in range(batch_size):
            if head_valid[image_id]:
                start_index = int(head_logits[image_id].argmax().item())
                head_token_index[image_id] = start_index
                head_confidence[image_id] = \
                    head_probabilities[image_id, start_index]
                end_index = start_index + 1
                tokens = tokens_by_image[image_id]
                while (end_index < text_length
                       and tokens[end_index].startswith('##')
                       and valid_mask[image_id, end_index]):
                    end_index += 1
                head_mask[image_id, start_index:end_index] = True
                head_char_span[image_id, 0] = offsets[
                    image_id, start_index, 0]
                head_char_span[image_id, 1] = offsets[
                    image_id, end_index - 1, 1]
            else:
                # A caption without a word-start candidate falls back to the
                # original content-token proposal ranking and skips B1.
                head_mask[image_id] = content_mask[image_id]

        context_mask = (
            content_mask & ~article_mask & ~head_mask &
            head_valid.unsqueeze(-1))
        return dict(
            subject_head_mask=head_mask,
            context_token_mask=context_mask,
            subject_head_valid=head_valid,
            subject_head_token_index=head_token_index,
            subject_head_confidence=head_confidence,
            subject_head_char_span=head_char_span)

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        language_routing = self._build_subject_and_context_masks(text_dict)
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)
        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)
        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict,
            subject_head_mask=language_routing['subject_head_mask'],
            batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)
        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        head_inputs_dict.update(language_routing)
        context_text = (text_dict['hidden']
                        if self.context_text_source == 'bert'
                        else encoder_outputs_dict['memory_text'])
        # B1 reads language evidence but cannot update either language path.
        head_inputs_dict['context_text'] = context_text.detach()
        return head_inputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        subject_head_mask: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        batch_size, _, _ = memory.shape
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        encoder_branch = self.decoder.num_layers
        enc_outputs_class = self.bbox_head.cls_branches[encoder_branch](
            output_memory, memory_text, text_token_mask)
        cls_out_features = self.bbox_head.cls_branches[
            encoder_branch].max_text_len
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            encoder_branch](output_memory) + output_proposals

        text_length = subject_head_mask.size(1)
        selection_logits = enc_outputs_class[..., :text_length].masked_fill(
            ~subject_head_mask[:, None, :], float('-inf')).max(dim=-1).values
        topk_indices = torch.topk(
            selection_logits, k=self.num_queries, dim=1).indices
        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, batch_size, 1).transpose(0, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat(
                [dn_bbox_query, topk_coords_unact], dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask)
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        return decoder_inputs_dict, head_inputs_dict
