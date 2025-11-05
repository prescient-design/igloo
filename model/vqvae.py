import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize
from evals.metrics import dihedral_distance_pairwise
from evals.align_loops import length_independent_dihedral_label
import json
from dataclasses import dataclass
from .transformer_stack import LoopTransformer
import numpy as np

@dataclass
class VQVAEOutput:
    loss: torch.Tensor
    reconstructed_angles: torch.Tensor
    encoded: torch.Tensor
    commit_loss: torch.Tensor
    recon_loss: torch.Tensor
    codebook_loss: torch.Tensor
    unit_circle_regularization: torch.Tensor
    dihedral_loss: torch.Tensor
    loop_length_loss: torch.Tensor
    pred_loop_length_loss: torch.Tensor
    quantized: torch.Tensor
    quantized_indices: torch.Tensor
    aa_loss: torch.Tensor
    pred_aa: torch.Tensor
    true_aa: torch.Tensor

    def to_dict(self):
        return {
            'loss': self.loss,
            'reconstructed_angles': self.reconstructed_angles,
            'encoded': self.encoded,
            'commit_loss': self.commit_loss,
            'recon_loss': self.recon_loss,
            'codebook_loss': self.codebook_loss,
            'unit_circle_regularization': self.unit_circle_regularization,
            'dihedral_loss': self.dihedral_loss,
            'loop_length_loss': self.loop_length_loss,
            'pred_loop_length_loss': self.pred_loop_length_loss,
            'quantized': self.quantized,
            'quantized_indices': self.quantized_indices,
            'aa_loss': self.aa_loss,
            'pred_aa': self.pred_aa,
            'true_aa': self.true_aa,
        }


def length_independent_dihedral_update(batch, dihedral_distance_batch, special_tokens_mask, max_residue_difference=1):
    length_difference = np.abs(batch['sequence_length'].cpu().numpy()[:, None] - batch['sequence_length'].cpu().numpy()[None, :])  # [B, B]
    dihedral_distance_batch[(length_difference > max_residue_difference)] = 4.0  # Only consider pairs with similar lengths
    row_indices, col_indices = np.where(length_difference <= max_residue_difference)  # Indices of pairs with similar lengths

    batch_angles = batch['angles'].cpu().numpy()  # [B, T, 3]
    batch_loop_coords = batch['loop_c_alpha_coords'].cpu().numpy()  # [B, T, 3]
    batch_stem_coords = batch['stem_c_alpha_coords'].cpu().numpy()
    special_tokens_mask_ = special_tokens_mask.cpu().numpy()  # [B, T]

    positive_examples = 0
    for r, c in zip(row_indices, col_indices):
        if r >= c:
            continue # matrix is symmetric, skip the upper triangle and diagonal
        D = length_independent_dihedral_label(
            angles1=batch_angles[r][~special_tokens_mask_[r]],
            angles2=batch_angles[c][~special_tokens_mask_[c]],
            loop_coords1=batch_loop_coords[r][~special_tokens_mask_[r]],
            loop_coords2=batch_loop_coords[c][~special_tokens_mask_[c]],
            stem_coords1=batch_stem_coords[r],
            stem_coords2=batch_stem_coords[c],
            max_residue_difference=max_residue_difference,
        )
        dihedral_distance_batch[r, c] = D
        dihedral_distance_batch[c, r] = D  # symmetric matrix
        positive_examples += 1 if D < 0.1 else 0
    batch_size = len(batch['angles'])
    prop_positives = positive_examples / ((batch_size**2-batch_size)/ 2)
    return dihedral_distance_batch


class VQVAE(nn.Module):
    def __init__(self, embedding_dim: int=128, codebook_size: int=128, num_encoder_layers: int=3, 
                 num_attention_heads: int=4, orthog_regularization: bool=False,
                 dihedral_temperature: float=0.1, dihedral_loss: bool=True,
                 unit_circle_transform_weight: float=0.01, commit_loss_weight: float=0.25,
                 loop_length_tolerance: int=0, loop_length_loss: bool=True,
                 loop_length_temperature: float=0.5, loop_length_pred_loss: bool=False,
                 learnable_codebook: bool=False):
        """
        Vector Quantized Variational Autoencoder (VQVAE) model.
        """
        super(VQVAE, self).__init__()
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.dihedral_temperature = dihedral_temperature
        self.dihedral_loss = dihedral_loss
        self.unit_circle_transform_weight = unit_circle_transform_weight
        self.commit_loss_weight = commit_loss_weight
        self.num_encoder_layers = num_encoder_layers
        self.num_attention_heads = num_attention_heads
        self.loop_length_tolerance = loop_length_tolerance
        self.loop_length_loss = loop_length_loss
        self.loop_length_temperature = loop_length_temperature
        self.loop_length_pred_loss = loop_length_pred_loss

        # if loop_length_loss:
        #     self.loop_length_projector = nn.Sequential(
        #         nn.Linear(embedding_dim, embedding_dim//2),
        #         nn.LayerNorm(embedding_dim//2),
        #         nn.ReLU(),
        #         nn.Linear(embedding_dim//2, 1)
        #     )
        # else:
        #     self.loop_length_projector = None

        self.loop_length_projector = nn.Linear(embedding_dim, 34) if loop_length_loss else None

        self.encoder = LoopTransformer(
            num_layers = num_encoder_layers,
            embed_dim = embedding_dim, 
            attention_heads = num_attention_heads,
        )
        
        self.orthog_regularization = orthog_regularization
        quantizer_args = {
            'dim': embedding_dim,
            'codebook_size': codebook_size,
            'learnable_codebook': learnable_codebook,
            'ema_update': not learnable_codebook,  # EMA update only if not using a learnable codebook
            'commitment_weight': commit_loss_weight,
            # 'threshold_ema_dead_code': 10,
        }
        if orthog_regularization:
            quantizer_args.update({
                'orthogonal_reg_weight': 10,
                'orthogonal_reg_max_codes': self.codebook_size,
            })
        self.quantizer = VectorQuantize(**quantizer_args)

    def get_config(self):
        return {
            'embedding_dim': self.embedding_dim,
            'codebook_size': self.codebook_size,
            'num_encoder_layers': self.num_encoder_layers,
            'orthog_regularization': self.orthog_regularization,
            'dihedral_temperature': self.dihedral_temperature,
            'dihedral_loss': self.dihedral_loss,
            'unit_circle_transform_weight': self.unit_circle_transform_weight,
            'commit_loss_weight': self.commit_loss_weight,
            'loop_length_tolerance': self.loop_length_tolerance,
            'num_attention_heads': self.num_attention_heads,
            'loop_length_loss': self.loop_length_loss,
            'loop_length_temperature': self.loop_length_temperature,
            'loop_length_pred_loss': self.loop_length_pred_loss,
            'learnable_codebook': self.quantizer.learnable_codebook,
        }

    def get_distance(self, x, y):
        # x: [b, embedding_dim]
        # y: [k, embedding_dim]
        x_expanded = x.unsqueeze(1)  # Shape [b, 1, embedding_dim]
        y_expanded = y.unsqueeze(0)  # Shape [1, k, embedding_dim]
        squared_diff = (x_expanded - y_expanded) ** 2
        squared_distances = torch.sum(squared_diff, dim=-1)
        distances = torch.sqrt(squared_distances)
        return distances # [b, k]

    @classmethod
    def load_from_config_and_weights(cls, config_path, weights_path, strict=True):
        with open(config_path, 'r') as f:
            config = json.load(f)
        model = cls(**config)
        model.load_state_dict(torch.load(weights_path), strict=strict)
        return model
    
    def forward(self, batch, val = False):
        masked_aa = batch['sequence'].eq(self.encoder.alphabet.mask_idx)
        true_aa = batch['true_sequence'][masked_aa].detach().cpu()
        special_tokens_mask = batch['sequence'].eq(self.encoder.alphabet.cls_idx) | batch['sequence'].eq(self.encoder.alphabet.eos_idx) | batch['sequence'].eq(self.encoder.alphabet.padding_idx)  # B, T

        result = self.encoder(batch['sequence'], batch['angles'], dihedral_angles_mask=batch["angles_mask"], context=batch['context'], repr_layers=[self.num_encoder_layers])
        encoded_sequence = result['representations'][self.num_encoder_layers]  # [batch_size, num_aa, embedding_dim]
        encoded_cls = encoded_sequence[:, 0, :]  # [batch_size, embedding_dim]

        if self.dihedral_loss:
            with torch.no_grad():
                dihedral_distance_batch = dihedral_distance_pairwise(
                    batch['angles'].cpu().numpy(),
                    mask=~special_tokens_mask.cpu().numpy()
                )  # [batch_size, batch_size], mask needed as special tokens have dihedral angles of 0 which makes the dihedral distance seem closer than it is
                
                dihedral_distance_batch = length_independent_dihedral_update(batch, dihedral_distance_batch, special_tokens_mask, max_residue_difference=self.loop_length_tolerance)
                dihedral_distance_batch = torch.tensor(dihedral_distance_batch, device=batch['angles'].device)  # Convert to tensor and move to device
                Y = torch.zeros_like(dihedral_distance_batch, device=dihedral_distance_batch.device)
                Y[dihedral_distance_batch < 0.1] = 1.0  # within threshold
                Y[dihedral_distance_batch > 0.47] = 0.0  # outside threshold
                ambiguous_mask = (dihedral_distance_batch >= 0.1) & (dihedral_distance_batch <= 0.47)
                Y[ambiguous_mask] = 0.5  # ambiguous cases

        quantized, quantized_indices, commit_loss = self.quantizer(encoded_cls)
        quantized_st = encoded_cls + (quantized - encoded_cls).detach()

        if self.dihedral_loss:
            E_norm = F.normalize(encoded_cls, p=2, dim=1)
            S = torch.matmul(E_norm, E_norm.T) # [B, B]
            logits = S / self.dihedral_temperature  # higher temp = softer
            mask = ~torch.eye(encoded_cls.size(0), dtype=torch.bool, device=encoded_cls.device)
            mask = mask & (Y != 0.5) # only consider non-ambiguous pairs
            dihedral_loss = F.binary_cross_entropy_with_logits(logits[mask], Y[mask].float())
        else:
            dihedral_loss = torch.tensor(0.0, device=encoded_cls.device)

        if self.loop_length_loss:
            E_norm = F.normalize(encoded_cls, p=2, dim=1)
            S = torch.matmul(E_norm, E_norm.T)  # [B, B]
            logits = S / self.loop_length_temperature  # higher temp = softer
            mask = ~torch.eye(encoded_cls.size(0), dtype=torch.bool, device=encoded_cls.device)
            length_difference = torch.abs(batch['sequence_length'][:, None] - batch['sequence_length'][None, :])  # [B, B]
            Y_length = (length_difference <= self.loop_length_tolerance).float()
            
            weights = torch.zeros_like(Y_length, device=Y_length.device)
            weights[Y_length == 0.0] = 2.0
            loop_length_loss = F.binary_cross_entropy_with_logits(logits[mask], Y_length[mask], weight=weights[mask])
        else:
            loop_length_loss = torch.tensor(0.0, device=encoded_cls.device)
        
        if self.loop_length_pred_loss:
            pred_loop_length = self.loop_length_projector(encoded_cls) # [B, 34]
            true_loop_length = batch['sequence_length'].float() - 1 # length is [1,34]
            pred_loop_length_loss = F.cross_entropy(pred_loop_length, true_loop_length.long(), reduction='mean')  # [B, 34] vs [B, 1]
            # pred_loop_length_loss = F.mse_loss(pred_loop_length, true_loop_length) 
            # instead of MSE, try binary cross entropy
        else:
            pred_loop_length_loss = torch.tensor(0.0, device=encoded_cls.device)

        logits_aa = result["logits"]
        aa_loss = F.cross_entropy(logits_aa[masked_aa], true_aa.to(logits_aa.device))
        pred_aa = torch.softmax(logits_aa[masked_aa], dim=-1).detach().cpu()

        masked_dihedrals = batch["angles_mask"]
        recon_cos = F.mse_loss(torch.cos(result["dihedral_angles"][masked_dihedrals]), torch.cos(batch['angles'][masked_dihedrals]), reduction='none')
        recon_sin = F.mse_loss(torch.sin(result["dihedral_angles"][masked_dihedrals]), torch.sin(batch['angles'][masked_dihedrals]), reduction='none')
        recon_loss = recon_sin + recon_cos
        recon_loss = recon_loss.mean()

        # recon_loss = F.mse_loss(result["dihedral_angles"][masked_aa], batch['angles'][masked_aa]) # [phi, psi, omega] values
        codebook_loss = F.mse_loss(encoded_cls.detach(), quantized)

        if val:
            codebook_loss = torch.tensor(0.0, device=encoded_cls.device)
            unit_circle_regularization = torch.tensor(0.0, device=encoded_cls.device)
        else:
            unit_circle_regularization = result["dihedral_regularizer"] * self.unit_circle_transform_weight

        loss = commit_loss + recon_loss + codebook_loss + unit_circle_regularization + aa_loss + dihedral_loss + loop_length_loss + pred_loop_length_loss

        return VQVAEOutput(
            loss=loss,
            reconstructed_angles=result["dihedral_angles"],
            encoded=encoded_cls,
            commit_loss=commit_loss,
            recon_loss=recon_loss,
            codebook_loss=codebook_loss,
            loop_length_loss=loop_length_loss,
            pred_loop_length_loss=pred_loop_length_loss,
            unit_circle_regularization=unit_circle_regularization,
            dihedral_loss=dihedral_loss,
            quantized=quantized,
            quantized_indices=quantized_indices,
            aa_loss=aa_loss,
            pred_aa=pred_aa,
            true_aa=true_aa,
        )


    def inference_sequence_only(self, batch):
        special_tokens_mask = batch['sequence'].eq(self.encoder.alphabet.cls_idx) | batch['sequence'].eq(self.encoder.alphabet.eos_idx) | batch['sequence'].eq(self.encoder.alphabet.padding_idx)  # B, T
        result = self.encoder(batch['sequence'], batch['angles'], dihedral_angles_mask=~special_tokens_mask, context=batch['context'], repr_layers=[self.num_encoder_layers])
        encoded_sequence = result['representations'][self.num_encoder_layers]  # [batch_size, num_aa, embedding_dim]
        encoded_cls = encoded_sequence[:, 0, :]  # [batch_size, embedding_dim]

        quantized, quantized_indices, commit_loss = self.quantizer(encoded_cls)
        return encoded_cls, quantized, quantized_indices


    def inference(self, batch, return_encoded_sequence=False):
        # batch['sequence'] = tokenized input sequence, [batch_size, max_length]
        # batch['angles'] = dihedral angles, [batch_size, max_length, 3]
        # batch['context'] can be None
        result = self.encoder(batch['sequence'], batch['angles'], context=batch['context'], repr_layers=[self.num_encoder_layers])
        encoded_sequence = result['representations'][self.num_encoder_layers]  # [batch_size, num_aa, embedding_dim]
        encoded_cls = encoded_sequence[:, 0, :]  # [batch_size, embedding_dim]

        quantized, quantized_indices, commit_loss = self.quantizer(encoded_cls)
        if return_encoded_sequence:
            return encoded_cls, quantized, quantized_indices, encoded_sequence
        return encoded_cls, quantized, quantized_indices