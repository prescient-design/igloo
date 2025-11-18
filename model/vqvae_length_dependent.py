import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize
from .metrics import dihedral_distance
import json
from dataclasses import dataclass

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
    quantized: torch.Tensor
    quantized_indices: torch.Tensor
    aa_loss: torch.Tensor
    pred_aa: torch.Tensor

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
            'quantized': self.quantized,
            'quantized_indices': self.quantized_indices,
            'aa_loss': self.aa_loss,
            'pred_aa': self.pred_aa,
        }

def angles_to_unit_circle(angles):
    x = torch.cos(angles) # [batch_size, L]
    y = torch.sin(angles) # [batch_size, L]
    coords = torch.stack((x, y), dim=-1) # [batch_size, L, 2]
    return coords.view(angles.shape[0], -1)  # Flatten to [batch_size, 2 * L]

def unit_circle_to_angles(unit_circle):
    x = unit_circle[:, ::2]  # Extract x coordinates
    y = unit_circle[:, 1::2]  # Extract y coordinates
    x_normalized = x / (x**2 + y**2).sqrt()  # Normalize x
    y_normalized = y / (x**2 + y**2).sqrt()  # Normalize y
    angles = torch.atan2(y_normalized, x_normalized)  # Compute angles from coordinates
    return angles


def nt_xent_loss(encoded, encoded_aug, temperature: float = 0.5):
    """
    encoded       : Tensor [B, D] – output of encoder for the original view
    encoded_aug   : Tensor [B, D] – output of encoder for the augmented view
    temperature   : Scaling factor (τ). Typical values 0.1–0.5
    
    Returns:
        Scalar loss (mean over 2B positives)
    """

    # 1. L2-normalise so that dot-product = cosine similarity
    z1 = F.normalize(encoded,     dim=1)      # [B, D]
    z2 = F.normalize(encoded_aug, dim=1)      # [B, D]

    # 2. Build the 2B × D matrix and the full similarity matrix
    reps = torch.cat([z1, z2], dim=0)         # [2B, D]
    sim  = torch.matmul(reps, reps.t())       # [2B, 2B]   cosine similarity
    sim  = sim / temperature                  # scale by τ

    B = encoded.size(0)
    diag = torch.eye(2*B, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(diag, -9e15)        # mask self-similarities

    # 3. Positive pairs are (i, i+B) and (i+B, i)
    pos_mask = torch.zeros_like(sim, dtype=torch.bool)
    pos_mask[:B,   B:2*B] = torch.eye(B, dtype=torch.bool, device=sim.device)
    pos_mask[B:2*B, :B]  = torch.eye(B, dtype=torch.bool, device=sim.device)

    # 4. Compute the loss
    positives   = sim[pos_mask].view(2*B, 1)                          # [2B, 1]
    denominator = torch.logsumexp(sim, dim=1, keepdim=True)           # [2B, 1]
    loss = (-positives + denominator).mean()

    return loss


class VQVAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int=64, embedding_dim: int=128, 
                 codebook_size: int=128, num_encoder_layers: int=3, num_decoder_layers: int=3,
                 vae_mode: bool=False, topk: int=1, orthog_regularization: bool=False,
                 dihedral_temperature: float=0.1, dihedral_loss: bool=True,
                 unit_circle_transform: bool=True, unit_circle_transform_weight: float=0.01,
                 commit_loss_weight: float=0.25):
        """
        Vector Quantized Variational Autoencoder (VQVAE) model.
        """
        super(VQVAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.vae_mode = vae_mode
        self.topk = topk # if topk > 1, quantized embeddings are a weighted average of the topk nearest neighbors in the codebook
        self.dihedral_temperature = dihedral_temperature
        self.dihedral_loss = dihedral_loss
        self.unit_circle_transform = unit_circle_transform
        self.unit_circle_transform_weight = unit_circle_transform_weight
        self.commit_loss_weight = commit_loss_weight

        self.num_aa = self.input_dim // 23

        if self.unit_circle_transform:
            input_dim = input_dim + self.num_aa * 3  # Each angle turns into (x,y) coordinates

        # Encoder
        encoder_layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        ]
        for _ in range(num_encoder_layers-2):
            encoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
        encoder_layers.extend([
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU()
        ])
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Vector Quantizer
        if vae_mode:
            self.quantizer = None
        else:
            self.orthog_regularization = orthog_regularization
            quantizer_args = {
                'dim': embedding_dim,
                'codebook_size': codebook_size,
                'learnable_codebook': False,
                'ema_update': True,
                'commitment_weight': commit_loss_weight,
            }
            if orthog_regularization:
                quantizer_args.update({
                    'orthogonal_reg_weight': 10,
                    'orthogonal_reg_max_codes': self.codebook_size,
                })
            self.quantizer = VectorQuantize(**quantizer_args)

        # Decoder
        decoder_layers = [
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU()
        ]
        for _ in range(num_decoder_layers-2):
            decoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            embedding_dim = hidden_dim
        decoder_layers.append(nn.Linear(hidden_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def get_config(self):
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'embedding_dim': self.embedding_dim,
            'codebook_size': self.codebook_size,
            'num_encoder_layers': self.num_encoder_layers,
            'num_decoder_layers': self.num_decoder_layers,
            'vae_mode': self.vae_mode,
            'topk': self.topk,
            'orthog_regularization': self.orthog_regularization,
            'dihedral_temperature': self.dihedral_temperature,
            'dihedral_loss': self.dihedral_loss,
            'unit_circle_transform': self.unit_circle_transform,
            'unit_circle_transform_weight': self.unit_circle_transform_weight,
            'commit_loss_weight': self.commit_loss_weight,
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
        # x: [batch_size, input_dim]
        batch_size = batch['angles'].shape[0]

        gt_aa = batch['sequence'].detach().reshape(batch_size, self.num_aa, 20) # [batch_size, num_aa, 20]
        gt_aa = torch.argmax(gt_aa, dim=-1)  # [batch_size, num_aa]

        # randomly mask sequences
        if val:
            mask_prop = 1.0
        else:
            mask_prop = 0.3

        aa_mask = torch.rand(batch_size, self.num_aa) < mask_prop  # [batch_size, num_aa]
        batch['sequence'] = batch['sequence'].reshape(batch_size, self.num_aa, 20)  # Reshape to [batch_size, num_aa, 20]
        batch['sequence'][aa_mask] = 0.0  # Set masked amino acids to zero
        batch['sequence'] = batch['sequence'].reshape(batch_size, -1)  # Reshape back to [batch_size, num_aa * 20]

        if self.dihedral_loss:
            with torch.no_grad():
                dihedral_distance_batch = dihedral_distance(batch['angles'].cpu().numpy(), batch['angles'].cpu().numpy())  # [batch_size, batch_size]
                dihedral_distance_batch = torch.tensor(dihedral_distance_batch, device=batch['angles'].device)  # Convert to tensor and move to device
                Y = torch.zeros_like(dihedral_distance_batch, device=dihedral_distance_batch.device)
                Y[dihedral_distance_batch < 0.1] = 1.0  # within threshold
                Y[dihedral_distance_batch > 0.47] = 0.0  # outside threshold
                ambiguous_mask = (dihedral_distance_batch >= 0.1) & (dihedral_distance_batch <= 0.47)
                Y[ambiguous_mask] = 0.5  # ambiguous cases
        

        if self.unit_circle_transform:
            batch_unit_circle = angles_to_unit_circle(batch['angles'])
            x = torch.cat((batch_unit_circle, batch['sequence']), dim=1)
            x_aug = torch.cat((batch_unit_circle, torch.zeros_like(batch['sequence'])), dim=1)
        else:
            x = torch.cat((batch['angles'], batch['sequence']), dim=1)  # [batch_size, input_dim]
            x_aug = torch.cat((batch['angles'], torch.zeros_like(batch['sequence'])), dim=1)  # [batch_size, input_dim]

        encoded = self.encoder(x) # [batch_size, embedding_dim]
        if val:
            x = x_aug # mask out sequence during validation
            encoded = self.encoder(x) # [batch_size, embedding_dim]
            aug_loss = torch.tensor(0.0, device=x.device)  # No augmentation loss during validation
        else:
            # ensure encoded and encoded_aug end up in the same space
            encoded = self.encoder(x) # [batch_size, embedding_dim]
            encoded_aug = self.encoder(x_aug)  # [batch_size, embedding_dim]
            aug_loss = F.mse_loss(encoded, encoded_aug)  # Mean Squared Error between original and augmented encodings, do not detach encoded (makes training unstable)
            # aug_loss = nt_xent_loss(encoded, encoded_aug, temperature=0.1)

        if self.vae_mode:
            quantized = encoded
            quantized_indices = torch.zeros(encoded.shape[0], dtype=torch.long, device=encoded.device)
            commit_loss = torch.zeros(1, device=encoded.device)
            quantized_st = quantized
        else:
            if self.topk == 1:
                quantized, quantized_indices, commit_loss = self.quantizer(encoded)
            else:
                codebook = self.quantizer._codebook.embed.squeeze(0)  # [codebook_size, embedding_dim]
                distances = self.get_distance(encoded, codebook)  # [batch_size, codebook_size]
                topk_distances, topk_indices = torch.topk(distances, self.topk, dim=1, largest=False)
                topk_vectors = codebook[topk_indices]  # [batch_size, topk, embedding_dim]
                
                # Calculate weights inversely proportional to distances (e.g., softmax)
                # We use negative distances for softmax to emphasize smaller distances
                weights = torch.nn.functional.softmax(-topk_distances, dim=1)  # [batch_size, topk]
                quantized = torch.sum(topk_vectors * weights.unsqueeze(-1), dim=1)  # [batch_size, embedding_dim]
                quantized_indices = topk_indices[:, 0] # [batch_size], use the closest codebook index
                commit_loss = torch.mean((quantized.detach() - encoded) ** 2)

            # straight through estimator
            quantized_st = encoded + (quantized - encoded).detach()

        if self.dihedral_loss:
            E_norm = F.normalize(encoded, p=2, dim=1)
            S = torch.matmul(E_norm, E_norm.T) # [B, B]
            logits = S / self.dihedral_temperature  # higher temp = softer
            mask = ~torch.eye(encoded.size(0), dtype=torch.bool, device=encoded.device)
            mask = mask & (Y != 0.5) # only consider non-ambiguous pairs
            dihedral_loss = F.binary_cross_entropy_with_logits(logits[mask], Y[mask].float())
        else:
            dihedral_loss = torch.tensor(0.0, device=encoded.device)
        
        reconstructed = self.decoder(quantized_st) # [batch_size, input_dim]

        if self.unit_circle_transform:
            reconstructed_unit_circle = reconstructed[:, :self.num_aa * 2 * 3]
            reconstructed_sequences = reconstructed[:, self.num_aa * 2 * 3:]  # [batch_size, num_aa * 20]
            reconstructed_angles = unit_circle_to_angles(reconstructed_unit_circle)

            unit_circle_regularization = torch.norm(reconstructed_unit_circle[:, ::2]**2 + reconstructed_unit_circle[:, 1::2]**2 - 1, p=2, dim=1)
            unit_circle_regularization = unit_circle_regularization.mean() * self.unit_circle_transform_weight

            logits_aa = reconstructed_sequences.reshape(batch_size, self.num_aa, 20).contiguous()  # [batch_size, num_aa, 20]
        else:
            reconstructed_angles = reconstructed[:, :self.num_aa * 3]
            reconstructed_sequences = reconstructed[:, self.num_aa * 3:]

            unit_circle_regularization = torch.tensor(0.0, device=reconstructed.device)

            logits_aa = reconstructed[:, self.num_aa * 3:].reshape(batch_size, self.num_aa, 20).contiguous()  # [batch_size, num_aa, 20]
        aa_loss = F.cross_entropy(logits_aa.view(-1, 20), gt_aa.view(-1))
        pred_aa = torch.softmax(logits_aa, dim=-1).detach().cpu()

        recon_loss = F.mse_loss(reconstructed_angles, batch['angles']) # [phi, psi, omega] values
        codebook_loss = F.mse_loss(encoded.detach(), quantized)

        if val:
            codebook_loss = torch.tensor(0.0, device=encoded.device)
            unit_circle_regularization = torch.tensor(0.0, device=encoded.device)

        loss = commit_loss + recon_loss + codebook_loss + unit_circle_regularization + aa_loss + dihedral_loss + aug_loss

        return VQVAEOutput(
            loss=loss,
            reconstructed_angles=reconstructed_angles,
            encoded=encoded,
            commit_loss=commit_loss,
            recon_loss=recon_loss,
            codebook_loss=codebook_loss,
            unit_circle_regularization=unit_circle_regularization,
            dihedral_loss=dihedral_loss,
            quantized=quantized,
            quantized_indices=quantized_indices,
            aa_loss=aa_loss,
            pred_aa=pred_aa,
        )