import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import json
from typing import Sequence, List, Tuple, Union, Optional
import ast

from ibex.loss.aligned_rmsd import CDR_RANGES_AHO, positions_to_backbone_dihedrals, region_mapping
from ibex.utils import region_mask_from_aho
from ibex import inference

proteinseq_toks = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
context_segments = ['FW1', 'CDR1', 'FW2', 'CDR2', 'FW3', 'CDR4', 'FW4', 'CDR3', 'FW5']

class Alphabet(object):
    def __init__(
        self,
        standard_toks: Sequence[str],
        special_toks: Sequence[str] = ("<unk>", "<cls>", "<mask>", "<pad>", "<eos>"),
    ):
        self.standard_toks = list(standard_toks)
        self.special_toks = list(special_toks)

        self.all_toks = self.special_toks + self.standard_toks
        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.get_idx("<pad>")
        self.cls_idx = self.get_idx("<cls>")
        self.mask_idx = self.get_idx("<mask>")
        self.eos_idx = self.get_idx("<eos>")
        self.all_special_tokens = ['<eos>', '<unk>', '<pad>', '<cls>', '<mask>']

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return self.tok_to_idx.copy()

    def encode(self, tok_list: List[str]) -> List[int]:
        return [self.tok_to_idx[tok] for tok in tok_list]


class MaskLoop:
    def __init__(self, mask_prob, alphabet):
        """
        Initialize the MaskLoop with a masking probability.

        Parameters:
        - mask_prob: Probability of masking a token.
        """
        self.mask_prob = mask_prob
        self.type_cdf = np.array([0.2, 0.4, 0.6, 0.8, 0.9, 1.0]) 
        self.alphabet = alphabet  # Token to use for masking sequences
        # Indexes represent the different ways to mask
        # 0: mask x% sequence only
        # 1: mask x% dihedral angles only
        # 2: mask x% sequence and dihedral angles
        # 3: Do not mask anything
        # 4: Mask sequence completely
        # 5: Mask dihedral angles completely
    
    def get_mask(self, tokens, mask_prob):
        return (torch.rand(tokens.shape) < mask_prob) & (tokens != self.alphabet.cls_idx) & (tokens != self.alphabet.eos_idx) & (tokens != self.alphabet.padding_idx)


    def __call__(self, item) -> List[int]:
        """
        Apply masking to the input sequence.

        Parameters:
        - sequence: A list of token indices.

        Returns:
        - A new list with some tokens masked.
        """
        type_choice = np.random.rand()
        if type_choice < self.type_cdf[0]:
            # mask x% sequence only
            tokens = item['sequence'].clone()
            mask = self.get_mask(tokens, self.mask_prob)
            tokens[mask] = self.alphabet.mask_idx
            item['sequence'] = tokens
        elif type_choice < self.type_cdf[1]:  
            # mask x% dihedral angles only
            mask = self.get_mask(item['sequence'], self.mask_prob) 
            item["angles_mask"] = mask
        elif type_choice < self.type_cdf[2]:
            # mask x% sequence and dihedral angles
            tokens = item['sequence'].clone()
            mask = self.get_mask(tokens, self.mask_prob)
            tokens[mask] = self.alphabet.mask_idx  
            item['sequence'] = tokens
            item["angles_mask"] = mask
        elif type_choice < self.type_cdf[3]:
            # Do not mask anything
            pass
        elif type_choice < self.type_cdf[4]:
            # Mask sequence completely
            tokens = item['sequence'].clone()
            mask = self.get_mask(tokens, 1.0)
            tokens[mask] = self.alphabet.mask_idx
            item['sequence'] = tokens
        else: 
            # Mask dihedral angles completely
            mask = self.get_mask(item['sequence'], 1.0)
            item["angles_mask"] = mask
        return item

class LoopSequenceDataset(Dataset):
    def __init__(self, data_path, context_path=None, context_dim=408, max_length=36):
        """
        Dataset for loop sequences, with dihedral angles and amino acid residues.

        Parameters:
        - data_path: A parquet file containing loop sequences.
        """
        super(LoopSequenceDataset, self).__init__()
        self.data_path = data_path
        self.data = []
        if self.data_path is not None:
            if self.data_path.endswith('.parquet'):
                df = pd.read_parquet(self.data_path)
                for col in ['loop_c_alpha_atoms', 'stem_c_alpha_atoms']: # Convert numpy arrays to lists
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: [y.tolist() for y in x] if isinstance(x[0], np.ndarray) else x)
                for col in ['phi', 'psi', 'omega']:
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
                self.data = df.to_dict(orient='records')
            elif self.data_path.endswith('.csv'):
                df = pd.read_csv(self.data_path)
                for col in ['phi', 'psi', 'omega', 'loop_c_alpha_atoms', 'stem_c_alpha_atoms']:
                    if col in df.columns:
                        df[col] = df[col].apply(ast.literal_eval)  # Convert string representations of lists to actual lists
                self.data = df.to_dict(orient='records')
            elif self.data_path.endswith('.jsonl'):
                with open(data_path, 'r') as f:
                    for line in f.readlines():
                        self.data.append(json.loads(line))
            else:
                raise ValueError("Unsupported file format. Please provide a .parquet, .csv, or .jsonl file.")
        self.context = pd.read_parquet(context_path).set_index('chain_id') if context_path else None # Load context data if provided
        self.context_dim = context_dim  # Dimension of the context vectors
        self.max_length = max_length  # max num of amino acids in a loop
        self.alphabet = Alphabet(standard_toks=proteinseq_toks)
        self.inference = False # Set to True if you do not want to apply masking
        self.apply_mask = MaskLoop(mask_prob=0.3, alphabet=self.alphabet)  # Initialize the masking function

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Add <cls> and <eos> tokens
        phi = [0.0] + self.data[idx]['phi'] + [0.0]
        psi = [0.0] + self.data[idx]['psi'] + [0.0]
        omega = [0.0] + self.data[idx]['omega'] + [0.0]
        sequence = self.data[idx]['loop_sequence']
        loop_id = self.data[idx]['loop_id']

        if 'loop_c_alpha_atoms' in self.data[idx] and 'stem_c_alpha_atoms' in self.data[idx]:
            loop_coords = [[0.0,0.0,0.0]] + self.data[idx]['loop_c_alpha_atoms'] + [[0.0,0.0,0.0]]
            stem_coords = self.data[idx]['stem_c_alpha_atoms']
        else:
            loop_coords = [[0.0,0.0,0.0]] * (len(sequence) + 2)
            stem_coords = [[0.0,0.0,0.0]] * 5

        assert len(sequence) <= self.max_length-2, f"Sequence length {len(sequence)} exceeds max length {self.max_length-2}" # -2 for <cls> and <eos>

        angles = torch.tensor([phi, psi, omega], dtype=torch.float32).transpose(0, 1) # Shape: Length x 3 (phi, psi, omega)
        tokens = torch.tensor([self.alphabet.cls_idx] + [self.alphabet.get_idx(aa) for aa in sequence] + [self.alphabet.eos_idx], dtype=torch.long)
        loop_coords = torch.tensor(loop_coords, dtype=torch.float32)  # Length x 3
        stem_coords = torch.tensor(stem_coords, dtype=torch.float32)  # Nstem x 3

        true_tokens = tokens.clone()  # Keep a copy of the original sequence for labels
    
        if len(tokens) < self.max_length:
            tokens = torch.cat([tokens, torch.full((self.max_length - len(tokens),), self.alphabet.padding_idx, dtype=torch.long)]) # max_length
            true_tokens = torch.cat([true_tokens, torch.full((self.max_length - len(true_tokens),), self.alphabet.padding_idx, dtype=torch.long)]) # max_length
            angles = torch.cat([angles, torch.zeros((self.max_length - angles.shape[0], 3), dtype=torch.float32)]) # max_length x 3
            loop_coords = torch.cat([loop_coords, torch.zeros((self.max_length - loop_coords.shape[0], 3), dtype=torch.float32)]) # max_length x 3
        angles_mask = torch.zeros((angles.shape[0]), dtype=torch.bool)  # max_length, Mask for angles, all False initially, True for masked values
        
        item = {
            "id": loop_id,
            "angles": angles, # Shape: max_length x 3 (phi, psi, omega)
            "angles_mask": angles_mask, # Mask for angles, all False initially
            "sequence": tokens, # Shape: max_length, padded with <pad> token
            "true_sequence": true_tokens, # Shape: max_length, original sequence for labels, padded with <pad> token
            "sequence_length": len(sequence), # Shape: 1, Length of the sequence without padding
            "loop_c_alpha_coords": loop_coords, # Shape: max_length x 3, padded with zeros
            "stem_c_alpha_coords": stem_coords, # Shape: Nstem x 3
        }
        
        if not self.context is None: 
            chain_id = loop_id[:-1]
            if chain_id in self.context.index:
                context_vec = np.stack([self.context.loc[chain_id][segment_name] for segment_name in context_segments])
                loop_type = 'CDR'+loop_id.split("_")[-1][-1]  # Extract loop type from loop_id
                context_to_mask = context_segments.index(loop_type)
                context_vec[context_to_mask, :] = 0
            else:
                context_vec = np.zeros((9, self.context_dim), dtype=np.float32)
            item['context'] = torch.tensor(context_vec, dtype=torch.float32)
            if not self.inference:
                # randomly mask context vectors
                type_choice = np.random.rand()
                if type_choice < 0.1: # no context
                    item['context'] = torch.zeros((9, self.context_dim), dtype=torch.float32)
                elif type_choice < 0.5: # mask context
                    context_mask = torch.rand(item['context'].shape[0]) < 0.3 # randomly mask 30% of the context vectors
                    item['context'][context_mask] = 0.0  # Set masked context vectors to zero
                else:
                    pass # keep context as is


        if not self.inference:
            item = self.apply_mask(item)
        return item

    @staticmethod
    def collate_fn(batch):
        ids = [item['id'] for item in batch]
        angles = torch.stack([item['angles'] for item in batch])
        angles_mask = torch.stack([item['angles_mask'] for item in batch])
        tokens = torch.stack([item['sequence'] for item in batch])
        true_tokens = torch.stack([item['true_sequence'] for item in batch])
        sequence_length = torch.tensor([item['sequence_length'] for item in batch], dtype=torch.long)
        loop_coords = torch.stack([item['loop_c_alpha_coords'] for item in batch])
        stem_coords = torch.stack([item['stem_c_alpha_coords'] for item in batch])
        if 'context' in batch[0].keys():
            context = torch.stack([item['context'] for item in batch])
        else:
            context = None
        batch = {
            'id': ids,
            'angles': angles,
            'angles_mask': angles_mask,
            'sequence': tokens,
            'true_sequence': true_tokens,
            'context': context,
            'sequence_length': sequence_length,
            'loop_c_alpha_coords': loop_coords,
            'stem_c_alpha_coords': stem_coords
        }
        return batch


class LoopSequenceOnlyDataset(LoopSequenceDataset):
    def __init__(self, data_path, context_path=None, context_dim=408, max_length=36):
        """
        Dataset for loop sequences, sequence only, without dihedral angles.

        Parameters:
        - data_path: A parquet file containing loop sequences.
        """
        super(LoopSequenceOnlyDataset, self).__init__(data_path=data_path, context_path=context_path, context_dim=context_dim, max_length=max_length)

    def __getitem__(self, idx):
        # Add <cls> and <eos> tokens
        sequence = self.data[idx]['loop_sequence']
        loop_id = self.data[idx]['loop_id']

        phi = [0.0] * (len(sequence) + 2)
        psi = [0.0] * (len(sequence) + 2)
        omega = [0.0] * (len(sequence) + 2)
        loop_coords = [[0.0,0.0,0.0]] * (len(sequence) + 2)
        stem_coords = [[0.0,0.0,0.0]] * 5
        
        assert len(sequence) <= self.max_length-2, f"Sequence length {len(sequence)} exceeds max length {self.max_length-2}" # -2 for <cls> and <eos>

        angles = torch.tensor([phi, psi, omega], dtype=torch.float32).transpose(0, 1) # Shape: Length x 3 (phi, psi, omega)
        tokens = torch.tensor([self.alphabet.cls_idx] + [self.alphabet.get_idx(aa) for aa in sequence] + [self.alphabet.eos_idx], dtype=torch.long)
        loop_coords = torch.tensor(loop_coords, dtype=torch.float32)  # Length x 3
        stem_coords = torch.tensor(stem_coords, dtype=torch.float32)  # Nstem x 3

        true_tokens = tokens.clone()  # Keep a copy of the original sequence for labels
    
        if len(tokens) < self.max_length:
            tokens = torch.cat([tokens, torch.full((self.max_length - len(tokens),), self.alphabet.padding_idx, dtype=torch.long)]) # max_length
            true_tokens = torch.cat([true_tokens, torch.full((self.max_length - len(true_tokens),), self.alphabet.padding_idx, dtype=torch.long)]) # max_length
            angles = torch.cat([angles, torch.zeros((self.max_length - angles.shape[0], 3), dtype=torch.float32)]) # max_length x 3
            loop_coords = torch.cat([loop_coords, torch.zeros((self.max_length - loop_coords.shape[0], 3), dtype=torch.float32)]) # max_length x 3
        angles_mask = torch.zeros((angles.shape[0]), dtype=torch.bool)  # max_length, Mask for angles, all False initially, True for masked values
        
        item = {
            "id": loop_id,
            "angles": angles, # Shape: max_length x 3 (phi, psi, omega)
            "angles_mask": angles_mask, # Mask for angles, all False initially
            "sequence": tokens, # Shape: max_length, padded with <pad> token
            "true_sequence": true_tokens, # Shape: max_length, original sequence for labels, padded with <pad> token
            "sequence_length": len(sequence), # Shape: 1, Length of the sequence without padding
            "loop_c_alpha_coords": loop_coords, # Shape: max_length x 3, padded with zeros
            "stem_c_alpha_coords": stem_coords, # Shape: Nstem x 3
        }

        if not self.context is None: 
            loop_type = 'CDR'+loop_id.split("_")[-1][-1]  # Extract loop type from loop_id
            chain_id = loop_id[:-1]
            if chain_id in self.context.index:
                context_vec = np.stack([self.context.loc[chain_id][segment_name] for segment_name in context_segments])
                context_to_mask = context_segments.index(loop_type)
                context_vec[context_to_mask, :] = 0
            else:
                context_vec = np.zeros((9, self.context_dim), dtype=np.float32)
            item['context'] = torch.tensor(context_vec, dtype=torch.float32)

        return item
    
    @staticmethod
    def collate_fn(batch):
        return LoopSequenceDataset.collate_fn(batch)


class AhoChainsLoopsDataset(torch.utils.data.Dataset):
    """
    Build per-loop items from AHo-aligned heavy/light chains provided in-memory.

    Each input element can be a dict with keys:
      - id: optional sample id (str)
      - heavy_aho: AHo-aligned sequence string for heavy (length ~149) or None
      - light_aho: AHo-aligned sequence string for light (length ~148) or None
      - heavy_angles: optional dict with keys 'phi','psi','omega' (lists length match heavy_aho)
      - light_angles: optional dict with keys 'phi','psi','omega' (lists length match light_aho)

    This dataset emits one item per loop (H1,H2,H3,L1,L2,L3), skipping empty loops.
    """

    def __init__(self, chain_items: List[dict], sequence_only: bool, max_length: int = 36, ibex_model: Optional[object] = None):
        super().__init__()
        self.sequence_only = sequence_only
        self.max_length = max_length
        self.alphabet = Alphabet(standard_toks=proteinseq_toks)
        self.ibex_model = ibex_model
        self.items: List[dict] = []

        for idx, entry in enumerate(chain_items):
            sample_id = entry.get("id", f"sample{idx}")
            heavy_aho: Optional[str] = entry.get("heavy_aho")
            light_aho: Optional[str] = entry.get("light_aho")

            # Precompute loop sequences from AHo using provided CDR ranges
            def _loop_seq_from_aho(aho_str: Optional[str], chain_label: str, loop_no: str) -> str:
                if not aho_str:
                    return ""
                start, end = CDR_RANGES_AHO[f"{chain_label}{loop_no}"]
                seg_slice = aho_str[start:end]
                return seg_slice.replace('-', '')

            # If ibex model is available and we want structural inputs, predict once per pair
            if not self.sequence_only:
                if self.ibex_model is None:
                    raise ValueError("Ibex model is not available. Please provide a model when initializing the dataset.")
                fv_heavy = heavy_aho.replace('-', '') if heavy_aho else ""
                fv_light = light_aho.replace('-', '') if light_aho else ""

                if not fv_heavy and not fv_light:
                    continue

                protein = inference(self.ibex_model, fv_heavy, fv_light, "", logging=False, return_pdb=False)

                # Convert outputs to torch tensors, supporting both numpy arrays and torch tensors
                pos_src = getattr(protein, 'atom_positions', None)
                if isinstance(pos_src, np.ndarray):
                    positions = torch.from_numpy(pos_src).float()  # [L, NAW, 3]
                elif torch.is_tensor(pos_src):
                    positions = pos_src.float()
                else:
                    positions = torch.tensor(pos_src, dtype=torch.float32)
                
                mask_src = getattr(protein, 'atom_mask', None)
                if isinstance(mask_src, np.ndarray):
                    mask = torch.from_numpy(mask_src.astype(bool))
                elif torch.is_tensor(mask_src):
                    mask = mask_src.bool()
                else:
                    mask = torch.tensor(mask_src, dtype=torch.bool)
                
                res_src = getattr(protein, 'residue_index', None)
                residue_index = None
                if res_src is not None:
                    if isinstance(res_src, np.ndarray):
                        residue_index = torch.from_numpy(res_src).long()
                    elif torch.is_tensor(res_src):
                        residue_index = res_src.long()
                    else:
                        residue_index = torch.tensor(res_src, dtype=torch.long)
                
                chain_src = getattr(protein, 'chain_index', None)
                chain_index = None
                if chain_src is not None:
                    if isinstance(chain_src, np.ndarray):
                        chain_index = torch.from_numpy(chain_src).long()
                    elif torch.is_tensor(chain_src):
                        chain_index = chain_src.long()
                    else:
                        chain_index = torch.tensor(chain_src, dtype=torch.long)

                dihedrals, dihedrals_mask = positions_to_backbone_dihedrals(positions, mask, residue_index=residue_index, chain_index=chain_index)
                # dihedrals: [L, 3]

                # Build region mask for mapping residues to loops
                region_mask = region_mask_from_aho(heavy_aho or "-"*149, light_aho or "")  # [L]

                # Emit items for the six CDR loops
                for chain_label, region_prefix in (("H", "cdrh"), ("L", "cdrl")):
                    aho_str = heavy_aho if chain_label == "H" else light_aho
                    if not aho_str:
                        continue
                    for loop_no in ("1", "2", "3"):
                        loop_region_name = f"{region_prefix}{loop_no}"
                        if loop_region_name not in region_mapping:
                            continue
                        region_idx = region_mapping[loop_region_name]

                        # Loop sequence from AHo (ungapped)
                        loop_seq = _loop_seq_from_aho(aho_str, chain_label, loop_no)
                        if len(loop_seq) == 0:
                            continue

                        loop_id = f"{sample_id}_{chain_label}{loop_no}"
                        sel = (region_mask == region_idx)
                        # Extract dihedral angles for selected residues, order preserved
                        sel_dihedrals = dihedrals[sel]
                        phi = sel_dihedrals[:, 0].tolist()
                        psi = sel_dihedrals[:, 1].tolist()
                        omega = sel_dihedrals[:, 2].tolist()

                        self.items.append({
                            "sample_id": sample_id,
                            "loop_id": loop_id,
                            "chain_label": chain_label,
                            "loop_label": f"{chain_label}{loop_no}",
                            "sequence": loop_seq,
                            "phi": [float(x) for x in phi],
                            "psi": [float(x) for x in psi],
                            "omega": [float(x) for x in omega],
                        })
            else:
                # Fallback path: use provided angles dicts or zeros, and slice by CDR ranges
                for chain_label, aho_str in (("H", heavy_aho), ("L", light_aho)):
                    if not aho_str:
                        continue
                    angles_key = "heavy_angles" if chain_label == "H" else "light_angles"
                    chain_angles = entry.get(angles_key)

                    for loop_no in ("1", "2", "3"):
                        start, end = CDR_RANGES_AHO[f"{chain_label}{loop_no}"]
                        seg_slice = aho_str[start:end]
                        keep = [i for i, c in enumerate(seg_slice) if c != "-"]
                        loop_seq = "".join(seg_slice[i] for i in keep)
                        if len(loop_seq) == 0:
                            continue
                        loop_id = f"{sample_id}_{chain_label}{loop_no}"

                        if self.sequence_only or chain_angles is None:
                            phi = [0.0] * len(loop_seq)
                            psi = [0.0] * len(loop_seq)
                            omega = [0.0] * len(loop_seq)
                        else:
                            full_phi = chain_angles.get("phi")
                            full_psi = chain_angles.get("psi")
                            full_omega = chain_angles.get("omega")
                            assert full_phi is not None and full_psi is not None and full_omega is not None, "Angles dict must include phi, psi, omega"
                            seg_phi = full_phi[start:end]
                            seg_psi = full_psi[start:end]
                            seg_omega = full_omega[start:end]
                            phi = [float(seg_phi[i]) for i in keep]
                            psi = [float(seg_psi[i]) for i in keep]
                            omega = [float(seg_omega[i]) for i in keep]

                        self.items.append({
                            "sample_id": sample_id,
                            "loop_id": loop_id,
                            "chain_label": chain_label,
                            "loop_label": f"{chain_label}{loop_no}",
                            "sequence": loop_seq,
                            "phi": phi,
                            "psi": psi,
                            "omega": omega,
                        })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        rec = self.items[i]
        seq = rec["sequence"]
        assert len(seq) <= self.max_length - 2, f"Sequence length {len(seq)} exceeds max length {self.max_length-2}"

        phi = [0.0] + rec["phi"] + [0.0]
        psi = [0.0] + rec["psi"] + [0.0]
        omega = [0.0] + rec["omega"] + [0.0]

        angles = torch.tensor([phi, psi, omega], dtype=torch.float32).transpose(0, 1)
        tokens = torch.tensor([self.alphabet.cls_idx] + [self.alphabet.get_idx(aa) for aa in seq] + [self.alphabet.eos_idx], dtype=torch.long)
        loop_coords = torch.zeros((len(tokens), 3), dtype=torch.float32)
        stem_coords = torch.zeros((5, 3), dtype=torch.float32)

        true_tokens = tokens.clone()
        if len(tokens) < self.max_length:
            pad = self.max_length - len(tokens)
            tokens = torch.cat([tokens, torch.full((pad,), self.alphabet.padding_idx, dtype=torch.long)])
            true_tokens = torch.cat([true_tokens, torch.full((pad,), self.alphabet.padding_idx, dtype=torch.long)])
            angles = torch.cat([angles, torch.zeros((pad, 3), dtype=torch.float32)])
            loop_coords = torch.cat([loop_coords, torch.zeros((pad, 3), dtype=torch.float32)])

        angles_mask = torch.zeros((angles.shape[0]), dtype=torch.bool)

        return {
            "id": rec["loop_id"],
            "angles": angles,
            "angles_mask": angles_mask,
            "sequence": tokens,
            "true_sequence": true_tokens,
            "sequence_length": len(seq),
            "loop_c_alpha_coords": loop_coords,
            "stem_c_alpha_coords": stem_coords,
            # No context
            "context": None,
        }

    @staticmethod
    def collate_fn(batch: List[dict]) -> dict:
        ids = [item['id'] for item in batch]
        angles = torch.stack([item['angles'] for item in batch])
        angles_mask = torch.stack([item['angles_mask'] for item in batch])
        tokens = torch.stack([item['sequence'] for item in batch])
        true_tokens = torch.stack([item['true_sequence'] for item in batch])
        sequence_length = torch.tensor([item['sequence_length'] for item in batch], dtype=torch.long)
        loop_coords = torch.stack([item['loop_c_alpha_coords'] for item in batch])
        stem_coords = torch.stack([item['stem_c_alpha_coords'] for item in batch])
        return {
            'id': ids,
            'angles': angles,
            'angles_mask': angles_mask,
            'sequence': tokens,
            'true_sequence': true_tokens,
            'context': None,
            'sequence_length': sequence_length,
            'loop_c_alpha_coords': loop_coords,
            'stem_c_alpha_coords': stem_coords,
        }
