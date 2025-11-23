
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from torch.nn import Module
from torch.optim import AdamW
from dataset import LoopSequenceDataset
from igloo.vqvae import VQVAE
from trainer import VQVAETrainer, get_save_dir
from utils import seed_everything
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for VQVAE with loop dataset")
    # Dataset parameters
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the parquet file containing train loop data")
    parser.add_argument("--val_data_path", type=str, required=True, help="Path to the parquet file containing val loop data")
    parser.add_argument("--context_path", type=str, default=None, help="Path to the parquet file containing precomputed PLM embeddings")
    parser.add_argument("--loop_length", type=int, default=36, help="Max length of the loop")

    # Training parameters
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training (e.g., 'cpu' or 'cuda')")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--codebook_learning_rate", type=float, default=None, help="Learning rate for codebook optimizer")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay for optimizer")
    parser.add_argument("--commit_loss_weight", type=float, default=0.1, help="Weight for commitment loss")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--orthog_regularization", action="store_true", help="Use orthogonal regularization for the codebook")
    parser.add_argument("--dihedral_temperature", type=float, default=0.1, help="Temperature for dihedral loss")
    parser.add_argument("--dihedral_loss", action="store_true", help="Use dihedral loss in the model")
    parser.add_argument("--loop_length_loss", action="store_true", help="Use loop length loss in the model")
    parser.add_argument("--loop_length_pred_loss", action="store_true", help="Use loop length prediction loss in the model")
    parser.add_argument("--learnable_codebook", action="store_true", help="Use learnable codebook in the model")
    parser.add_argument("--final_learning_rate", type=float, default=None, help="Final learning rate for the model")
    parser.add_argument("--num_warmup_epochs", type=int, default=0, help="Number of warmup epochs before learning rate scheduling")

    # Model parameters
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension for the model")
    parser.add_argument("--codebook_size", type=int, default=128, help="Codebook size for the model")
    parser.add_argument("--num_encoder_layers", type=int, default=3, help="Number of encoder layers")
    parser.add_argument("--unit_circle_transform_weight", type=float, default=0.01, help="Weight for unit circle transform loss")
    parser.add_argument("--loop_length_tolerance", type=int, default=0, help="Tolerance for difference in loop length in dihedral loss")
    parser.add_argument("--frozen_lm_head", action="store_true", help="Freeze the language model head during training")

    # Logging parameters
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save model checkpoints")
    parser.add_argument("--project_name", type=str, default="VQVAE_Loop_Training", help="Project name for wandb")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")

    # Pretrained checkpoints
    parser.add_argument("--pretrained_model_weights", type=str, default=None, help="Path to pretrained model weights")
    parser.add_argument("--pretrained_model_config", type=str, default=None, help="Path to pretrained model config")

    return parser.parse_args()

def main(args):
    seed_everything(args.seed)
    train_dataset = LoopSequenceDataset(args.train_data_path, max_length=args.loop_length, context_path=args.context_path)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    val_dataset = LoopSequenceDataset(args.val_data_path, max_length=args.loop_length, context_path=args.context_path)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)
    if args.pretrained_model_weights and args.pretrained_model_config:
        model = VQVAE.load_from_config_and_weights(args.pretrained_model_config, args.pretrained_model_weights)
    else:
        model = VQVAE(
            embedding_dim=args.embedding_dim,
            codebook_size=args.codebook_size,
            orthog_regularization=args.orthog_regularization,
            dihedral_temperature=args.dihedral_temperature,
            num_encoder_layers=args.num_encoder_layers,
            dihedral_loss=args.dihedral_loss,
            unit_circle_transform_weight=args.unit_circle_transform_weight,
            commit_loss_weight=args.commit_loss_weight,
            loop_length_tolerance=args.loop_length_tolerance,
            loop_length_loss=args.loop_length_loss,
            loop_length_pred_loss=args.loop_length_pred_loss,
            learnable_codebook=args.learnable_codebook,
        )
    model = model.to(args.device)

    no_decay = ["bias", "LayerNorm.weight"]
    if args.codebook_learning_rate is None:
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    else:
        params_no_weight_decay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
        general_params = [p for n, p in model.named_parameters() if (not ('quantizer._codebook' in n)) and (not any(nd in n for nd in no_decay))]
        codebook_params = [p for n, p in model.named_parameters() if ('quantizer._codebook' in n) and (not any(nd in n for nd in no_decay))]
        assert len(codebook_params) > 0, "Codebook parameters not found in the model"
        assert len(general_params) > 0, "General parameters not found in the model"
        print(f"General parameters: {len(general_params)}, Codebook parameters: {len(codebook_params)}, No weight decay parameters: {len(params_no_weight_decay)}")
        optimizer = AdamW(
            [
                {'params': general_params, 'lr': args.learning_rate, 'weight_decay': args.weight_decay},
                {'params': codebook_params, 'lr': args.codebook_learning_rate, 'weight_decay': args.weight_decay},
                {'params': params_no_weight_decay, 'lr': args.learning_rate, 'weight_decay': 0.0}
            ]
        )
    
    if args.final_learning_rate is not None:
        scheduler = CosineAnnealingLR(optimizer, T_max=(args.num_epochs - args.num_warmup_epochs), eta_min=args.final_learning_rate)
    else:
        scheduler = None

    if args.frozen_lm_head:
        trainable_params = len([p for name, p in model.named_parameters() if p.requires_grad])
        model.encoder.lm_head.requires_grad_(False)
        new_trainable_params = len([p for name, p in model.named_parameters() if p.requires_grad])
        assert new_trainable_params < trainable_params, "LM head should be frozen"
    
    print("Number of model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    if args.use_wandb:
        wandb.init(project="Igloo", config=args, dir="./wandb/", name=args.project_name)    

    trainer = VQVAETrainer(model, optimizer, train_dataloader, val_loader=val_loader, device=args.device, epochs=args.num_epochs,
                           use_wandb=args.use_wandb, save_dir=get_save_dir(args.save_dir), scheduler=scheduler, warmup_epochs=args.num_warmup_epochs)
    
    training_config = vars(args)
    with open(f"{trainer.save_dir}/training_config.json", "w") as f:
        json.dump(training_config, f, indent=4)

    trainer.train()

if __name__ == "__main__":
    args = parse_args()
    main(args)