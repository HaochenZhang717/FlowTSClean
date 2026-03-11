from Trainers import DSPFlowTrainer
from generation_models import DSPFlow

import argparse
import torch
import json
import os
import numpy as np
from torch.utils.data import Subset
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader




def save_args_to_jsonl(args, output_path):
    args_dict = vars(args)
    with open(output_path, "w") as f:
        json.dump(args_dict, f)
        f.write("\n")  # JSONL 一行一个 JSON


def get_args():
    parser = argparse.ArgumentParser(description="parameters for flow-ts pretraining")

    """what to do"""

    """time series general parameters"""
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--feature_size", type=int, required=True)

    """model parameters"""
    parser.add_argument("--n_layer_enc", type=int, required=True)
    parser.add_argument("--n_layer_dec", type=int, required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--n_heads", type=int, required=True)

    """data parameters"""
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--valid_data_path", type=str, required=True)

    """training parameters"""
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--max_epochs", type=int, required=True)
    parser.add_argument("--grad_clip_norm", type=float, required=True)
    parser.add_argument("--grad_accum_steps", type=int, required=True)
    parser.add_argument("--early_stop", type=str, required=True)
    parser.add_argument("--patience", type=int, required=True)

    """wandb parameters"""
    parser.add_argument("--wandb_project", type=str,required=True)
    parser.add_argument("--wandb_run", type=str, required=True)

    """gpu parameters"""
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)

    return parser.parse_args()



def unconditional_trian(args):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    save_args_to_jsonl(args, f"{args.ckpt_dir}/config.jsonl")

    model = DSPFlow(
        seq_length=args.seq_len,
        feature_size=args.feature_size,
        n_layer_enc=args.n_layer_enc,
        n_layer_dec=args.n_layer_dec,
        d_model=args.d_model,
        n_heads=args.n_heads,
        mlp_hidden_times=4,
    )

    train_data = torch.from_numpy(np.load(args.train_data_path, allow_pickle=True))
    train_set = TensorDataset(train_data)

    valid_data = torch.from_numpy(np.load(args.valid_data_path, allow_pickle=True))
    val_set = TensorDataset(valid_data)


    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, num_workers=16,
        shuffle=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, num_workers=16,
        shuffle=False, drop_last=False,
    )

    optimizer= torch.optim.Adam(model.parameters(), lr=args.lr)


    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    trainer = DSPFlowTrainer(
        optimizer=optimizer,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=args.max_epochs,
        device=device,
        save_dir=args.ckpt_dir,
        wandb_run_name=args.wandb_run,
        wandb_project_name=args.wandb_project,
        grad_clip_norm=args.grad_clip_norm,
        grad_accum_steps=args.grad_accum_steps,
    )

    trainer.uncond_train(config=vars(args))



def main():
    args = get_args()

    unconditional_trian(args)



if __name__ == "__main__":
    main()
