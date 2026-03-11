from typing import List, Tuple, Union
from PIL import Image
import numpy as np
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler
import os
import json
from collections import defaultdict
from Trainers import DSPFlowTrainer
from generation_models import CausalFlow

import argparse
import torch
import json
import os
import numpy as np
from torch.utils.data import Subset
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader



def text2ts_collate_fn(batch):

    B = len(batch)

    C = batch[0]["history"].shape[1]

    max_hist = max(x["history_len"] for x in batch)
    max_tgt = max(x["target_len"] for x in batch)

    history = torch.zeros(B, max_hist, C)
    target = torch.zeros(B, max_tgt, C)

    hist_mask = torch.zeros(B, max_hist)
    tgt_mask = torch.zeros(B, max_tgt)

    text_embed = torch.stack([x["text_embed"] for x in batch])

    ts_id = torch.tensor([x["ts_id"] for x in batch])
    block_id = torch.tensor([x["block_id"] for x in batch])

    image_id = torch.tensor([x["image_id"] for x in batch])

    for i, item in enumerate(batch):

        h_len = item["history_len"]
        t_len = item["target_len"]

        history[i, :h_len] = item["history"]
        target[i, :t_len] = item["target"]

        hist_mask[i, :h_len] = 1
        tgt_mask[i, :t_len] = 1

    # ------------------------------------------------
    # encoder self attention mask
    # (B,N_enc,N_enc)
    # ------------------------------------------------

    attn_mask = hist_mask.unsqueeze(1) * hist_mask.unsqueeze(2)

    # ------------------------------------------------
    # cross attention mask
    # (B,N_dec,N_enc)
    # decoder query -> encoder key
    # ------------------------------------------------

    cross_attn_mask = tgt_mask.unsqueeze(2) * hist_mask.unsqueeze(1)

    batch_dict = {

        "history": history.permute(0,2,1).contiguous(),   # (B,C,N_enc)
        "target": target.permute(0,2,1).contiguous(),     # (B,C,N_dec)

        "text_embed": text_embed,

        "attn_mask": attn_mask.bool(),                   # (B,N_enc,N_enc)
        "cross_attn_mask": cross_attn_mask.bool(),       # (B,N_dec,N_enc)

        "hist_mask": hist_mask,
        "tgt_mask": tgt_mask,

        "ts_id": ts_id,
        "block_id": block_id,
        "image_id": image_id,
    }

    return batch_dict




class Text2TSDataset(Dataset):

    def __init__(
        self,
        ts_path: str,
        text_embed_path: str,
        num_segments: int = 4,
    ):
        self.ts = np.load(ts_path, allow_pickle=True)   # (N, T, C)
        self.text_embed = torch.load(text_embed_path, map_location="cpu")

        self.num_segments = num_segments
        self.N, self.T, self.C = self.ts.shape

        assert self.T % num_segments == 0
        self.seg_len = self.T // num_segments

        self.ids = sorted(
            self.text_embed.keys(),
            key=lambda x: int(x.replace("image", ""))
        )

        self.num_block_choices = num_segments

    def __len__(self):
        return len(self.ids) * self.num_block_choices

    def __getitem__(self, idx):

        sample_idx = idx // self.num_block_choices
        block_id = idx % self.num_block_choices

        image_id = self.ids[sample_idx]
        ts_id = int(image_id.replace("image", ""))

        ts = torch.from_numpy(self.ts[ts_id]).float()   # (T,C)

        start = block_id * self.seg_len
        end = (block_id + 1) * self.seg_len

        history = ts[:end]           # encoder
        target = ts[start:end]       # decoder

        history_len = history.shape[0]
        target_len = target.shape[0]

        # text condition
        channel_embeds = []
        for c in range(self.C):
            key = f"seg{block_id+1}_channel{c}"
            channel_embeds.append(self.text_embed[image_id][key])

        text_embed = torch.stack(channel_embeds, dim=0)  # (C,D)

        return {
            "history": history,       # (N_enc,C)
            "target": target,         # (N_dec,C)
            "text_embed": text_embed,
            "history_len": history_len,
            "target_len": target_len,
            "ts_id": ts_id,
            "block_id": block_id,
            "image_id": image_id
        }

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
    parser.add_argument("--train_ts_path", type=str, required=True)
    parser.add_argument("--train_embed_path", type=str, required=True)
    parser.add_argument("--valid_ts_path", type=str, required=True)
    parser.add_argument("--valid_embed_path", type=str, required=True)

    """training parameters"""
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--max_epochs", type=int, required=True)
    parser.add_argument("--grad_clip_norm", type=float, required=True)
    parser.add_argument("--grad_accum_steps", type=int, required=True)

    """wandb parameters"""
    parser.add_argument("--wandb_project", type=str,required=True)
    parser.add_argument("--wandb_run", type=str, required=True)

    """gpu parameters"""
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)

    return parser.parse_args()



def conditional_trian(args):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    save_args_to_jsonl(args, f"{args.ckpt_dir}/config.jsonl")

    model = CausalFlow(
        seq_length=args.seq_len,
        feature_size=args.feature_size,
        n_layer_enc=args.n_layer_enc,
        n_layer_dec=args.n_layer_dec,
        d_model=args.d_model,
        n_heads=args.n_heads,
        mlp_hidden_times=4,
    )

    train_set = Text2TSDataset(
        args.train_ts_path,
        args.train_embed_path,
    )

    val_set = Text2TSDataset(
        args.train_ts_path,
        args.train_embed_path,
    )


    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, num_workers=16,
        shuffle=True, drop_last=True, collate_fn=text2ts_collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, num_workers=16,
        shuffle=False, drop_last=False, collate_fn=text2ts_collate_fn
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
        compile=False
    )

    trainer.cond_train(config=vars(args))



def main():
    args = get_args()

    conditional_trian(args)



if __name__ == "__main__":
    main()
