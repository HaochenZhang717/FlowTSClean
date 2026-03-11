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
from generation_models import DSPFlow

import argparse
import torch
import json
import os
import numpy as np
from torch.utils.data import Subset
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader



def text2ts_collate_fn(batch):

    ts = torch.stack([item["ts"] for item in batch], dim=0)                # (B, T, C)
    text_embed = torch.stack([item["text_embed"] for item in batch], dim=0).mean(1) # (B, C, D)

    batch_dict = {
        "ts": ts,
        "text_embed": text_embed,
    }

    return batch_dict


class Text2TSDataset(Dataset):
    """
    Block-causal diffusion dataset.

    原始:
        self.ts: (N, T, C)

    每个 __getitem__ 返回一个 (image, target_block) 的训练样本:
        - ts:         (C, T)
        - text_embed: (C, D)   # 只给 target block 的所有 channel text embedding
        - loss_mask:  (T,)     # 只有 target block 对应区间为 1
        - attn_mask: (T,)     # 只有 target block 对应区间为 1
        - block_id:   int
        - image_id:   str
    """

    def __init__(
        self,
        ts_path: str,
        text_embed_path: str,
    ):
        self.ts = np.load(ts_path, allow_pickle=True)          # (N, T, C)
        self.text_embed = torch.load(text_embed_path, map_location="cpu")


        self.N, self.T, self.C = self.ts.shape

        # image ids from embedding dict, e.g. ["image0", "image1", ...]
        self.ids = sorted(
            self.text_embed.keys(),
            key=lambda x: int(x.replace("image", "")))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        image_id = self.ids[idx]
        ts_id = int(image_id.replace("image", ""))

        # ts: (T, C)
        ts = self.ts[ts_id]

        assert ts.shape[0] == self.T and ts.shape[1] == self.C

        # 转成 (C, T)，更适合大多数 diffusion model
        ts = torch.from_numpy(ts).float()  # (C, T)

        # -----------------------------
        # text condition:
        # 只取 target block 的所有 channel embedding
        # shape -> (C, D)
        # -----------------------------
        channel_embeds = []
        for target_block in range(4):
            for c in range(self.C):
                key = f"seg{target_block + 1}_channel{c}"   # target_block 是 0-based, caption key 是 1-based
                emb = self.text_embed[image_id][key]
                channel_embeds.append(emb)

        text_embed = torch.stack(channel_embeds, dim=0)   # (C, D)

        # -----------------------------
        # 构造 masks
        # -----------------------------

        sample = {
            "ts": ts,                     # (C, T)
            "text_embed": text_embed,     # (C, D)
            "image_id": image_id,
            "ts_id": ts_id,
        }

        return sample


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

    model = DSPFlow(
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
    )

    trainer.cond_train(config=vars(args))



def main():
    args = get_args()

    conditional_trian(args)



if __name__ == "__main__":
    main()
