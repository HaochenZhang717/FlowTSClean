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

    histories = [b["history"] for b in batch]
    targets   = [b["target"]  for b in batch]

    text_embed = torch.stack([b["text_embed"] for b in batch])

    # ------------------------------------------------
    # pad history
    # ------------------------------------------------

    max_hist = max(h.shape[0] for h in histories)

    hist_pad = []
    hist_mask = []

    for h in histories:

        n = h.shape[0]

        pad = torch.zeros(max_hist-n, h.shape[1])

        hist_pad.append(torch.cat([h,pad],0))

        m = torch.zeros(max_hist)
        m[:n] = 1
        hist_mask.append(m)

    history = torch.stack(hist_pad)       # (B,N_enc,C)
    hist_mask = torch.stack(hist_mask)    # (B,N_enc)

    # ------------------------------------------------
    # pad target
    # ------------------------------------------------

    max_tgt = max(t.shape[0] for t in targets)

    tgt_pad = []
    tgt_mask = []

    for t in targets:

        n = t.shape[0]

        pad = torch.zeros(max_tgt-n, t.shape[1])

        tgt_pad.append(torch.cat([t,pad],0))

        m = torch.zeros(max_tgt)
        m[:n] = 1
        tgt_mask.append(m)

    target = torch.stack(tgt_pad)         # (B,N_dec,C)
    tgt_mask = torch.stack(tgt_mask)      # (B,N_dec)

    # ------------------------------------------------
    # build attention masks
    # ------------------------------------------------

    encoder_self_mask = hist_mask.unsqueeze(1) * hist_mask.unsqueeze(2)

    decoder_self_mask = tgt_mask.unsqueeze(1) * tgt_mask.unsqueeze(2)

    cross_attn_mask = tgt_mask.unsqueeze(2) * hist_mask.unsqueeze(1)

    return {

        "history": history,    # (B,C,N_enc)

        "target": target,      # (B,C,N_dec)

        "text_embed": text_embed,

        "encoder_self_mask": encoder_self_mask.bool(),

        "decoder_self_mask": decoder_self_mask.bool(),

        "cross_attn_mask": cross_attn_mask.bool(),

        "hist_mask": hist_mask,

        "tgt_mask": tgt_mask,
    }



class Text2TSDataset(Dataset):

    def __init__(
        self,
        ts_path,
        text_embed_path,
        num_segments=4,
    ):
        self.ts = np.load(ts_path)              # (N,T,C)
        self.text_embed = torch.load(text_embed_path, map_location="cpu")

        self.num_segments = num_segments

        self.N, self.T, self.C = self.ts.shape
        assert self.T % num_segments == 0

        self.segment_length = self.T // num_segments

        self.ids = sorted(
            self.text_embed.keys(),
            key=lambda x: int(x.replace("image",""))
        )

        self.block_ids = list(range(num_segments))

    def __len__(self):
        return len(self.ids) * self.num_segments

    def __getitem__(self, idx):

        sample_idx = idx // self.num_segments
        block_id = idx % self.num_segments

        image_id = self.ids[sample_idx]
        ts_id = int(image_id.replace("image",""))

        ts = torch.from_numpy(self.ts[ts_id]).float()   # (T,C)

        start = block_id * self.segment_length
        end   = (block_id+1) * self.segment_length

        # ------------------------------------------------
        # history tokens
        # ------------------------------------------------

        history = ts[:start]                 # (N_hist,C)

        dummy = torch.zeros(1, self.C)

        history = torch.cat([dummy, history], dim=0)  # (N_hist+1,C)

        history_len = history.shape[0]

        # ------------------------------------------------
        # target tokens
        # ------------------------------------------------

        target = ts[start:end]               # (seg_len,C)

        # ------------------------------------------------
        # text embedding
        # ------------------------------------------------

        channel_embeds = []

        for c in range(self.C):

            key = f"seg{block_id+1}_channel{c}"

            emb = self.text_embed[image_id][key]

            channel_embeds.append(emb)

        text_embed = torch.stack(channel_embeds, dim=0)

        return {
            "history": history,          # (N_hist+1,C)
            "target": target,            # (seg_len,C)
            "text_embed": text_embed,
            "block_id": block_id,
            # "image_id": image_id,
            "ts_id": ts_id,
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
        train_set, batch_size=args.batch_size, num_workers=0,
        shuffle=True, drop_last=True, collate_fn=text2ts_collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, num_workers=0,
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

    trainer.cond_causal_train(config=vars(args))



def main():
    args = get_args()

    conditional_trian(args)



if __name__ == "__main__":
    main()
