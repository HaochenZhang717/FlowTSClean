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
    parser.add_argument("--test_data_path", type=str, required=True)

    """training parameters"""
    parser.add_argument("--batch_size", type=int, required=True)

    """gpu parameters"""
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    return parser.parse_args()



def uncond_sample(args):
    model = DSPFlow(
        seq_length=args.seq_len,
        feature_size=args.feature_size,
        n_layer_enc=args.n_layer_enc,
        n_layer_dec=args.n_layer_dec,
        d_model=args.d_model,
        n_heads=args.n_heads,
        mlp_hidden_times=4,
    )
    model.load_state_dict(torch.load(args.ckpt_path))
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model.to(device=device)
    model.eval()

    test_data = torch.from_numpy(np.load(args.test_data_path, allow_pickle=True))
    test_set = TensorDataset(test_data)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, num_workers=16,
        shuffle=False, drop_last=False,
    )

    results = {"reals":[], "samples":[]}
    for batch in tqdm(test_loader):
        signals = batch[0].to(device=device, dtype=torch.float32)

        with torch.no_grad():
            samples = model.uncond_generation(signals)

        for signal, sample in zip(signals, samples):
            results["reals"].append(signal.cpu())
            results["samples"].append(sample.cpu())

    results["reals"] = torch.cat(results["reals"])
    results["samples"] = torch.cat(results["samples"])
    torch.save(results, args.output_path)


if __name__ == "__main__":
    args = get_args()
    uncond_sample(args)