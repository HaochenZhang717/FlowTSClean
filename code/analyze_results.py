import torch
import matplotlib.pyplot as plt
import numpy as np
from metrics.discriminative_torch import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics



def calculate_scores_from_real_language(pt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(pt_path, map_location=device)
    real = []
    fake = []
    for datum in data:
        real.append(datum["reals"])
        fake.append(datum["samples"])
    real = torch.stack(real)
    fake = torch.stack(fake)

    real = real.permute(0, 2, 1)
    fake = fake.permute(0, 2, 1)
    print("Real shape:", real.shape)
    print("Fake shape:", fake.shape)
    discriminative_score = discriminative_score_metrics(
        real, fake,
        real.shape[-1],
        device,
    )
    print(f"Discriminative Score Metrics: {discriminative_score}")

    predictive_score = predictive_score_metrics(
        real, fake, device
    )
    print(f"Predictive Score Metrics: {predictive_score}")

    return (
        discriminative_score,
        predictive_score,
    )


if __name__ == "__main__":
    calculate_scores_from_real_language(
        pt_path="/playpen/haochenz/FlowTS/unconditional/stock_0311/sample_results.pth"
    )