from pathlib import Path

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def profile_forward(model: nn.Module, X: torch.Tensor, device: torch.device, trace: bool = False) -> str:
    model = model.to(device).eval()
    X = X.to(device)

    with torch.no_grad():
        for _ in range(3):
            model(X)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True, profile_memory=True, with_stack=True) as prof:
        with record_function("forward"):
            with torch.no_grad():
                model(X)
            if device.type == "mps":
                torch.mps.synchronize()

    sort_by = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    table = prof.key_averages().table(sort_by=sort_by, row_limit=10)
    print(table)

    if trace:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(RESULTS_DIR / "trace.json"))
    return table


if __name__ == "__main__":
    from ..models.cnn import CNN

    torch.manual_seed(0)
    profile_forward(CNN(), torch.randn(128, 3, 32, 32), torch.device("cpu"))
