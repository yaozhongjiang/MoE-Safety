"""Tiny hardware positive/negative control, before downloading any model."""
import torch
import torch.nn.functional as F
from adapter import nvtx


def main():
    x = torch.ones((1, 512), device="cuda")
    anchor = torch.zeros(1, device="cuda")
    for _ in range(3):
        F.silu(x)
    torch.cuda.synchronize()
    for e in range(4):
        coord = f"R000000_P000_L000_E{e:03d}"
        with nvtx("moecho", True), nvtx("slot_" + coord, True):
            anchor.add_(1)
        with nvtx("moecho", True), nvtx("act_" + coord, True):
            if e in (1, 3):
                F.silu(x)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
