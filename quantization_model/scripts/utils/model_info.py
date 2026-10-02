import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.models.tinycnn import TinyCNN
from src.settings import COMMON


def print_model_info():
    model = TinyCNN(num_classes=COMMON.num_classes, enable_quant_stubs=False)

    print("=" * 40)
    print(" MODEL ARCHITECTURE")
    print("=" * 40)
    print(model)
    print()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 40)
    print(" PARAMETER INFO")
    print("=" * 40)
    print(f"Total Parameters:      {total_params:,}")
    print(f"Trainable Parameters:  {trainable_params:,}")
    print(f"Non-trainable Params:  {total_params - trainable_params:,}")
    print()

    param_size_mb = (total_params * 4) / (1024 ** 2)
    param_size_kb_int8 = total_params / 1024

    print("=" * 40)
    print(" ESTIMATED MODEL SIZE (WEIGHTS ONLY)")
    print("=" * 40)
    print(f"FP32 Size:  {param_size_mb:.4f} MB")
    print(f"INT8 Size:  {param_size_kb_int8:.2f} KB")
    print()

    dummy_input = torch.randn(1, 3, COMMON.image_size, COMMON.image_size)
    output = model(dummy_input)
    print("=" * 40)
    print(f" TENSOR SHAPES (Using 1x3x{COMMON.image_size}x{COMMON.image_size} input)")
    print("=" * 40)
    print(f"Input Shape:  {tuple(dummy_input.shape)}")
    print(f"Output Shape: {tuple(output.shape)}")


if __name__ == "__main__":
    print_model_info()
