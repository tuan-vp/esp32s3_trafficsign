import os
import sys
import inspect

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.settings import COMMON, ONNX_EXPORT, ensure_parent_dir
from src.training.checkpoint import load_tinycnn_checkpoint


def main():
    common = COMMON
    cfg = ONNX_EXPORT
    device = torch.device("cpu")

    model, _ = load_tinycnn_checkpoint(
        checkpoint_path=cfg.checkpoint_path,
        num_classes=common.num_classes,
        device=device,
        enable_quant_stubs=False,
    )

    ensure_parent_dir(cfg.onnx_path)
    dummy_input = torch.randn(1, 3, 32, 32, device=device)

    export_kwargs = {
        "export_params": True,
        "opset_version": cfg.opset,
        "do_constant_folding": True,
        "input_names": ["input"],
        "output_names": ["output"],
    }
    # Prefer legacy exporter to avoid auto-upgrading to opset 18 and failed down-conversion.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(model, dummy_input, cfg.onnx_path, **export_kwargs)

    file_size_kb = os.path.getsize(cfg.onnx_path) / 1024
    print(f"Exported ONNX: {cfg.onnx_path}")
    print(f"ONNX size: {file_size_kb:.2f} KB")


if __name__ == "__main__":
    main()
