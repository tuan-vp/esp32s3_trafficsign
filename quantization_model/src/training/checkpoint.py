from typing import Tuple

import torch

from src.models.tinycnn import TinyCNN


def load_tinycnn_checkpoint(
    checkpoint_path: str,
    num_classes: int = 10,
    device: torch.device = torch.device("cpu"),
    enable_quant_stubs: bool = False,
) -> Tuple[TinyCNN, dict]:
    payload = torch.load(checkpoint_path, map_location=device)

    model = TinyCNN(num_classes=num_classes, enable_quant_stubs=enable_quant_stubs)
    state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    metadata = payload if isinstance(payload, dict) else {}
    return model, metadata
