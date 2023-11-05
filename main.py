"""
Test pytorch linalg module with pylint
"""

import torch


def my_qr_decomposition(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    My QR
    """
    tensor_q, tensor_r = torch.linalg.qr(tensor)
    return tensor_q, tensor_r
