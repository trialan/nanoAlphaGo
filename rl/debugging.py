import torch


def check_no_nan_gradients(network):
    """Check the gradients for NaN or extreme values."""
    for name, param in network.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any()
            max_val = param.grad.max().item()
            min_val = param.grad.min().item()
            assert max_val < 1e10 and min_val > -1e10


def assert_no_nan_outputs(outputs):
    max_val = outputs.max().item()
    min_val = outputs.min().item()
    assert max_val < 1e10 and min_val > -1e10


