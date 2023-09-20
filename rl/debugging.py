import torch

def has_nan_params(model):
    """ Check if the given neural network model has any NaN parameters. """
    for param in model.parameters():
        if torch.isnan(param).any():
            return True
    return False


def check_advantages(advantages, correct_len):
    assert len(advantages) == correct_len
    assert not torch.isnan(advantages).any(), "NaN values detected in advantages!"
    max_val = advantages.max().item()
    min_val = advantages.min().item()
    assert max_val < 1e10 and min_val > -1e10, f"Extreme values detected in advantages: Min: {min_val}, Max: {max_val}"


def check_nn_gradients(network):
    """Check the gradients of a PyTorch network for NaN or extreme values."""
    for name, param in network.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN values detected in the gradients of {name}!"
            max_val = param.grad.max().item()
            min_val = param.grad.min().item()
            assert max_val < 1e10 and min_val > -1e10, f"Extreme values detected in the gradients of {name}: Min: {min_val}, Max: {max_val}"

