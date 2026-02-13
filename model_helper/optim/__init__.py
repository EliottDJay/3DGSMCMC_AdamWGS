from diff_gaussian_rasterization import SparseGaussianAdam
import torch
from utils.logger import Logger as Log

def get_optimizer(args, params, **kwargs):
    """
    Factory function to get an optimizer by name.

    Args:
        name (str): Name of the optimizer ('adam').
        params (iterable): Parameters to optimize.
        **kwargs: Additional arguments for the optimizer.

    Returns:
        torch.optim.Optimizer: An instance of the specified optimizer.
    """
    opt_dict = args.get('optimizer', {})
    name = opt_dict.get('name', 'adam').lower()
    Log.info(f"Using optimizer: {name}")
    #print(opt_dict)
    beta1 = opt_dict.get('beta1', 0.9)
    beta2 = opt_dict.get('beta2', 0.999)
    kwargs['betas'] = (beta1, beta2)

    if name.lower() == 'adam':
        return torch.optim.Adam(params, **kwargs)
    elif name.lower() == 'sparseadam':
        return SparseGaussianAdam(params, lr=0.0, eps=1e-15, opt_dict=opt_dict)
    else:
        raise ValueError(f"Unknown optimizer: {name}")