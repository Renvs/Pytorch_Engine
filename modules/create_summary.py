import torch 

from torch.utils.tensorboard.writer import SummaryWriter

def create_summary_writer(
    name: str,
    model_name: str,
    epoch: int,
    extra: str = None,
) -> SummaryWriter:
    """
    Creates a TensorBoard SummaryWriter with a unique directory name.

    Args:
        name: The name of the experiment.
        model_name: The name of the model.
        extra: An optional extra string to append to the directory name.

    Returns:
        A SummaryWriter object.
    """
    from datetime import datetime
    import os

    timestamp = datetime.now().strftime("%Y-%m-%d")

    log_dir = os.path.join(
        "runs",
        timestamp,
        name,
        epoch,
        model_name,
    )

    if extra:
        log_dir = os.path.join(log_dir, extra)

    writer = SummaryWriter(log_dir=log_dir)

    return writer