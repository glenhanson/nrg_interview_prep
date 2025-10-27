import torch


def log_odds_to_probability(log_odds: float) -> float:
    """
    Convert log-odds to probability using PyTorch sigmoid function.
    Args:
        log_odds (float): The log-odds value to convert.
    Returns:
        float: The corresponding probability value.
    """
    log_odds_tensor = torch.tensor(log_odds)
    probability_tensor = torch.sigmoid(log_odds_tensor)
    return probability_tensor.item()
