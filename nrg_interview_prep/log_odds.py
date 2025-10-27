import torch
from typing import Optional


def log_odds_to_probability(log_odds: Optional[float] = 3.2) -> None:
    """
    Convert log-odds to probability using PyTorch sigmoid function.
    Args:
        log_odds (float): The log-odds value to convert. Default is 3.2.
    Returns:
        float: The corresponding probability value.
    """
    log_odds_tensor = torch.tensor(log_odds)
    # Check for numeric validity
    if torch.any(torch.isnan(log_odds_tensor)) or torch.any(torch.isinf(log_odds_tensor)):
        raise ValueError("Input contains NaN or Inf values.")
    probability_tensor = torch.sigmoid(log_odds_tensor)
    print(
        f"Converted log-odds {log_odds} to probability {probability_tensor.item()}")
