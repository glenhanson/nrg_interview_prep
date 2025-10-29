import fire
from nrg_interview_prep.log_odds import log_odds_to_probability
from nrg_interview_prep.survival_model import train_model


def main():
    "Expose src code functions to CLI"
    fire.Fire({"logit_to_probs": log_odds_to_probability,
              "train_survival_model": train_model})
