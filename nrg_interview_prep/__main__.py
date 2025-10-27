import fire
from nrg_interview_prep.log_odds import log_odds_to_probability


def main():
    "Expose src code functions to CLI"
    fire.Fire({"logit_to_probs": log_odds_to_probability})
