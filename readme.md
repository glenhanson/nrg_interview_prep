# README — nrg_interview_prep

## Loom Video Link
I recorded an extremely quick Loom video to talk through my code some [link here](https://www.loom.com/share/197956a96c46466282f3344cc46df590).


## Quick overview
This repository is set up to be developed either inside a devcontainer (recommended) or locally with a virtual environment. Devcontainers provide a reproducible dev environment (packages, OS, tools) that lives with the repo, while a local venv is a lightweight alternative.

## Why I like using a devcontainer
- Reproducibility: the container pins OS packages, Python version, and system libs so everyone gets the same runtime.
- Isolation: avoids contaminating the host environment.
- Source-control friendly: devcontainer configuration (Dockerfile, devcontainer.json) lives in the repo so onboarding is one command.
- CI parity: containers reduce “works on my machine” issues because CI can reuse the same image.
- Tooling: can preinstall linters, formatters, and debug tools inside the container.

To open the project in a devcontainer, use VS Code -> “Remote-Containers: Open Folder in Container”. 

## Local (no Docker) alternative
If you prefer not to use Docker/devcontainers, create a virtual environment and install the package and development dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip
pip3 install -e .[dev]
```

This installs the package in editable mode and pulls dev dependencies (tests, linting, etc.) defined in setup.cfg.

## Running the example scripts
Relevant source code is exposed in the CLI via the `nrg` command. 
Here are some examples.

```bash
nrg --help
nrg logit_to_probs --help
nrg logit_to_probs
nrg logit_to_probs -l 4.5

nrg train_survival_model --help
nrg train_survival_model
```

## Streamlit app for Question 2.
I created a lightweight Streamlit app for exploring the data and resulting model for question #2.
Launch it via shell command
```bash
sh launch_question2_app.sh
```

## Thank you!
I'm looking forward to interviewing next week!

