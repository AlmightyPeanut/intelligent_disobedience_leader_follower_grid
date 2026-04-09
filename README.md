# The Intelligent Disobedience Game: Leader-Follower Grid Environment

This project is part of an ongoing research collaboration with [Prof. Reuth Mirsky](https://facultyprofiles.tufts.edu/reuth-mirsky) at Tufts University on intelligent disobedience in AI systems.

**When should an AI agent disobey its leader to prevent harm?** This repository implements a grid-based environment for studying *intelligent disobedience* — situations where an assistant deliberately overrides an instruction because it has information suggesting the action is dangerous.

📄 **Paper:** *The Intelligent Disobedience Game: Formulating Disobedience in Stackelberg Games and Markov Decision Processes* — B. Hornig, R. Mirsky (RaD-AI workshop at AAMAS '26)

## Motivation

Intelligent disobedience arises when an assistant deliberately overrides an instruction to prevent harm — like a guide dog refusing to lead its handler into oncoming traffic. As autonomous systems are increasingly deployed in collaborative roles such as assistive robotics, decision support, and teleoperation, this tension becomes practical: a human operator may propose an action that advances their objective but inadvertently introduces risk, while the automated system may possess additional information about environmental hazards. Designing protocols that allow machines to selectively disobey harmful instructions — without undermining the human's objectives — is a central challenge in shared autonomy.

This project formalizes this interaction through the **Intelligent Disobedience Game (IDG)**, a sequential decision-making framework in which a leader suggests actions toward a task objective and a follower may obey or disobey to prevent harm. We characterize optimal strategies for both players, with particular attention to multi-step extensions where disobedience may have delayed consequences (safety traps).

## What This Repo Contains

- **Grid environment** (`env.py`): A configurable leader-follower grid world where the leader issues directional commands and the follower can choose to obey or disobey based on its own local observations.
- **RL training pipeline** (`rl_modules/`): Reinforcement learning modules for training follower agents with different disobedience policies.
- **Experiment runner** (`run_experiments.py`): Configurable experiment scripts for evaluating obedient vs. disobedient follower strategies across varying information asymmetry conditions.
- **Evaluation and metrics** (`eval.py`, `metrics.py`): Tools for measuring team performance, disobedience frequency, and outcome comparisons.

## Key Concepts

- **Intelligent Disobedience Game (IDG)**: A sequential game where a leader suggests actions toward a task objective and a follower decides whether to obey or disobey to prevent harm.
- **Information asymmetry**: The leader and follower have different observation spaces — the follower may detect environmental hazards invisible to the leader.
- **Safety traps**: Multi-step scenarios where obedience appears safe in the short term but leads to delayed harmful consequences.

## Getting Started

```bash
pip install -r requirements.txt
python run_experiments.py
```

Configuration options are in `config.py`.

## Citation

```
@inproceedings{hornig2026intelligent,
  title={The Intelligent Disobedience Game: Formulating Disobedience in Stackelberg Games and Markov Decision Processes},
  author={Hornig, Benedikt and Mirsky, Reuth},
  booktitle={RaD-AI Workshop at AAMAS '26},
  year={2026}
}
```
