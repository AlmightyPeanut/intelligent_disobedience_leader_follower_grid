"""Evaluate proposer/validator pairings that do NOT involve the LLM.

Runs: perfect_x_perfect, perfect_x_always, and (if a PPO checkpoint exists)
ppo_x_perfect, ppo_x_always. No API calls, no rate limits.

    source .venv/bin/activate
    python run_eval_no_llm.py
    python run_eval_no_llm.py --max-configs 200
"""

from __future__ import annotations

import argparse

from eval_common import (
    add_config_sampling_args,
    add_proposer_args,
    always_approve_factory,
    learned_proposer_checkpoint,
    load_learned_proposer,
    perfect_proposer_factory,
    perfect_validator_factory,
    print_summary,
    resolve_variations,
    run_pairing,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_proposer_args(parser)
    add_config_sampling_args(parser)
    args = parser.parse_args()

    variations = resolve_variations(args, tag="run_eval_no_llm")

    pairings = [
        ("perfect_x_perfect", perfect_proposer_factory, perfect_validator_factory),
        ("perfect_x_always",  perfect_proposer_factory, always_approve_factory),
    ]
    learned_ckpt = learned_proposer_checkpoint(args.proposer, override=args.checkpoint)
    if learned_ckpt.exists():
        learned_factory = load_learned_proposer(args.proposer, override=args.checkpoint)
        pairings += [
            (f"{args.proposer}_x_perfect", learned_factory, perfect_validator_factory),
            (f"{args.proposer}_x_always",  learned_factory, always_approve_factory),
        ]
    else:
        print(f"[run_eval_no_llm] No {args.proposer.upper()} proposer checkpoint at {learned_ckpt}; "
              f"skipping {args.proposer}_* rows. Run train_{args.proposer}_proposer.py first to enable them.")

    results = []
    for name, p_factory, v_factory in pairings:
        print(f"\n>>> Running {name}")
        results.append(run_pairing(
            name, p_factory, v_factory, variations,
            video_dir="videos/no_llm",
        ))

    print_summary(results, len(variations))


if __name__ == "__main__":
    main()
