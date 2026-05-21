"""Evaluate proposer combos against the LLM validator (with strategic guidance).

For non-LLM combos, such as perfect/PPO proposer x perfect/always validator, use
run_eval_no_llm.py. that script does call the LLM and don't share rate limits.

    cd CS150-MAS-Rover
    source .venv/bin/activate
    python run_llm_eval_w_strat.py
    python run_llm_eval_w_strat.py --max-configs 100
"""

from __future__ import annotations

import argparse

from eval_common import (
    add_config_sampling_args,
    add_proposer_args,
    build_inference_module,
    learned_proposer_checkpoint,
    load_learned_proposer,
    perfect_proposer_factory,
    print_summary,
    resolve_variations,
    run_pairing,
)
from llm_validator_w_strat import LLMValidator


def main() -> None:
    parser = argparse.ArgumentParser()
    add_proposer_args(parser)
    add_config_sampling_args(parser)
    args = parser.parse_args()

    variations = resolve_variations(args, tag="run_llm_eval_w_strat")

    llm_validator = lambda env: build_inference_module(env, "validator", LLMValidator)

    pairings = [
        ("perfect_x_llm", perfect_proposer_factory, llm_validator),
    ]
    learned_ckpt = learned_proposer_checkpoint(args.proposer, override=args.checkpoint)
    if learned_ckpt.exists():
        pairings += [
            (f"{args.proposer}_x_llm", load_learned_proposer(args.proposer, override=args.checkpoint), llm_validator),
        ]
    else:
        print(f"[run_llm_eval_w_strat] No {args.proposer.upper()} proposer checkpoint at {learned_ckpt}; "
              f"skipping {args.proposer}_x_llm. Run train_{args.proposer}_proposer.py first to enable it.")

    results = []
    for name, p_factory, v_factory in pairings:
        print(f"\n>>> Running {name}")
        results.append(run_pairing(
            name, p_factory, v_factory, variations,
            video_dir="videos/w_strat",
        ))

    print_summary(results, len(variations))


if __name__ == "__main__":
    main()
