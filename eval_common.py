"""Shared evaluation helpers for the IDG grid-world runners.

Used by:
  - run_eval_no_llm.py      (perfect/PPO proposer x perfect/always validator)
  - run_llm_eval_no_strat.py (same + LLM validator without strategic guidance)
  - run_llm_eval_w_strat.py  (same + LLM validator with strategic guidance)

Keeps rollout / metrics / table-printing / config-sampling code in one place so
the runner scripts only runs the combinations and call the shared helpers, and so we can easily add new combos
"""

from __future__ import annotations

import argparse
import itertools
import random
import sys
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import tqdm
import tree
from ray.rllib import SampleBatch
from ray.rllib.core.rl_module import RLModule, RLModuleSpec

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from env import GridWorldEnv  
from rl_modules.always_approve_validator import AlwaysApproveValidatorRLM  
from rl_modules.perfect_proposer import PerfectProposerRLM  
from rl_modules.perfect_validator import PerfectValidatorRLM  
from utils import GRID_SIZE, MAX_ENV_STEPS, NUM_LAVA_TILES, ProposerPolicies  

_LEARNED_PROPOSER_DIRS = {
    "ppo": "ppo_proposer_perfect",
    "sac": "sac_proposer_perfect",
}


def learned_proposer_checkpoint(kind: str, override: str | None = None) -> Path:
    if kind not in _LEARNED_PROPOSER_DIRS:
        raise ValueError(f"Unknown proposer kind: {kind!r}. Expected one of {list(_LEARNED_PROPOSER_DIRS)}.")
    dir_name = override if override else _LEARNED_PROPOSER_DIRS[kind]
    return (
        _REPO_ROOT / "checkpoints" / dir_name
        / "learner_group" / "learner" / "rl_module" / ProposerPolicies.LEARNED
    )


ModuleFactory = Callable[[GridWorldEnv], RLModule]


def sample_valid_env_variations(size: int, num_lava_tiles: int):
    start = (0, 0)
    goal = (size - 1, size - 1)
    blocked = {start, goal}
    inner = [(x, y) for x in range(size) for y in range(size) if (x, y) not in blocked]
    return [
        lava for lava in itertools.combinations(inner, num_lava_tiles)
        if _reachable(size, set(lava))
    ]


def _reachable(size: int, lava: set[tuple[int, int]]) -> bool:
    start = (0, 0)
    goal = (size - 1, size - 1)
    if start in lava or goal in lava:
        return False
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nr, nc = r + dr, c + dc
            if (0 <= nr < size and 0 <= nc < size
                    and (nr, nc) not in lava and (nr, nc) not in seen):
                seen.add((nr, nc))
                queue.append((nr, nc))
    return False


def build_inference_module(
    env: GridWorldEnv,
    agent: str,
    module_class: type[RLModule],
) -> RLModule:
    spec = RLModuleSpec(
        module_class=module_class,
        observation_space=env.observation_spaces[agent],
        action_space=env.action_spaces[agent],
        inference_only=True,
    )
    return spec.build()


def load_learned_proposer(kind: str, override: str | None = None) -> ModuleFactory:
    ckpt = learned_proposer_checkpoint(kind, override=override)

    def factory(_env: GridWorldEnv) -> RLModule:
        if not ckpt.exists():
            raise FileNotFoundError(
                f"{kind.upper()} proposer checkpoint not found at {ckpt}. "
                f"Run `python train_{kind}_proposer.py` first."
            )
        return RLModule.from_checkpoint(str(ckpt))

    return factory


def perfect_proposer_factory(env: GridWorldEnv) -> RLModule:
    return build_inference_module(env, "proposer", PerfectProposerRLM)


def perfect_validator_factory(env: GridWorldEnv) -> RLModule:
    return build_inference_module(env, "validator", PerfectValidatorRLM)


def always_approve_factory(env: GridWorldEnv) -> RLModule:
    return build_inference_module(env, "validator", AlwaysApproveValidatorRLM)


def _extract_action(module: RLModule, out: dict, stochastic: bool = False) -> int:
    if SampleBatch.ACTIONS in out:
        return int(out[SampleBatch.ACTIONS].item())
    logits = out[SampleBatch.ACTION_DIST_INPUTS]
    if stochastic:
        probs = torch.softmax(logits, dim=-1)
        return int(torch.distributions.Categorical(probs=probs).sample().item())
    return int(torch.argmax(logits, dim=-1).item())


def run_pairing(
    name: str,
    proposer_factory: ModuleFactory,
    validator_factory: ModuleFactory,
    variations: list,
    video_dir: Path | str = "videos",
    save_video: bool = True,
) -> dict:
    """Run a (proposer, validator) pairing across the given lava configurations."""
    env = GridWorldEnv(
        size=GRID_SIZE,
        render=True,
        record_render=True,
        # put a reasonable upper bound on steps to prevent infinite loops in case of agent running in circles
        max_steps=200,
        single_agent=False,
        num_lava_tiles=NUM_LAVA_TILES,
        proposer_sees_lava=False,
    )

    proposer_module = proposer_factory(env)
    validator_module = validator_factory(env)

    final_rewards: list[float] = []
    validator_rewards: list[float] = []

    for variation in tqdm.tqdm(variations, desc=name):
        if not variation:
            obs, _ = env.reset()
        else:
            obs, _ = env.reset(options={"lava_positions": list(variation)})
        obs = tree.map_structure(lambda x: torch.tensor(np.expand_dims(x, axis=0)), obs)

        terminated = {"__all__": False}
        truncated = {"__all__": False}
        rewards: dict = {"proposer": 0.0, "validator": 0.0}

        while not terminated["__all__"]:
            actions: dict = {}
            if "proposer" in obs:
                # Sample from the proposer's action distribution instead of taking argmax
                # Ties and near-ties break randomly
                # so the agent doesn't get locked into spinning in place on ambiguous states
                out = proposer_module.forward_exploration({SampleBatch.OBS: obs["proposer"]})
                actions["proposer"] = _extract_action(proposer_module, out, stochastic=True)
            elif "validator" in obs:
                out = validator_module.forward_inference({SampleBatch.OBS: obs["validator"]})
                actions["validator"] = _extract_action(validator_module, out)
            else:
                raise RuntimeError(f"No actionable agent in obs: {list(obs.keys())}")

            obs, rewards, terminated, truncated, _ = env.step(actions)
            if "validator" in rewards:
                validator_rewards.append(rewards["validator"])
            obs = tree.map_structure(lambda x: torch.tensor(np.expand_dims(x, axis=0)), obs)
            if truncated["__all__"]:
                break

        if not truncated["__all__"] and "proposer" in rewards:
            final_rewards.append(rewards["proposer"])
        else:
            final_rewards.append(-1.0)

    proposer_arr = np.array(final_rewards)
    val_arr = np.array(validator_rewards)

    result = {
        "name": name,
        "n_configs": len(proposer_arr),
        "goal_wins": int(np.sum(proposer_arr > 0.0)),
        "goal_pct": float(np.mean(proposer_arr > 0.0) * 100) if len(proposer_arr) else 0.0,
        "lava_deaths": int(np.sum(proposer_arr < 0.0)),
        "validator_mean_reward": float(np.mean(val_arr)) if len(val_arr) else 0.0,
        "wanted_pct": float(np.mean(val_arr >= 0.0) * 100) if len(val_arr) else 0.0,
        "good_disobey_pct": float(np.mean(val_arr > 0.0) * 100) if len(val_arr) else 0.0,
        "bad_disobey_pct": float(np.mean(val_arr < 0.0) * 100) if len(val_arr) else 0.0,
        "n_validator_decisions": int(len(val_arr)),
    }

    if hasattr(validator_module, "_call_count"):
        result["llm_calls"] = int(validator_module._call_count)
        result["llm_cache_hits"] = int(validator_module._cache_hits)

    if save_video:
        video_dir_path = Path(video_dir)
        video_dir_path.mkdir(parents=True, exist_ok=True)
        video_path = video_dir_path / f"{name}.mp4"
        try:
            env.save_video(str(video_path))
            result["video"] = str(video_path)
        except Exception as e:
            result["video_error"] = str(e)

    return result


def format_table(results: list[dict]) -> str:
    cols = [
        ("pairing",    lambda r: r["name"]),
        ("goal %",     lambda r: f"{r['goal_pct']:6.2f}"),
        ("goal/N",     lambda r: f"{r['goal_wins']}/{r['n_configs']}"),
        ("val_reward", lambda r: f"{r['validator_mean_reward']:+.4f}"),
        ("wanted %",   lambda r: f"{r['wanted_pct']:6.2f}"),
        ("good_dis %", lambda r: f"{r['good_disobey_pct']:6.2f}"),
        ("bad_dis %",  lambda r: f"{r['bad_disobey_pct']:6.2f}"),
        ("decisions",  lambda r: str(r["n_validator_decisions"])),
    ]
    headers = [h for h, _ in cols]
    rows = [[fn(r) for _, fn in cols] for r in results]
    widths = [max(len(headers[i]), max(len(row[i]) for row in rows)) for i in range(len(cols))]

    sep = "  ".join("-" * w for w in widths)
    lines = [sep,
             "  ".join(headers[i].ljust(widths[i]) for i in range(len(cols))),
             sep]
    for row in rows:
        lines.append("  ".join(row[i].ljust(widths[i]) for i in range(len(cols))))
    lines.append(sep)
    return "\n".join(lines)


def print_table(results: list[dict]) -> None:
    print(format_table(results))


def print_summary(results: list[dict], n_configs: int) -> None:
    from datetime import datetime

    header = ("\n" + "=" * 72 + "\n"
              f"Side-by-side ({GRID_SIZE}x{GRID_SIZE} grid, {NUM_LAVA_TILES} lava, "
              f"{n_configs} configs each)\n" + "=" * 72)
    body = format_table(results)
    print(header)
    print(body)

    script = Path(sys.argv[0]).stem if sys.argv and sys.argv[0] else "eval"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).resolve().parent / "eval_results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{script}_g{GRID_SIZE}_n{n_configs}_{ts}.txt"
    out_path.write_text(header + "\n" + body + "\n")
    print(f"\nSaved results to {out_path}")

    for r in results:
        if "llm_calls" in r:
            denom = max(1, r["llm_calls"] + r["llm_cache_hits"])
            print(f"\n{r['name']} LLM stats: {r['llm_calls']} unique calls, "
                  f"{r['llm_cache_hits']} cache hits "
                  f"(hit rate {r['llm_cache_hits'] / denom * 100:.1f}%)")
        if "video" in r:
            print(f"Saved {r['video']}")
        if "video_error" in r:
            print(f"Video save failed for {r['name']}: {r['video_error']}")


def add_proposer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--proposer", choices=sorted(_LEARNED_PROPOSER_DIRS.keys()),
                        default="ppo",
                        help="learned proposer to evaluate (default: ppo)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="override checkpoint dir under ./checkpoints/ "
                             "(e.g. ppo_proposer_perfect_g5_i2000). "
                             "Defaults to the canonical name for --proposer.")


def add_config_sampling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-configs", type=int, default=None,
                        help="if set, randomly subsample this many lava configurations")
    parser.add_argument("--seed", type=int, default=1,
                        help="RNG seed for subsampling, if --max-configs is set")
    parser.add_argument("--grid-size", type=int, default=3,
                        help="determines the size of the square grid")


def resolve_variations(args: argparse.Namespace, tag: str) -> list:
    set_grid(args)
    variations = sample_valid_env_variations(GRID_SIZE, NUM_LAVA_TILES)
    if args.max_configs is not None and args.max_configs < len(variations):
        rng = random.Random(args.seed)
        variations = rng.sample(variations, args.max_configs)
        print(f"[{tag}] Subsampled {len(variations)} configs (seed={args.seed})")
    return variations


def set_grid(args: argparse.Namespace) -> tuple[int, int]:
    """Propagate --grid-size to every module that caches GRID_SIZE/NUM_LAVA_TILES.

    Also returns (size, num_lava) so callers whose own module-level names were
    bound at import time (e.g. `from utils import GRID_SIZE`) can use the fresh
    values directly instead of reading their stale local binding.
    """
    global GRID_SIZE, NUM_LAVA_TILES
    GRID_SIZE = args.grid_size
    NUM_LAVA_TILES = max(1, int((GRID_SIZE ** 2) * 0.25))

    # Also mutate rover.utils so anything that reads utils.GRID_SIZE at call time sees the override.
    import utils as _utils
    _utils.GRID_SIZE = GRID_SIZE
    _utils.NUM_LAVA_TILES = NUM_LAVA_TILES

    print(f"[set_grid] GRID_SIZE={GRID_SIZE} NUM_LAVA_TILES={NUM_LAVA_TILES}")
    return GRID_SIZE, NUM_LAVA_TILES