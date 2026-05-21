# rover

An alternate proposer/validator stack for the Intelligent Disobedience Game grid environment. Adds:

- **Reward-shaped proposer training** — a per-step penalty plus BFS-distance shaping that lets PPO/SAC actually converge on 4×4 and 5×5 grids, where the original sparse reward stalls.
- **A reformatted perfect proposer** — `rover/rl_modules/perfect_proposer.py` now plans with BFS over a lava-aware grid and is fully deterministic: it tracks its own (pos, dir) from validator feedback, marks any cell that triggered a `forward` disobey as lava, and re-plans each step. Same role as the original, but predictable to reason about and easy to use as a ground-truth reference.
- **PPO and SAC learned proposers** — full RLlib training scripts with checkpoint management.
- **An LLM-based validator** — a "guide dog" that decides per-step whether to obey or block the proposer's action by querying an LLM. Two prompt variants (with and without strategic guidance) are included.
- **A side-by-side eval script** — runs every proposer × validator pairing on the same lava configurations and prints/saves a comparison table.

This directory is self-contained and does not modify the upstream env or modules. It can be used alongside upstream code or on its own.

## Setup

Python 3.13 recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### LLM Proxy credentials (only if using the LLM validator)

The LLM validator uses a generic LLM-proxy client (`llmproxy/`) that expects an OpenAI-compatible chat-completions endpoint. Create a `.env` file in this directory:

```
LLMPROXY_ENDPOINT=<proxy-endpoint-url>
LLMPROXY_API_KEY=<your-api-key>
```

The non-LLM pairings (`perfect`, `always-approve`) work without this.

## Training the learned proposers

The perfect proposer always works without training. Learned proposers (`ppo`, `sac`) are only evaluated if a checkpoint exists.

```bash
python train_ppo_proposer.py --grid-size 4 --iters 3000
python train_sac_proposer.py --grid-size 4 --iters 3000
```

Both scripts save to a named directory encoding grid size + iters under `checkpoints/`:

```
checkpoints/
├── ppo_proposer_perfect_g4_i3000/                # actual checkpoint
├── ppo_proposer_perfect -> ppo_proposer_perfect_g4_i3000   # canonical symlink
├── sac_proposer_perfect_g4_i3000/
└── sac_proposer_perfect -> sac_proposer_perfect_g4_i3000
```

The symlink `ppo_proposer_perfect` / `sac_proposer_perfect` always points at the most recent successful run. Older runs are preserved — repoint the symlink or use `--checkpoint` at eval time to pick one.

> **Note:** the absolute path of the working directory must contain no whitespace characters. Ray worker launch fails on paths with spaces.

## Running the evaluation

Evaluation is split into three scripts so LLM rate limits only hit the pairings that actually use the LLM.

### Pick the model first

Each script loads the learned proposer via the canonical symlink `checkpoints/{proposer}_proposer_perfect`. Check what it currently points at:

```bash
ls -l checkpoints/ppo_proposer_perfect
ls -l checkpoints/sac_proposer_perfect
```

If the target isn't the model you want (wrong grid size, wrong iter count, stale run):

1. **Repoint the symlink** (affects all future runs):
   ```bash
   cd checkpoints
   rm ppo_proposer_perfect    # removes the link only
   ln -s ppo_proposer_perfect_g4_i3000 ppo_proposer_perfect
   ```

2. **Override per-run with `--checkpoint`** (without touching the symlink):
   ```bash
   python run_eval_no_llm.py --proposer ppo --grid-size 4 --checkpoint ppo_proposer_perfect_g3_i3000
   ```
   The flag takes the directory name under `checkpoints/`, not a full path. `--grid-size` must match what the model was trained on.

### Run

```bash
# Non-LLM combos (perfect/learned proposer x perfect/always validator). No API calls.
python run_eval_no_llm.py --grid-size 4 --max-configs 100 --proposer ppo

# LLM validator without strategic guidance in the prompt.
python run_llm_eval_no_strat.py --grid-size 4 --max-configs 100 --proposer ppo

# LLM validator with strategic guidance in the prompt.
python run_llm_eval_w_strat.py --grid-size 4 --max-configs 100 --proposer ppo
```

Common flags:
- `--proposer {ppo,sac}` — which learned proposer to evaluate (default `ppo`)
- `--checkpoint <dir-name>` — override which checkpoint dir under `./checkpoints/` to load. Defaults to the canonical symlink for `--proposer`.
- `--grid-size N` — grid side length; `NUM_LAVA_TILES` is auto-set to `max(1, N²·0.25)`
- `--max-configs N` — randomly subsample N lava configurations (useful for staying under LLM rate limits)
- `--seed S` — RNG seed for subsampling (default 1)

On headless machines, prefix with `SDL_VIDEODRIVER=dummy` so pygame doesn't open a window:

```bash
SDL_VIDEODRIVER=dummy python run_eval_no_llm.py
```

### What each script runs

| Script                      | Pairings                                                                                  |
|-----------------------------|-------------------------------------------------------------------------------------------|
| `run_eval_no_llm.py`        | `perfect_x_perfect`, `perfect_x_always`, `{proposer}_x_perfect`*, `{proposer}_x_always`*  |
| `run_llm_eval_no_strat.py`  | `perfect_x_llm`, `{proposer}_x_llm`*                                                      |
| `run_llm_eval_w_strat.py`   | `perfect_x_llm`, `{proposer}_x_llm`*                                                      |

*`{proposer}_*` rows only run if the selected checkpoint exists.

### Output layout

```
logs/
├── llm_eval_no_strat.jsonl       # every LLM call from run_llm_eval_no_strat.py
└── llm_eval_w_strat.jsonl        # every LLM call from run_llm_eval_w_strat.py
videos/
├── no_llm/{combo}.mp4
├── no_strat/{combo}.mp4
└── w_strat/{combo}.mp4
eval_results/
└── {script}_g{grid}_n{configs}_{timestamp}.txt   # side-by-side table, one per eval run
```

Directories are auto-created if missing. Each run prints and saves a side-by-side comparison table (goal %, validator reward, wanted/good-disobey/bad-disobey rates).

## Layout

```
rover/
├── env.py                               # Reward-shaped GridWorldEnv
├── config.py                            # RLlib AlgorithmConfig for PPO/SAC
├── utils.py                             # Shared constants and dispatch tables
└── rl_modules/
    ├── perfect_proposer.py              # BFS planner with persistent lava memory
    ├── perfect_validator.py             # Rule-based safety oracle
    ├── always_approve_validator.py      # Lazy baseline
    └── catalog/                         # Dict-obs catalog override (SAC needs this)
llmproxy/                                # Generic OpenAI-compatible LLM client
llm_validator_no_strat.py                # LLM validator, no strategic guidance
llm_validator_w_strat.py                 # LLM validator, with strategic guidance
eval_common.py                           # Shared rollout / metrics / table helpers
run_eval_no_llm.py                       # Non-LLM pairings
run_llm_eval_no_strat.py                 # LLM pairings (no-strat prompt)
run_llm_eval_w_strat.py                  # LLM pairings (w-strat prompt)
train_ppo_proposer.py                    # Train the PPO proposer
train_sac_proposer.py                    # Train the SAC proposer
requirements.txt
```

## Notes

- This was developed as part of a Tufts CS150 (Multi-Agent Systems) project on intelligent disobedience with LLM validators.
- The LLM-validator prompts are starting points, better prompts likely produce better validator behavior. Both prompt files are short and easy to edit.
- Running with the LLM validator costs API calls per env step; for a 4×4 grid with 100 configs across 4 pairings that's ~12k calls. Plan your rate limits accordingly.
