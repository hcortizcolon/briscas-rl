# Story 2.1: Train Best Agent with DQN

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a researcher,
I want to train a point-maximizing agent against the engine's random player using DQN,
So that I can produce an agent that plays Briscas as well as possible given observable information.

## Acceptance Criteria

1. **Given** a working `BriscasEnv` and seed value, **When** `train_agent(agent_type="best", total_timesteps=N, seed=S, output_path=P)` is called, **Then** SB3's DQN is trained against the engine's built-in random player using SB3 DQN defaults except `learning_starts=1000` **And** the reward signal is the normalized point differential from `BriscasEnv` (multiplied by `reward_scale=1.0` for best agent) **And** `output_path` is a file path without extension (SB3 appends `.zip`, metadata uses `.json`) (AC: FR6)

2. **Given** training is in progress, **When** games complete, **Then** win rate over a rolling window of the last 1000 completed games is logged to stdout every 100 completed games via `WinRateCallback` **And** periodic checkpoint snapshots are saved to `models/checkpoints/` via SB3's `CheckpointCallback` (AC: FR8, FR9)

3. **Given** training completes, **When** the final model is saved, **Then** the model is saved as an SB3 `.zip` file to the specified output path (default: `models/best_agent_{timesteps}k.zip`) **And** a metadata JSON is saved alongside it containing: `agent_type`, `seed`, `total_timesteps`, `reward_type`, `timestamp` **And** a training summary is printed to stdout: final win rate (last 1000 games), total timesteps, total games played, model path (AC: FR9, FR11)

4. **Given** the CLI entry point `scripts/train.py`, **When** invoked with `--agent best --timesteps 200000 --seed 42`, **Then** seeds are propagated via `set_all_seeds()`, training runs, and model is saved to default output path (AC: FR6, FR16)

## Tasks / Subtasks

- [ ] Task 1: Add `reward_scale` parameter to `BriscasEnv` (AC: #1) — Do NOT use Gymnasium's `RewardWrapper`. Do NOT create a separate wrapper class. Modify `BriscasEnv` directly.
  - [ ] Add `reward_scale: float = 1.0` to `BriscasEnv.__init__()`
  - [ ] Multiply terminal reward by `self.reward_scale` in `step()`
  - [ ] Add `info["game_result"] = "win" | "loss" | "draw"` to the info dict at terminal state (before scaling), based on raw point differential: >0 win, <0 loss, ==0 draw. Callbacks read this instead of interpreting scaled rewards.
  - [ ] Add tests for `reward_scale=1.0` (default, unchanged behavior) and `reward_scale=-1.0` (negated)
  - [ ] Add test that `info["game_result"]` is present and correct at terminal state for win, loss, and draw scenarios
  - [ ] Verify all 85 existing tests still pass (no regressions)
- [ ] Task 2: Create `training/train.py` — `train_agent()` function (AC: #1, #2, #3)
  - [ ] Create `training/__init__.py` — export `train_agent` and `WinRateCallback` via `__all__`. Follow pattern from `gym_env/__init__.py`.
  - [ ] Implement `train_agent(agent_type, total_timesteps, seed, output_path, engine_url, checkpoint_freq)` function
  - [ ] Pre-flight connectivity check: call `env.reset()` before `model.learn()` — if `EngineConnectionError` raised, log clear message `"Cannot connect to game engine at {engine_url}. Is it running?"` and re-raise
  - [ ] Wrap training in try/finally: `env.close()` in finally block to clean up engine state on crash
  - [ ] Configure SB3 `DQN("MlpPolicy", env, verbose=1, learning_starts=1000)` — use SB3 defaults for all other hyperparameters. Do NOT create a hyperparameter config system. SB3 auto-wraps in `DummyVecEnv` — no explicit wrapping needed.
  - [ ] Set `reward_scale = -1.0 if agent_type == "worst" else 1.0` and pass to `BriscasEnv`
  - [ ] Configure `CheckpointCallback(save_freq=checkpoint_freq, save_path="models/checkpoints/", name_prefix="{agent_type}_agent")`
  - [ ] Implement `WinRateCallback(BaseCallback)` — on `dones[0] == True`, read `infos[0]["terminal_info"]["game_result"]` (NOT `infos[0]["game_result"]` — DummyVecEnv auto-reset moves terminal info), track rolling window of last 1000 completed games, log every 100 completed games, format: `"Win rate (last 1000 games): 67.3% | Games played: 4521"`. Also log on first game completion: `"First game completed at timestep {n} | Result: {result}"` to confirm callback is working. Expose `self.games_played` and `self.win_rate` as attributes so `train_agent()` can read final stats after `.learn()` completes.
  - [ ] Save model via `model.save(output_path)` — pass path **without** `.zip` extension (SB3 appends it)
  - [ ] Write metadata JSON to `output_path + ".json"`: `{agent_type, seed, total_timesteps, reward_type: "normalized_differential", timestamp}`
  - [ ] Print training summary to stdout: agent type, total timesteps, total games played (from `callback.games_played`), final win rate (from `callback.win_rate`), model path, metadata path. Never use the word "episodes" — use "timesteps" and "games".
  - [ ] `os.makedirs()` for `models/` and `models/checkpoints/` before saving
- [ ] Task 3: Create `scripts/train.py` — CLI entry point (AC: #4)
  - [ ] Add `sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))` at the top of the script (before project imports) so `python scripts/train.py` works from the project root. This is required because Python doesn't add the project root to `sys.path` when running scripts from a subdirectory.
  - [ ] argparse with flags: `--agent {best,worst}`, `--timesteps` (default 200000), `--seed` (default 42), `--output` (optional override, no extension), `--engine-url` (default `http://localhost:5000`), `--checkpoint-freq` (default 10000). Help text for `--timesteps`: "Total training timesteps. Minimum ~50k for any learning (learning_starts=1000)."
  - [ ] Call `set_all_seeds(seed)` before training
  - [ ] Construct default output path: `models/{agent_type}_agent_{timesteps // 1000}k` (no extension, e.g., `models/best_agent_200k`)
  - [ ] Call `train_agent()` — let `EngineConnectionError` propagate to top level
  - [ ] Configure `logging.basicConfig()` for stdout logging
- [ ] Task 4: Write tests (AC: #1, #2, #3)
  - [ ] **Unit tests (mocked SB3):**
    - [ ] Test `reward_scale` on `BriscasEnv` (default=1.0 unchanged, -1.0 negates reward)
    - [ ] Test `info["game_result"]` present at terminal state, correct for win/loss/draw
    - [ ] Test `train_agent()` orchestration — verify DQN instantiated with correct params, `.learn()` called, model saved, metadata written
    - [ ] Test metadata JSON contains all required fields and correct values
    - [ ] Test metadata JSON path = model path with `.zip` replaced by `.json`
    - [ ] Test default output path construction (no extension)
    - [ ] Test `WinRateCallback` — simulate `dones[0]=True` with `infos[0]["game_result"]="win"/"loss"/"draw"`, verify rolling count, log frequency (every 100 games), and exposed `games_played`/`win_rate` attributes
    - [ ] Test DQN configured with `learning_starts=1000` (not SB3 default of 50000)
    - [ ] Test pre-flight check raises clear error when engine unreachable
    - [ ] Test `env.close()` called even when training raises exception
  - [ ] **Integration test (real SB3, mocked adapter):** `@pytest.mark.integration`
    - [ ] Run `train_agent(agent_type="best", total_timesteps=100, ...)` with mocked `EngineAdapter` (not mocked SB3)
    - [ ] Verify SB3 + BriscasEnv + callbacks work together end-to-end
    - [ ] Verify `.zip` model file and `.json` metadata file are created on disk
    - [ ] ~100 timesteps, should complete in <5 seconds
- [ ] Task 5: Update `.gitignore` and create directories (AC: #2)
  - [ ] Add `models/checkpoints/` to `.gitignore`
  - [ ] Add `models/*.zip` and `models/*.json` to `.gitignore` (retrain from seeds)

## Dev Notes

### Technical Requirements

- **DQN Configuration:** `DQN("MlpPolicy", env, verbose=1, learning_starts=1000)`. Use SB3 defaults for ALL other hyperparameters (learning_rate, buffer_size, batch_size, exploration_fraction, etc.). Do NOT create a hyperparameter config system, config file, or config dataclass. Hardcoded call with one override.
- **CRITICAL — `learning_starts` pitfall:** SB3's DQN defaults to `learning_starts=50000`, meaning the agent does pure random exploration for the first 50k timesteps before any gradient updates. With `--timesteps 50000`, zero learning would occur. We override to `learning_starts=1000`.
- **Terminology:** Use "timesteps" (individual env.step() calls) and "games" (full Briscas games from reset to terminal). Never use "episodes" in code, logs, or output — it's ambiguous in the SB3 context.
- **SB3 DummyVecEnv auto-wrapping:** When you pass a raw Gymnasium env to `DQN()`, SB3 auto-wraps it in `DummyVecEnv`. This means `dones` becomes `np.array([True])` (array, not scalar) and `infos` becomes `[{...}]` (list, not dict). All callback code must index into these: `dones[0]`, `infos[0]`.
- **CRITICAL — DummyVecEnv auto-reset overwrites info:** When `dones[0] == True`, SB3's `DummyVecEnv` has ALREADY auto-reset the env by the time the callback fires. The terminal step's info is NOT in `infos[0]` directly — it is stored in `infos[0]["terminal_info"]`. Read game_result via `infos[0]["terminal_info"]["game_result"]`. Similarly, terminal observation is in `infos[0]["terminal_observation"]`. This is the #1 SB3 callback footgun.
- **Reward scaling inside the env:** Architecture mandates "never modify the reward signal outside the env wrapper." The `reward_scale` parameter on `BriscasEnv` satisfies this. Do NOT use Gymnasium's `RewardWrapper` or any external wrapper.
- **`info["game_result"]`:** Added to `BriscasEnv.step()` at terminal state BEFORE reward scaling. Values: `"win"` (raw differential > 0), `"loss"` (< 0), `"draw"` (== 0). This is the single source of truth for game outcomes — callbacks and evaluation code read this instead of interpreting reward values.
- **File path convention:** All output paths are specified WITHOUT extension. `model.save(path)` appends `.zip`. Metadata is written to `path + ".json"`. This keeps the pairing consistent and testable.
- **Pre-flight check:** Before `model.learn()`, call `env.reset()` once. If `EngineConnectionError` is raised, log a clear message and re-raise. This catches a missing game engine before SB3 wraps the error in internal exception handling.
- **Cleanup on crash:** `train_agent()` uses try/finally with `env.close()` in finally. This calls `adapter.delete_game()` to prevent orphan games on the engine.
- **Empty hand edge case:** If training crashes with `ZeroDivisionError` in `action % len(hand)`, the root cause is `len(hand) == 0` in `step()`. This is a pre-existing env edge case that may only surface during long training runs (hundreds of games). If encountered, add a guard in `BriscasEnv.step()`: if hand is empty, return `terminated=True` with current scores as reward. Track as a bugfix within Task 1.
- **Metadata timestamp:** Use UTC: `datetime.datetime.now(datetime.timezone.utc).isoformat()`. Do NOT use naive `datetime.now()` — timestamps must be unambiguous.

### Architecture Compliance

- **File locations:** `training/train.py` for library code, `scripts/train.py` for CLI entry point. [Source: architecture.md#Complete Project Directory Structure]
- **Boundary 3 — Scripts vs Library:** `scripts/train.py` handles argparse, logging setup, seed initialization, and default path construction. `training/train.py` contains `train_agent()` with no CLI dependencies — importable from tests and notebooks. [Source: architecture.md#Architectural Boundaries]
- **Adapter pattern preserved:** `BriscasEnv` receives adapter via constructor injection. `train_agent()` creates `RESTAdapter(engine_url)` and passes it. No code in `training/` directly calls the REST API. [Source: architecture.md#Environment Wrapper Contract]
- **Reward inside env:** `reward_scale` on `BriscasEnv` — reward is computed and scaled within `step()`. [Source: architecture.md#Enforcement Guidelines — "Never modify the reward signal outside the env wrapper"]
- **Single observation source:** Training uses `BriscasEnv._get_observation()` — no reimplementation of observation encoding in training code. [Source: architecture.md#Enforcement Guidelines]
- **Seed propagation:** `scripts/train.py` calls `set_all_seeds(seed)` before anything else. `seed.py` handles random, numpy, torch. [Source: architecture.md#Seed Propagation]
- **Error handling:** `logging` module, not print statements. `EngineConnectionError` propagates — training code does NOT catch it (only adapter catches raw HTTP errors). [Source: architecture.md#Error Handling]
- **Naming conventions:** snake_case files (`train.py`, `train_agent`), PascalCase classes (`WinRateCallback`), UPPER_SNAKE constants if needed. [Source: architecture.md#Code Naming Conventions]
- **Checkpointing:** SB3's `CheckpointCallback` per architecture spec. No custom checkpoint logic. [Source: architecture.md#Agent Training Pattern]
- **Model save format:** SB3 `.zip` + metadata `.json` alongside. [Source: architecture.md#Agent Training Pattern]
- **One agent architecture, parameterized:** `train_agent()` accepts `agent_type` and sets `reward_scale` accordingly. Single function, single DQN setup. Do NOT create separate "BestAgent" and "WorstAgent" classes or functions. [Source: architecture.md#Architectural Principles]

### Library / Framework Requirements

| Library | Version | Purpose | Notes |
|---|---|---|---|
| `stable-baselines3` | >=2.7.0,<3.0 | `DQN`, `CheckpointCallback`, `BaseCallback` | Already in requirements.txt |
| `torch` | >=2.3.0 | Neural network backend for DQN | Already in requirements.txt |
| `gymnasium` | >=1.2.0,<2.0 | Environment interface | Already in requirements.txt |
| `numpy` | (transitive) | Array operations | Already available |
| `json` | stdlib | Metadata file I/O | No install needed |
| `argparse` | stdlib | CLI argument parsing | No install needed |
| `logging` | stdlib | Structured logging | No install needed |
| `datetime` | stdlib | Timestamp for metadata | No install needed |
| `os` | stdlib | Directory creation, path manipulation | No install needed |
| `pytest` | latest stable | Testing | Already in requirements.txt |

**No new dependencies needed.** All imports come from SB3 (already in requirements.txt) or Python stdlib.

### SB3 API Reference (Latest Stable — v2.7.x)

```python
# DQN instantiation
from stable_baselines3 import DQN
model = DQN("MlpPolicy", env, verbose=1, learning_starts=1000)

# Training with callbacks
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
model.learn(total_timesteps=200000, callback=[checkpoint_cb, winrate_cb])

# Save (appends .zip automatically)
model.save("models/best_agent_200k")  # creates best_agent_200k.zip

# Load (for Story 2.3 — do NOT implement in this story)
model = DQN.load("models/best_agent_200k", env=env)

# CheckpointCallback
CheckpointCallback(save_freq=10000, save_path="models/checkpoints/", name_prefix="best_agent")

# Custom callback skeleton
class WinRateCallback(BaseCallback):
    def _on_step(self) -> bool:
        # self.locals["dones"] — np.array([bool]) for single env
        # self.locals["infos"] — [{...}] for single env
        # CRITICAL: When dones[0]==True, DummyVecEnv already auto-reset.
        # Terminal info is in infos[0]["terminal_info"], NOT infos[0] directly.
        if self.locals["dones"][0]:
            result = self.locals["infos"][0]["terminal_info"]["game_result"]
        return True  # return False to halt training
```

### File Structure Requirements

Files to create/modify in this story:
```
briscas_rl/
├── gym_env/
│   └── briscas_env.py              # MODIFIED — add reward_scale param + info["game_result"]
├── training/
│   ├── __init__.py                 # NEW — exports train_agent, WinRateCallback
│   └── train.py                    # NEW — train_agent() + WinRateCallback
├── scripts/
│   └── train.py                    # NEW — CLI entry point with argparse
├── models/                         # NEW directory (gitignored contents)
│   └── checkpoints/                # NEW directory (gitignored)
├── tests/
│   ├── test_briscas_env.py         # MODIFIED — add reward_scale + game_result tests
│   └── test_training.py            # NEW — unit tests + integration test
├── .gitignore                      # MODIFIED — add models patterns
└── seed.py                         # NO CHANGES
```

### Testing Standards

- Use `pytest` as testing framework [Source: architecture.md#Development Tooling]
- Test files mirror source: `tests/test_training.py` ↔ `training/train.py` [Source: architecture.md#Structure Patterns]
- **Two test tiers:**
  - Unit tests: mock SB3's DQN class, test orchestration logic, callback behavior, metadata writing, path construction
  - Integration test: `@pytest.mark.integration` — real SB3, mocked adapter, 100 timesteps, verifies end-to-end. Reuse adapter mock patterns from `tests/test_briscas_env.py`.
- All 85 existing tests MUST continue passing — do not modify existing test behavior
- Expected new tests: ~12-15 unit tests + 1 integration test
- Run full `pytest` suite as final verification

### Previous Story Intelligence (Story 1.3)

**What was built:**
- `seed.py` at project root — `set_all_seeds(seed: int) -> None`
- Seeds `random`, `numpy`, `torch`; CUDA determinism when GPU available
- Input validation: `TypeError` for non-int, `ValueError` for out-of-range
- 12 tests in `tests/test_seed.py`

**Patterns established across all 3 stories:**
- `logging.getLogger(__name__)` for logger setup
- `__init__.py` exports all public types via `__all__`
- Single-line `"""..."""` docstrings
- Tests use `unittest.mock` for mocking
- Inline type hints, no `.pyi` stubs
- Module-level functions (no unnecessary classes)

**Key insight from Story 1.3:** `set_all_seeds()` seeds global RNG state (numpy, torch, random). `BriscasEnv.reset(seed=seed)` seeds Gymnasium's internal `self.np_random`. These coexist — `set_all_seeds()` for global reproducibility in `scripts/train.py`, `env.reset(seed=)` for env-specific RNG. Do NOT call `set_all_seeds()` inside `train_agent()` — it's called in the CLI script before training starts.

**85 tests total** across 4 test files (23 adapter + 36 env + 14 observation + 12 seed). All must pass after this story.

### Git Intelligence

**Recent commits (5 total):**
1. `6cdb434` — Story 1.3: Enhanced seeding with input validation, 12 tests
2. `fc3cb08` — Story 1.3: Initial seeding implementation
3. `c07b736` — Story 1.3: seed.py creation
4. `d31f6d2` — Story 1.2: BriscasEnv, observation encoding, 50 tests
5. `98b8684` — Story 1.1: EngineAdapter, RESTAdapter, dataclasses, 23 tests

**Code conventions observed:**
- Inline type hints, no `.pyi` stubs
- Single-line docstrings
- `logging.getLogger(__name__)` pattern
- Module-level functions (no unnecessary classes)
- `__init__.py` with `__all__` exports
- Tests use `unittest.mock` and `pytest` fixtures

### Latest Tech Information

**SB3 v2.7.x DQN key parameters:**
- `learning_starts` (default 50000) — timesteps before first gradient update. **Override to 1000.**
- `buffer_size` (default 1000000) — replay buffer size. SB3 default is fine.
- `batch_size` (default 32) — minibatch size. SB3 default is fine.
- `exploration_fraction` (default 0.1) — fraction of total timesteps for epsilon decay. SB3 default is fine.
- `target_update_interval` (default 10000) — steps between target network updates. SB3 default is fine.

**SB3 save/load:**
- `model.save(path)` appends `.zip` if not present. Saves policy weights, optimizer state, hyperparameters.
- `DQN.load(path)` — class method, NOT instance method. Do not instantiate then load.
- Replay buffer is NOT saved by default (saves ~100MB+ disk space).

**CheckpointCallback (SB3 v2.7.x):**
- `save_freq` is in timesteps (not episodes/games)
- `save_replay_buffer=False` by default — leave as False (large files, restart from scratch on crash)
- Saves to `{save_path}/{name_prefix}_{num_timesteps}_steps.zip`

### Project Context Reference

- PRD: `_bmad-output/planning-artifacts/prd.md` — FR6 (train best agent), FR8 (monitor progress), FR9 (save model), FR11 (summary stats)
- Architecture: `_bmad-output/planning-artifacts/architecture.md` — Agent Training Pattern, Enforcement Guidelines, Project Structure
- Epics: `_bmad-output/planning-artifacts/epics.md` — Epic 2, Story 2.1 full acceptance criteria
- Previous stories: `_bmad-output/implementation-artifacts/1-*.md` — patterns, conventions, test counts

### References

- [Source: architecture.md#Agent Training Pattern]
- [Source: architecture.md#Complete Project Directory Structure]
- [Source: architecture.md#Architectural Boundaries — Boundary 3: Scripts ↔ Library Code]
- [Source: architecture.md#Environment Wrapper Contract]
- [Source: architecture.md#Enforcement Guidelines]
- [Source: architecture.md#Seed Propagation]
- [Source: architecture.md#Error Handling]
- [Source: architecture.md#Code Naming Conventions]
- [Source: architecture.md#Architectural Principles — "One agent architecture, parameterized"]
- [Source: epics.md#Story 2.1 — "largest story in the project"]
- [Source: prd.md#Agent Training — FR6, FR8, FR9, FR11]
- [Source: 1-3-reproducible-seeding.md — Previous story patterns and seed.py usage]
- [Source: SB3 docs — DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)
- [Source: SB3 docs — Callbacks](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html)
- [Source: SB3 docs — Save/Load](https://stable-baselines3.readthedocs.io/en/master/guide/save_format.html)

## Dev Agent Record

### Agent Model Used

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### Change Log

### File List
