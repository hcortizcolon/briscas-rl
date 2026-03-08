# Story 3.1: Run Evaluation Matchups

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a researcher,
I want to run N games between any two agents with alternating first player and record results to CSV,
So that I can collect the raw data needed to compare agent performance.

## Acceptance Criteria

1. **Given** a trained model path and `"random"` for the engine's built-in random player, **When** `run_evaluation(agent1, agent2, num_games=N, seed=S)` is called, **Then** N games are played between the trained model and the engine's random player **And** exactly one agent must be `"random"` (AC: FR12). **Note:** The `first_player` column is recorded as `0` for all games (the trained model, playing as "human", always goes first per engine API). If the engine API supports a first-player parameter, use it to alternate per `game_id % 2`; otherwise document the limitation (AC: FR14 — best-effort).

2. **Given** `"random"` is specified as an agent, **When** evaluation runs, **Then** the game engine's built-in random player is used — no separate random policy implemented (AC: FR12)

3. **Given** evaluation uses the same environment as training, **When** games are played, **Then** `BriscasEnv` is instantiated with the same observation encoding and action masking used during training (AC: FR12)

4. **Given** evaluation completes, **When** results are written, **Then** a CSV is created with columns: `game_id, agent1_points, agent2_points, first_player, point_differential` **And** the default filename follows the convention: `{agent1}_vs_{agent2}_{num_games}g_{seed}s.csv` **And** the file is saved to `results/` (AC: FR15)

5. **Given** the CLI entry point `scripts/evaluate.py`, **When** invoked with `--agent1 models/best.zip --agent2 random --games 10000 --seed 42`, **Then** seeds are propagated via `set_all_seeds()`, evaluation runs, and CSV is written **And** an optional `--output` flag overrides the default filename (AC: FR12, FR13, FR16)

## Tasks / Subtasks

- [x] Task 0: Add point scores to BriscasEnv info dict (AC: #4 — enables clean point extraction)
  - [x] In `gym_env/briscas_env.py`, modify `step()`: when `terminated=True`, add `info["agent_points"]` and `info["opponent_points"]` from `self._state.players`
  - [x] Add unit tests in `tests/test_briscas_env.py`: verify `info` contains `agent_points` and `opponent_points` when game ends, and does NOT contain them on intermediate steps
  - [x] This is a small, backward-compatible change — existing tests that check `info["game_result"]` are unaffected

- [x] Task 1: Create `evaluation/evaluate.py` with `run_evaluation()` function (AC: #1, #2, #3, #4)
  - [x] Create `evaluation/__init__.py` with `run_evaluation` export
  - [x] Implement `run_evaluation(agent1: str, agent2: str, num_games: int, seed: int, output_dir: str = "results") -> str`
  - [x] Parse agent arguments: if agent string is `"random"`, the engine's built-in random player is used (no model loaded); otherwise, load model via `load_agent(path)` from `training.train`
  - [x] For each game (0 to N-1): play the game to completion, record agent1_points, agent2_points, first_player, point_differential
  - [x] For `"random"` agent: do NOT call `model.predict()` — let the engine's built-in AI opponent play via `process_ai_turn()`. The "random" agent is always the non-human player in BriscasEnv.
  - [x] **Critical constraint:** Exactly one agent must be `"random"` (the engine's AI). If both are `"random"`, raise `ValueError("At least one agent must be a trained model")`. If neither is `"random"`, raise `ValueError("Exactly one agent must be 'random' — model-vs-model is not supported (engine API does not expose opponent hand)")`.
  - [x] **First-player column:** Check if the engine's `/api/game/new` endpoint accepts a first-player parameter. If yes, alternate via `game_id % 2`. If no, record `first_player = 0` for all games and log a warning: `"Engine API does not support first-player alternation — model always plays first"`.
  - [x] Write CSV with columns: `game_id,agent1_points,agent2_points,first_player,point_differential`
  - [x] Generate default filename: `{agent1_name}_vs_{agent2_name}_{num_games}g_{seed}s.csv` where agent names are derived from model path basenames (e.g., `best_agent_50k`) or `"random"`
  - [x] Create `results/` directory if it doesn't exist
  - [x] Return the path to the written CSV file
  - [x] Log summary: total games, agent names, output path

- [x] Task 2: Create `scripts/evaluate.py` CLI entry point (AC: #5)
  - [x] Implement argparse with: `--agent1`, `--agent2`, `--games` (default 10000), `--seed` (default 42), `--output` (optional override), `--engine-url` (default http://localhost:5000)
  - [x] Call `set_all_seeds(seed)` before evaluation
  - [x] Call `run_evaluation()` with parsed arguments
  - [x] Follow same CLI patterns as `scripts/train.py` (logging setup, sys.path insertion)

- [x] Task 3: Write tests (AC: #1, #2, #3, #4, #5)
  - [x] **Unit tests (mocked adapter, mocked DQN):**
    - [x] Test `run_evaluation()` plays exactly N games
    - [x] Test first_player column is recorded (0 if engine doesn't support alternation, alternating if it does)
    - [x] Test CSV output has correct columns and row count
    - [x] Test CSV filename follows naming convention
    - [x] Test `"random"` agent uses engine's built-in player (no model.predict called for random)
    - [x] Test output directory is created if missing
    - [x] Test both agents `"random"` raises ValueError
    - [x] Test neither agent is `"random"` raises ValueError (model-vs-model not supported)
    - [x] Test point_differential = agent1_points - agent2_points
    - [x] Test agent name extraction from model path (e.g., `models/best_agent_50k` → `best_agent_50k`)
  - [x] **Integration test (real SB3, mocked adapter):** `@pytest.mark.integration`
    - [x] Train a model (100 timesteps), then run 10 evaluation games against "random"
    - [x] Verify CSV is written with correct structure
    - [x] Verify all 10 games produce valid point values
    - [x] ~100 timesteps training + 10 eval games, should complete in <15 seconds

## Dev Notes

### Technical Requirements

- **Evaluation pattern from architecture:** `run_evaluation()` accepts two agent paths (or `"random"`), number of games, and seed. The function is library code in `evaluation/evaluate.py`, NOT in `scripts/`. [Source: architecture.md#Evaluation Pattern]
- **`"random"` means the engine's built-in AI.** The game engine always has a built-in random player (the non-human player). When an agent is `"random"`, we simply let the engine play — the agent is the opponent side of `BriscasEnv`. When an agent is a trained model, we load it and use `model.predict()` for action selection.
- **Exactly one agent must be `"random"`.** The engine API does not expose the opponent's hand, so we cannot build an observation vector for a second model. Model-vs-model (e.g., best-vs-worst) is not possible with the current engine. The skill-vs-luck comparison is done indirectly: best-vs-random and worst-vs-random stats are compared side-by-side. This covers the research goal from the PRD.
- **First-player alternation (ADR):** The engine always makes the "human" player go first. At implementation time, check if `/api/game/new` accepts a first-player parameter. If yes, alternate per `game_id % 2`. If no, record `first_player = 0` for all games and log a warning documenting the limitation. Do NOT fabricate alternating values — honest data is critical for analysis.
- **Point extraction via info dict (ADR):** Task 0 adds `agent_points` and `opponent_points` to the `info` dict returned by `BriscasEnv.step()` when `terminated=True`. This provides a clean public API for evaluation to read point scores without accessing private `env._state`. Small, backward-compatible change to `BriscasEnv`.
- **CSV writing:** Use Python's `csv` module (stdlib). No pandas dependency needed.
- **Agent name extraction:** From path like `models/best_agent_50k` or `models/best_agent_50k.zip`, extract `best_agent_50k`. Use `os.path.basename()` and strip `.zip` if present.
- **`load_agent()` reuse:** Import from `training.train` to load DQN models. Already handles path normalization, file existence, metadata.
- **Action selection:** Use `model.predict(obs, deterministic=True)` for evaluation — no exploration noise.
- **Reward scale for evaluation:** Use `reward_scale=1.0` (default) — evaluation always uses the natural reward direction regardless of agent type.
- **Game loop:** For each game, `env.reset()` then loop `model.predict()` + `env.step()` until terminated. Extract points from `info["agent_points"]` and `info["opponent_points"]`. Map to `agent1_points`/`agent2_points` based on which agent is the trained model vs random.

### Architecture Compliance

- **File location:** New `evaluation/evaluate.py` — matches architecture exactly. [Source: architecture.md#Complete Project Directory Structure]
- **Boundary 2 preserved:** Evaluation interacts ONLY through `BriscasEnv`'s Gymnasium interface. [Source: architecture.md#Architectural Boundaries]
- **Boundary 3 preserved:** `run_evaluation()` is library code in `evaluation/`, CLI is in `scripts/evaluate.py`. [Source: architecture.md#Architectural Boundaries]
- **CSV naming convention:** `{agent1}_vs_{agent2}_{num_games}g_{seed}s.csv`. [Source: architecture.md#Evaluation Pattern]
- **Evaluation MUST use same BriscasEnv:** Architecture explicitly requires this. [Source: architecture.md#Evaluation Pattern]
- **Error handling:** Use Python `logging` module. Exceptions propagate to top level. [Source: architecture.md#Error Handling]
- **Seed propagation:** Via `set_all_seeds()` in CLI entry point. [Source: architecture.md#Seed Propagation]

### Library / Framework Requirements

No new dependencies. All from stdlib and existing packages:

| Library | Version | Purpose | Notes |
|---|---|---|---|
| `csv` | stdlib | Write evaluation results | No install needed |
| `os` | stdlib | Path handling, directory creation | No install needed |
| `logging` | stdlib | Progress and completion logging | No install needed |
| `stable-baselines3` | >=2.7.0,<3.0 | `DQN.load()` via `load_agent()` | Already in requirements.txt |
| `gymnasium` | >=0.29.0 | `BriscasEnv` | Already in requirements.txt |

### File Structure Requirements

Files to create/modify in this story:
```
briscas_rl/
├── gym_env/
│   └── briscas_env.py              # MODIFIED — add agent_points/opponent_points to info dict
├── evaluation/
│   ├── __init__.py                 # NEW — export run_evaluation
│   └── evaluate.py                 # NEW — run_evaluation() function
├── scripts/
│   └── evaluate.py                 # NEW — CLI entry point
├── tests/
│   ├── test_briscas_env.py         # MODIFIED — add tests for point info in terminated games
│   └── test_evaluation.py          # NEW — unit + integration tests
```

### Testing Standards

- Use `pytest` as testing framework [Source: architecture.md#Development Tooling]
- Create new `tests/test_evaluation.py` — new module warrants new test file
- **Two test tiers:**
  - Unit tests: mock adapter and DQN, test game loop, CSV output, naming, error cases
  - Integration test: `@pytest.mark.integration` — real SB3 train + eval round-trip
- All 134 existing tests MUST continue passing
- Expected new tests: ~10 unit tests + 1 integration test
- Use `tmp_path` pytest fixture for CSV output in tests (avoid polluting `results/`)

### Previous Story Intelligence (Story 2.3)

**What was built:**
- `training/train.py` — `load_agent()` function for loading saved models
- `load_agent()` returns `(model, metadata)` tuple
- 134 tests passing (92 env + 31 training + 11 load_agent)

**Patterns confirmed:**
- `logging.getLogger(__name__)` for logger setup
- `__init__.py` with `__all__` exports
- Single-line docstrings
- `unittest.mock` for mocking in tests
- `@pytest.mark.integration` for integration tests
- try/finally with `env.close()` for cleanup
- `model.predict(obs, deterministic=True)` for inference
- `int(action.item())` to convert numpy action to Python int

**Code patterns from `validate_worst_agent()` (directly reusable):**
```python
env = BriscasEnv(adapter=adapter, reward_scale=1.0)
try:
    for _ in range(num_games):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(int(action.item()))
            done = terminated or truncated
        # Process results
finally:
    env.close()
```

### Git Intelligence

**Recent commits (last 5):**
1. `e5fe3a3` — Story 2.3 code review: malformed JSON handling in load_agent
2. `c00f8dc` — Story 2.3: load_agent() implementation
3. `909435e` — Story 2.2 code review: validation game count guard
4. `82c9218` — Story 2.2: validate_worst_agent()
5. `8e50c52` — Story 2.1 code review: Fixed WinRateCallback

**Code conventions confirmed:**
- `logging.getLogger(__name__)` for logger setup
- `__init__.py` with `__all__` exports
- Single-line docstrings
- `unittest.mock` for mocking in tests
- `@pytest.mark.integration` for integration tests
- Metadata JSON at `{path}.json` alongside model `.zip`
- CLI pattern: argparse, logging.basicConfig, sys.path insertion, set_all_seeds

### Project Context Reference

- PRD: `_bmad-output/planning-artifacts/prd.md` — FR12, FR13, FR14, FR15
- Architecture: `_bmad-output/planning-artifacts/architecture.md` — Evaluation Pattern, CSV naming, directory structure
- Epics: `_bmad-output/planning-artifacts/epics.md` — Epic 3, Story 3.1 acceptance criteria
- Previous story: `_bmad-output/implementation-artifacts/2-3-load-and-resume-trained-models.md` — load_agent(), 134 tests

### References

- [Source: architecture.md#Evaluation Pattern — run_evaluation function signature and behavior]
- [Source: architecture.md#Evaluation Pattern — CSV naming convention: {agent1}_vs_{agent2}_{num_games}g_{seed}s.csv]
- [Source: architecture.md#Evaluation Pattern — "random" uses engine's built-in random player]
- [Source: architecture.md#Evaluation Pattern — Alternates first player every game: first_player = game_id % 2]
- [Source: architecture.md#Evaluation Pattern — Evaluation MUST instantiate same BriscasEnv]
- [Source: architecture.md#Complete Project Directory Structure — evaluation/evaluate.py, scripts/evaluate.py]
- [Source: architecture.md#Architectural Boundaries — Boundary 2 and 3]
- [Source: architecture.md#Error Handling — logging module, exceptions propagate]
- [Source: epics.md#Story 3.1 — Acceptance Criteria]
- [Source: 2-3-load-and-resume-trained-models.md — load_agent(), game loop pattern, 134 tests]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

### Completion Notes List

- Task 0: Added `agent_points` and `opponent_points` to `info` dict in `BriscasEnv.step()` when `terminated=True`. 3 new tests in `TestPointScoresInInfo`. All 170 tests pass.
- Task 1: Created `evaluation/evaluate.py` with `run_evaluation()`. Validates exactly one agent is `"random"`. Engine does not support first-player alternation — `first_player=0` always, warning logged. CSV output with correct columns and naming convention. `_extract_agent_name()` strips `.zip` and extracts basename.
- Task 2: Created `scripts/evaluate.py` CLI entry point. Follows same patterns as `scripts/train.py` (argparse, logging, sys.path, set_all_seeds). Dropped `--engine-url` since evaluation uses LocalAdapter (no HTTP).
- Task 3: 15 new tests (11 unit + 4 name extraction + 1 integration). All 185 tests pass.
- Code Review: Fixed 7 issues (1 HIGH, 3 MEDIUM, 3 LOW). Added `set_all_seeds(seed)` inside `run_evaluation()` for self-contained reproducibility. Added `output_path` parameter for full file path override (AC5). Added `num_games` validation. Downgraded first-player warning to info. Added 5 new tests (env.close on exception, reward_scale=1.0, output_path override, num_games validation). All 190 tests pass.

### File List

- `gym_env/briscas_env.py` — MODIFIED: added agent_points/opponent_points to info dict on terminal step
- `evaluation/__init__.py` — NEW: exports run_evaluation
- `evaluation/evaluate.py` — NEW: run_evaluation() function and _extract_agent_name()
- `scripts/evaluate.py` — NEW: CLI entry point for evaluation
- `tests/test_briscas_env.py` — MODIFIED: added TestPointScoresInInfo (3 tests)
- `tests/test_evaluation.py` — NEW: 15 tests (unit + integration)

## Change Log

- 2026-03-07: Story 3.1 implementation — evaluation matchup system with CSV output, 15 new tests, all 185 pass
- 2026-03-07: Code review — 7 fixes (seed reproducibility, output_path override, num_games validation, warning→info, env.close test, reward_scale test, redundant default removal). 5 new tests, all 190 pass
