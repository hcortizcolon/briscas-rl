# Story 2.2: Train and Validate Worst Agent

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a researcher,
I want to train a point-minimizing agent with negated reward and validate it performs worse than random,
So that I have a confirmed anti-optimal agent for the skill-vs-luck comparison.

## Acceptance Criteria

1. **Given** the same training harness from Story 2.1, **When** `train_agent(agent_type="worst", ...)` is called, **Then** the only difference is reward is negated (`reward_scale=-1.0`) **And** all other training parameters, checkpointing, and save behavior are identical (AC: FR7)

2. **Given** training completes for the worst agent, **When** a quick validation evaluation runs against random (e.g., 1000 games), **Then** the worst agent's win rate vs random is reported **And** if win rate is not meaningfully below 50%, a warning is logged: "Worst agent may not be producing true anti-optimal play" (AC: FR7)

3. **Given** the CLI entry point `scripts/train.py`, **When** invoked with `--agent worst`, **Then** training runs with negated reward and validation is performed automatically upon completion (AC: FR7, FR16)

## Tasks / Subtasks

- [x] Task 1: Implement `validate_worst_agent()` function in `training/train.py` (AC: #2)
  - [x] Define `WORST_AGENT_WARNING_THRESHOLD = 0.45` as a module-level constant with comment: `# 5% margin below 50%; statistically significant at p < 0.001 with 1000 games`
  - [x] Add `validate_worst_agent(model, adapter: EngineAdapter, num_games: int = 1000) -> float` function — accepts an existing adapter instance, not engine_url, to reuse the connection from training
  - [x] Create a fresh `BriscasEnv(adapter=adapter, reward_scale=1.0)` — use `reward_scale=1.0` because we want to read raw game_result, not negated rewards. Note: `reward_scale` doesn't affect `info["game_result"]` (computed from raw points), but 1.0 is semantically clearer for an evaluation context.
  - [x] Run `num_games` evaluation games: loop `env.reset()` / `model.predict(obs, deterministic=True)` / `env.step(action)` until terminal, track wins/losses/draws from `info["game_result"]`
  - [x] `model.predict(obs)` returns `(action, _states)` where `action` is `np.ndarray`. SB3 internally handles unsqueezing a raw (non-batched) observation, so passing a 1D obs from a raw env works correctly. Use `action.item()` or `int(action[0])` to get the scalar for `env.step()`.
  - [x] Return win rate as float (0.0 to 1.0)
  - [x] Log summary: "Worst agent validation: {wins}W / {losses}L / {draws}D over {num_games} games | Win rate: {rate:.1f}%"
  - [x] If win rate >= `WORST_AGENT_WARNING_THRESHOLD`, log WARNING: "Worst agent may not be producing true anti-optimal play — win rate {rate:.1f}% is not meaningfully below 50%"
  - [x] Clean up env with `env.close()` in finally block

- [x] Task 2: Integrate validation into `train_agent()` (AC: #2, #3)
  - [x] After model save and metadata write, add: `if agent_type == "worst":` block that calls `validate_worst_agent(model, adapter)`
  - [x] Wrap validation call in its own try/except block — if validation fails (e.g., `EngineConnectionError`), log WARNING "Validation failed: {error}" but do NOT crash. The trained model and base metadata are already saved at this point.
  - [x] On successful validation: re-open metadata JSON, add `validation_win_rate` and `validation_games` fields, re-save. This ensures base metadata is always written even if validation fails.
  - [x] On failed validation: metadata JSON retains base fields only (no validation fields), and a warning is logged. The model is still usable.
  - [x] Note: AC #1 (reward negation, identical parameters) is already fully covered by existing Story 2.1 code and tests — `train_agent()` sets `reward_scale=-1.0` when `agent_type="worst"`. No new code needed for AC #1.

- [x] Task 3: Write tests (AC: #1, #2, #3)
  - [x] **Unit tests (mocked env):**
    - [x] Test `validate_worst_agent()` returns correct win rate from simulated game results
    - [x] Test warning logged when win rate >= 0.45
    - [x] Test no warning logged when win rate < 0.45
    - [x] Test env.close() called even if validation raises exception
    - [x] Test validation failure (EngineConnectionError) is caught gracefully — model and base metadata still saved, warning logged
    - [x] Test metadata JSON includes `validation_win_rate` and `validation_games` when agent_type is "worst" and validation succeeds
    - [x] Test metadata JSON does NOT include validation fields when agent_type is "best"
    - [x] Test metadata JSON does NOT include validation fields when agent_type is "worst" but validation fails
  - [x] **Integration test (real SB3, mocked adapter):** `@pytest.mark.integration`
    - [x] Run `train_agent(agent_type="worst", total_timesteps=100, ...)` with mocked adapter
    - [x] Verify training completes with negated reward (reward_scale=-1.0)
    - [x] Verify validation runs after training (model.predict is called in validation loop)
    - [x] Verify `.zip` model file and `.json` metadata file are created
    - [x] ~100 timesteps + 10 validation games, should complete in <10 seconds

- [x] Task 4: Update `training/__init__.py` exports (AC: #1)
  - [x] Add `validate_worst_agent` to `__all__` list

## Dev Notes

### Technical Requirements

- **Validation is a simple eval loop, NOT the full evaluation module from Epic 3.** Do NOT create `evaluation/evaluate.py` yet — that's Story 3.1. Validation here is a minimal loop inside `training/train.py` that confirms the worst agent underperforms random.
- **Model predict:** Use `model.predict(obs, deterministic=True)` — deterministic=True disables exploration noise for evaluation. Returns `(action, _states)` tuple where `action` is `np.ndarray`. SB3 handles unsqueezing raw (non-batched) observations internally, so passing a 1D obs from a raw env works correctly. Use `action.item()` or `int(action[0])` for `env.step()`.
- **CRITICAL: Don't use DummyVecEnv for validation.** Create a raw `BriscasEnv` (not wrapped by SB3). Call `env.reset()` and `env.step()` directly. Use `model.predict()` which handles the observation preprocessing internally even for unwrapped envs. This avoids the auto-reset complexity of DummyVecEnv.
- **Warning threshold constant:** `WORST_AGENT_WARNING_THRESHOLD = 0.45` — module-level named constant with comment. The epics say "meaningfully below 50%". A 45% threshold gives 5% margin. With 1000 games, this is statistically distinguishable from random (50%) at p < 0.001. Do NOT use a magic number in the comparison.
- **Validation failure is non-fatal.** If `validate_worst_agent()` raises (e.g., engine goes down between training and validation), `train_agent()` catches the exception, logs a warning, and proceeds. The trained model and base metadata are already saved before validation runs. This is graceful degradation — a failed validation gate is better than losing a trained model.
- **Reward scale for validation env:** Use `reward_scale=1.0` during validation. We only need `info["game_result"]` which is computed from raw (unscaled) point differential. The reward_scale doesn't affect game_result, but using 1.0 is clearer.
- **Terminology:** Continue using "timesteps" and "games" — never "episodes".
- **Validation game count:** Default 1000 games. Not configurable via CLI for now — hardcoded in `train_agent()`. If needed later, add `--validation-games` flag.

### Architecture Compliance

- **File locations:** All changes in `training/train.py` — no new files needed. [Source: architecture.md#Complete Project Directory Structure]
- **Boundary 3 preserved:** Validation logic is in `training/train.py` (library code), not in `scripts/train.py`. [Source: architecture.md#Architectural Boundaries]
- **One agent, parameterized:** `train_agent()` already handles worst agent via `reward_scale=-1.0`. Validation is added as a conditional post-training step. [Source: architecture.md#Architectural Principles]
- **Adapter reuse:** `validate_worst_agent()` receives the existing `adapter` instance from `train_agent()` — no redundant `RESTAdapter` creation. Validation creates a new `BriscasEnv` with this adapter (different `reward_scale`), preserving the adapter pattern boundary. [Source: architecture.md#Environment Wrapper Contract]
- **Error handling:** `logging` module for all output. Training errors propagate (unchanged). Validation errors are caught and logged as warnings — non-fatal because model is already saved. [Source: architecture.md#Error Handling]
- **Worst agent validation gate:** Architecture explicitly calls this out as a story-level validation step. [Source: epics.md#Epic 2 Notes]

### Library / Framework Requirements

No new dependencies. All imports from SB3 (already installed) and stdlib:

| Library | Version | Purpose | Notes |
|---|---|---|---|
| `stable-baselines3` | >=2.7.0,<3.0 | `model.predict()` for eval | Already in requirements.txt |
| `gymnasium` | >=1.2.0,<2.0 | `BriscasEnv` for eval loop | Already in requirements.txt |
| `logging` | stdlib | Warning/info messages | No install needed |

### File Structure Requirements

Files to create/modify in this story:
```
briscas_rl/
├── training/
│   ├── __init__.py                 # MODIFIED — add validate_worst_agent to __all__
│   └── train.py                    # MODIFIED — add validate_worst_agent(), integrate into train_agent()
├── tests/
│   └── test_training.py            # MODIFIED — add validation tests
└── (no other files changed)
```

### Testing Standards

- Use `pytest` as testing framework [Source: architecture.md#Development Tooling]
- Add tests to existing `tests/test_training.py` — do NOT create a new test file
- **Two test tiers:**
  - Unit tests: mock `BriscasEnv` and `model.predict()`, test validation logic, warning thresholds, metadata fields
  - Integration test: `@pytest.mark.integration` — real SB3, mocked adapter, verify worst agent training + validation end-to-end
- All 111 existing tests MUST continue passing
- Expected new tests: ~8-10 unit tests + 1 integration test
- **AC #1 coverage note:** AC #1 (reward negation, identical parameters) is already tested by Story 2.1's existing unit tests for `train_agent()` with `agent_type="worst"`. No new tests needed for AC #1 — just verify existing tests still pass.

### Previous Story Intelligence (Story 2.1)

**What was built:**
- `training/train.py` — `train_agent()` with full DQN setup, `WinRateCallback`, checkpointing, metadata JSON
- `scripts/train.py` — CLI with `--agent {best,worst}`, `--timesteps`, `--seed`, `--output`, `--engine-url`, `--checkpoint-freq`
- `reward_scale=-1.0` for worst agent already wired in `train_agent()`
- 111 tests passing (92 env + 19 training)

**Critical bug fix from code review:** WinRateCallback reads `infos[0]["game_result"]` directly — NOT `infos[0]["terminal_info"]["game_result"]`. SB3 v2.7.x DummyVecEnv preserves terminal info directly in `infos[0]`.

**Pattern for validation:** The eval loop should use `model.predict(obs, deterministic=True)` on a raw (unwrapped) env. SB3's `model.predict()` handles observation preprocessing even when the env isn't wrapped in DummyVecEnv.

### Git Intelligence

**Recent commits (last 5):**
1. `8e50c52` — Story 2.1 code review: Fixed WinRateCallback to read game_result directly from infos
2. `e63ca5c` — Story 2.1: Reward scaling, game result tracking, training module, WinRateCallback
3. `bc3504d` — Story 2.1: Story file creation
4. `6cdb434` — Story 1.3: Enhanced seeding with validation
5. `fc3cb08` — Story 1.3: Initial seeding implementation

**Code conventions confirmed:**
- `logging.getLogger(__name__)` for logger setup
- `__init__.py` with `__all__` exports
- Single-line docstrings
- `unittest.mock` for mocking in tests
- `@pytest.mark.integration` for integration tests
- try/finally with `env.close()` for cleanup

### Project Context Reference

- PRD: `_bmad-output/planning-artifacts/prd.md` — FR7 (train worst agent)
- Architecture: `_bmad-output/planning-artifacts/architecture.md` — Agent Training Pattern, Worst Agent Validation Gate
- Epics: `_bmad-output/planning-artifacts/epics.md` — Epic 2, Story 2.2 acceptance criteria
- Previous story: `_bmad-output/implementation-artifacts/2-1-train-best-agent-with-dqn.md` — full training harness, patterns, 111 tests

### References

- [Source: architecture.md#Agent Training Pattern]
- [Source: architecture.md#Architectural Principles — "One agent architecture, parameterized"]
- [Source: architecture.md#Architectural Boundaries — Boundary 3: Scripts ↔ Library Code]
- [Source: architecture.md#Error Handling]
- [Source: epics.md#Epic 2 Notes — "Worst agent validation gate"]
- [Source: epics.md#Story 2.2 — Acceptance Criteria]
- [Source: prd.md#Agent Training — FR7]
- [Source: 2-1-train-best-agent-with-dqn.md — Training harness, WinRateCallback, code review fixes]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

None — clean implementation, all tests passed first run.

### Completion Notes List

- Implemented `validate_worst_agent()` in `training/train.py` with eval loop using raw BriscasEnv (reward_scale=1.0), deterministic model.predict(), and win/loss/draw tracking
- Added `WORST_AGENT_WARNING_THRESHOLD = 0.45` module-level constant
- Integrated validation as post-training step for worst agent in `train_agent()` with graceful error handling (try/except around validation, base metadata always saved first)
- On successful validation, metadata JSON is re-opened and `validation_win_rate`/`validation_games` fields added
- Added 10 new tests (5 unit tests for `validate_worst_agent`, 4 unit tests for validation integration in `train_agent`, 1 integration test for worst agent end-to-end)
- Updated `training/__init__.py` to export `validate_worst_agent`
- All 121 tests passing (was 111), 0 regressions

#### Code Review Fixes (2026-03-03)

- **H1**: Added `VALIDATION_NUM_GAMES = 1000` constant — `train_agent()` now passes it explicitly to `validate_worst_agent()` and writes it to metadata, eliminating fragile hardcoded coupling
- **H2**: Rewrote `test_integration_train_worst_agent` to exercise real `validate_worst_agent()` instead of mocking it — now verifies model.predict is actually called in the validation loop
- **M1**: Training env is now closed before validation runs, preventing shared adapter issues (double `delete_game()` on same session)
- **M2**: `validate_worst_agent()` now explicitly checks for `"draw"` and logs a warning for unexpected `game_result` values instead of silently counting them as draws
- **L1**: Added threshold boundary test (`win_rate == 0.45` triggers warning)
- **L2**: Added `num_games <= 0` guard with `ValueError`
- All 123 tests passing (was 121), 0 regressions

### Change Log

- 2026-03-03: Story 2.2 implementation complete — validate_worst_agent(), train_agent() integration, 10 new tests
- 2026-03-03: Code review — fixed 6 issues (2 HIGH, 2 MEDIUM, 2 LOW), added 2 tests, 123 total passing

### File List

- `training/train.py` — MODIFIED: added WORST_AGENT_WARNING_THRESHOLD, VALIDATION_NUM_GAMES, validate_worst_agent(), validation integration in train_agent(), explicit draw handling, num_games guard
- `training/__init__.py` — MODIFIED: added validate_worst_agent and VALIDATION_NUM_GAMES to __all__
- `tests/test_training.py` — MODIFIED: added TestValidateWorstAgent (7 tests), TestTrainAgentValidation (4 tests), test_integration_train_worst_agent (1 test)
