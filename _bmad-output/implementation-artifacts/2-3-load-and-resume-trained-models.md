# Story 2.3: Load and Resume Trained Models

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a researcher,
I want to load a previously saved agent model from disk,
So that I can evaluate it or continue training without retraining from scratch.

## Acceptance Criteria

1. **Given** a saved SB3 model `.zip` file exists at a known path, **When** `load_agent(path)` is called, **Then** the model is loaded via `DQN.load(path)` **And** the loaded model can be used for evaluation via `.predict(observation)` to select actions (AC: FR10). **Note:** Resume training via `.learn()` is SB3-native capability — the caller sets the env via `model.set_env(env)` after loading. This is SB3's contract, not tested by this story.

2. **Given** a model path that does not exist, **When** `load_agent(path)` is called, **Then** a `FileNotFoundError` is raised with a clear message indicating the file was not found (AC: FR10)

3. **Given** the metadata JSON alongside the model, **When** `load_agent(path)` is called, **Then** the metadata (agent_type, seed, total_timesteps, reward_type, timestamp) is loaded and returned alongside the model for logging and verification (AC: FR10)

4. **Given** the metadata JSON does not exist alongside the model, **When** `load_agent(path)` is called, **Then** the model is still loaded successfully **And** a warning is logged indicating metadata was not found **And** metadata is returned as `None` (AC: FR10)

## Tasks / Subtasks

- [x] Task 1: Implement `load_agent()` function in `training/train.py` (AC: #1, #2, #3, #4)
  - [x] Add `load_agent(model_path: str) -> tuple[DQN, dict | None]` function
  - [x] Normalize `model_path`: if path ends with `.zip`, strip it (SB3's `DQN.load()` expects path without extension). If not, use as-is.
  - [x] Check if model file exists: look for `{model_path}.zip` — if not found, raise `FileNotFoundError` with message: `"Model file not found: {model_path}.zip"`
  - [x] Load model via `DQN.load(model_path)` — SB3 handles .zip internally
  - [x] Look for metadata at `{model_path}.json` — if exists, load and parse JSON; if not, log WARNING: `"No metadata file found at {model_path}.json"` and set metadata to `None`
  - [x] Log INFO: `"Loaded agent from {model_path}.zip | Agent type: {agent_type} | Trained: {total_timesteps} timesteps"` (if metadata available) or `"Loaded agent from {model_path}.zip | No metadata available"` (if not)
  - [x] Return `(model, metadata)` tuple

- [x] Task 2: Write tests (AC: #1, #2, #3, #4)
  - [x] **Unit tests (mocked SB3):**
    - [x] Test `load_agent()` calls `DQN.load()` with correct path
    - [x] Test `load_agent()` returns `(model, metadata)` tuple when metadata exists
    - [x] Test `load_agent()` returns `(model, None)` when metadata JSON missing, and warning logged
    - [x] Test `load_agent()` raises `FileNotFoundError` when `.zip` file doesn't exist
    - [x] Test `load_agent()` strips `.zip` extension from input path if provided
    - [x] Test `load_agent()` handles path without `.zip` extension
    - [x] Test metadata contents are correct (agent_type, seed, total_timesteps, reward_type, timestamp)
    - [x] Test corrupted/invalid `.zip` file — SB3 exception propagates (not caught by `load_agent()`)
    - [x] Test double `.zip` extension edge case (`path.zip.zip`) — only one `.zip` stripped, SB3 resolves the rest
  - [x] **Integration test (real SB3, mocked adapter):** `@pytest.mark.integration`
    - [x] Train a model with `train_agent()` (100 timesteps), then load it with `load_agent()`
    - [x] Verify loaded model can call `.predict()` and returns valid action (numpy array)
    - [x] Verify metadata round-trip: `agent_type`, `seed`, `total_timesteps`, `reward_type`, `timestamp` all match what was saved
    - [x] ~100 timesteps training + load, should complete in <10 seconds

- [x] Task 3: Update `training/__init__.py` exports (AC: #1)
  - [x] Add `load_agent` to imports and `__all__` list

## Dev Notes

### Technical Requirements

- **This is a thin wrapper around SB3's `DQN.load()`.** Do NOT over-engineer. SB3 handles all model deserialization, weight loading, and optimizer state restoration. Our function just adds file existence checking, metadata loading, and logging.
- **SB3 `DQN.load()` behavior:** Accepts path without `.zip` extension (SB3 appends it internally). Returns a fully functional `DQN` model that can call `.predict()` or `.learn()`. No `env` argument needed for inference-only use; pass `env` if resuming training (SB3 re-attaches the env).
- **Resume training is SB3-native.** To continue training: `model = DQN.load(path); model.set_env(env); model.learn(additional_timesteps)`. No custom resume logic needed in `load_agent()` — the caller handles env setup. `load_agent()` is only responsible for loading the model and metadata.
- **Path normalization:** Users might pass `"models/best_agent_50k"` or `"models/best_agent_50k.zip"`. Handle both by stripping `.zip` suffix if present before passing to `DQN.load()`. Only strip once — `"foo.zip.zip"` becomes `"foo.zip"`, which is correct (SB3 appends `.zip` internally).
- **Metadata is optional but expected.** Story 2.1 always writes metadata JSON alongside the model. But models from checkpoints (`models/checkpoints/`) don't have metadata, and users might have models from other sources. Graceful degradation: load model, warn about missing metadata, return `None`.
- **`load_agent()` does NOT accept an `env` parameter.** It loads the model for inference-only use. To resume training, the caller must call `model.set_env(env)` after loading. This keeps `load_agent()` simple and avoids coupling it to env construction logic.
- **Corrupted model files:** If the `.zip` file exists but is corrupted/invalid, SB3 will raise its own exception (e.g., `zipfile.BadZipFile`). `load_agent()` does NOT catch this — let it propagate. The file existence check only guards against missing files, not invalid ones.
- **Resume training is SB3's responsibility, not ours.** `DQN.load()` restores weights + optimizer state. `model.set_env(env); model.learn(more_timesteps)` resumes training. This is SB3-native behavior — do NOT test it in this story. Our tests only verify load + predict works.
- **No CLI changes in this story.** `scripts/train.py` doesn't need a load/resume flag yet. `load_agent()` is a library function that will be consumed by `evaluation/evaluate.py` in Epic 3 (Story 3.1) and potentially by a resume-training script later.
- **Terminology:** Continue using "timesteps" — never "episodes".

### Architecture Compliance

- **File locations:** All changes in `training/train.py` — no new files needed. [Source: architecture.md#Complete Project Directory Structure]
- **Boundary 3 preserved:** `load_agent()` is library code in `training/train.py`, not in `scripts/`. [Source: architecture.md#Architectural Boundaries]
- **Agent Training Pattern:** Architecture specifies "Models saved as SB3 `.zip` files in `models/` with descriptive names" and "metadata JSON saved alongside each `.zip` model file". `load_agent()` is the read counterpart to this write pattern. [Source: architecture.md#Agent Training Pattern]
- **Error handling:** Use Python `logging` module. `FileNotFoundError` for missing model file (clear, standard Python exception). Missing metadata is a warning, not an error. [Source: architecture.md#Error Handling]

### Library / Framework Requirements

No new dependencies. All imports from SB3 (already installed) and stdlib:

| Library | Version | Purpose | Notes |
|---|---|---|---|
| `stable-baselines3` | >=2.7.0,<3.0 | `DQN.load()` for model loading | Already in requirements.txt |
| `json` | stdlib | Parse metadata JSON | No install needed |
| `logging` | stdlib | Info/warning messages | No install needed |
| `os` | stdlib | File existence check | No install needed |

### File Structure Requirements

Files to create/modify in this story:
```
briscas_rl/
├── training/
│   ├── __init__.py                 # MODIFIED — add load_agent to __all__
│   └── train.py                    # MODIFIED — add load_agent()
├── tests/
│   └── test_training.py            # MODIFIED — add load_agent tests
└── (no other files changed)
```

### Testing Standards

- Use `pytest` as testing framework [Source: architecture.md#Development Tooling]
- Add tests to existing `tests/test_training.py` — do NOT create a new test file
- **Two test tiers:**
  - Unit tests: mock `DQN.load()`, test path normalization, file checks, metadata loading, error handling
  - Integration test: `@pytest.mark.integration` — real SB3 train + load round-trip, verify model predict works
- All 123 existing tests MUST continue passing
- Expected new tests: ~9 unit tests + 1 integration test

### Previous Story Intelligence (Story 2.2)

**What was built:**
- `training/train.py` — `validate_worst_agent()`, `WORST_AGENT_WARNING_THRESHOLD`, `VALIDATION_NUM_GAMES` constants
- Metadata JSON now includes `validation_win_rate` and `validation_games` for worst agents
- 123 tests passing (92 env + 31 training)

**Code review fixes applied:**
- `VALIDATION_NUM_GAMES` constant for explicit game count
- Training env closed before validation (shared adapter fix)
- Explicit draw handling in validation
- `num_games <= 0` guard

**Patterns confirmed:**
- `logging.getLogger(__name__)` for logger setup
- `__init__.py` with `__all__` exports
- Single-line docstrings
- `unittest.mock` for mocking in tests
- `@pytest.mark.integration` for integration tests
- try/finally with `env.close()` for cleanup
- Metadata JSON written to `{output_path}.json`

### Git Intelligence

**Recent commits (last 5):**
1. `909435e` — Story 2.2 code review: validation game count guard, explicit draw handling
2. `82c9218` — Story 2.2: validate_worst_agent(), integration into train_agent()
3. `8e50c52` — Story 2.1 code review: Fixed WinRateCallback
4. `e63ca5c` — Story 2.1: Training module, WinRateCallback, reward scaling
5. `bc3504d` — Story 2.1: Story file creation

**Code conventions confirmed:**
- `logging.getLogger(__name__)` for logger setup
- `__init__.py` with `__all__` exports
- Single-line docstrings
- `unittest.mock` for mocking in tests
- `@pytest.mark.integration` for integration tests
- Metadata JSON at `{path}.json` alongside model `.zip`

### Project Context Reference

- PRD: `_bmad-output/planning-artifacts/prd.md` — FR10 (load trained model)
- Architecture: `_bmad-output/planning-artifacts/architecture.md` — Agent Training Pattern, model persistence
- Epics: `_bmad-output/planning-artifacts/epics.md` — Epic 2, Story 2.3 acceptance criteria
- Previous story: `_bmad-output/implementation-artifacts/2-2-train-and-validate-worst-agent.md` — validation, metadata patterns, 123 tests

### References

- [Source: architecture.md#Agent Training Pattern — "Models saved as SB3 .zip files"]
- [Source: architecture.md#Agent Training Pattern — "metadata JSON saved alongside each .zip model file"]
- [Source: architecture.md#Architectural Boundaries — Boundary 3: Scripts ↔ Library Code]
- [Source: architecture.md#Error Handling]
- [Source: epics.md#Story 2.3 — Acceptance Criteria]
- [Source: prd.md#Agent Training — FR10]
- [Source: 2-2-train-and-validate-worst-agent.md — Metadata patterns, 123 tests]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

- Integration test initially used wrong observation dim (42 instead of 13) — fixed to match BriscasEnv's 13-dim observation space.

### Completion Notes List

- Implemented `load_agent(model_path)` in `training/train.py` — thin wrapper around `DQN.load()` with path normalization, file existence check, metadata loading, and logging.
- 9 unit tests (mocked SB3) covering: correct DQN.load call, model+metadata tuple return, missing metadata warning, FileNotFoundError, .zip stripping, path without extension, metadata contents, corrupted zip propagation, double .zip edge case.
- 1 integration test: real SB3 train→load round-trip with predict verification and metadata round-trip.
- Updated `training/__init__.py` with `load_agent` export.
- All 133 tests passing (123 existing + 10 new). Zero regressions.

### File List

- `training/train.py` — MODIFIED (added `load_agent()` function)
- `training/__init__.py` — MODIFIED (added `load_agent` and `WORST_AGENT_WARNING_THRESHOLD` to imports and `__all__`)
- `tests/test_training.py` — MODIFIED (added `TestLoadAgentUnit` class with 10 unit tests + 1 integration test)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` — MODIFIED (updated 2-3 status to review)

### Change Log

- 2026-03-03: Implemented Story 2.3 — `load_agent()` function with tests (10 new tests, 133 total passing)
- 2026-03-03: Code review fixes — Added malformed JSON handling in `load_agent()`, added test for corrupted metadata, added action range assertion in integration test, removed unused imports, exported `WORST_AGENT_WARNING_THRESHOLD`, updated File List (134 total passing)
