# Story 1.3: Reproducible Seeding

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a researcher,
I want to set a single seed that propagates to all random sources,
So that my training and evaluation runs are reproducible.

## Acceptance Criteria

1. **Given** a seed value (e.g., 42), **When** `set_all_seeds(seed)` is called from `seed.py`, **Then** `random.seed(seed)`, `numpy.random.seed(seed)`, `torch.manual_seed(seed)` are each called directly **And** CUDA determinism is enabled when GPU is available (`torch.backends.cudnn.deterministic=True`, `torch.backends.cudnn.benchmark=False`) **And** the seed value is logged **And** the game engine's seed/shuffle endpoint is skipped (not available in current engine) (AC: FR16)

2. **Given** `set_all_seeds(42)` is called, **When** checking random state immediately after, **Then** `numpy.random.random()` produces the same value across separate invocations with the same seed **And** `torch.rand(1)` produces the same value across separate invocations with the same seed (AC: FR17)

## Tasks / Subtasks

- [x] Task 1: Create `seed.py` — `set_all_seeds(seed: int) -> None` function (AC: #1)
  - [x] Call `random.seed(seed)` (Python stdlib)
  - [x] Call `numpy.random.seed(seed)`
  - [x] Call `torch.manual_seed(seed)`
  - [x] If `torch.cuda.is_available()`: call `torch.cuda.manual_seed_all(seed)`, set `torch.backends.cudnn.deterministic = True`, set `torch.backends.cudnn.benchmark = False`
  - [x] Log the seed value being set via `logging.getLogger(__name__)`
  - [x] Add comment documenting that engine seeding is not available (no seed endpoint in current game engine) and `PYTHONHASHSEED` limitation
  - [x] Add input validation: `TypeError` for non-int seeds (including bool), `ValueError` for out-of-range seeds (< 0 or >= 2^32)
- [x] Task 2: Write tests `tests/test_seed.py` (AC: #1, #2)
  - [x] Test that `set_all_seeds(42)` results in `numpy.random.random()` producing the same value on repeated calls with same seed
  - [x] Test that `set_all_seeds(42)` results in `torch.rand(1)` producing the same value on repeated calls with same seed
  - [x] Test that `set_all_seeds(42)` results in `random.random()` producing the same value on repeated calls with same seed
  - [x] Test that two separate calls with different seeds produce different numpy random values
  - [x] Test that `seed=0` works correctly (edge case)
  - [x] Test that logging output includes the seed value (use `caplog` fixture)
  - [x] Test multi-element torch tensor reproducibility with same seed
  - [x] Test CUDA branch is exercised when GPU available (mock `torch.cuda.is_available`)
  - [x] Test CUDA branch is skipped when no GPU (mock `torch.cuda.is_available`)
  - [x] Test negative seed raises `ValueError`
  - [x] Test seed >= 2^32 raises `ValueError`
  - [x] Test non-int seed raises `TypeError`

## Dev Notes

### Technical Requirements

- **Direct seeding approach:** Call `random.seed()`, `numpy.random.seed()`, and `torch.manual_seed()` directly — do NOT wrap SB3's `set_random_seed`. This keeps `seed.py` independent of the RL library, makes each seed source explicit and traceable to the architecture spec, and avoids coupling a foundational utility to SB3.
- **Function signature:** `def set_all_seeds(seed: int) -> None` — single parameter, auto-detects CUDA. Do NOT expose `using_cuda` as a parameter; keep the API simple ("one seed controls everything").
- **CUDA determinism:** When `torch.cuda.is_available()` is True, call `torch.cuda.manual_seed_all(seed)` and set `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` for full GPU reproducibility on the RTX 3060Ti.
- **Engine seeding:** The game engine uses `random.shuffle()` with no external seed control. No seed/shuffle API endpoint exists. The architecture says to call the engine's seed endpoint "if available" — it is NOT available, so skip it. Document this limitation clearly in `seed.py`.
- **Implication:** Game-level randomness (card deals) will NOT be reproducible across engine calls. Reproducibility applies to the RL training side only (policy initialization, exploration noise, batch sampling). Over thousands of episodes, training convergence should still be reproducible given the same seed for the RL components.

### Architecture Compliance

- **File location:** Top-level `seed.py` per architecture.md project structure. NOT in `gym_env/` or a `utils/` directory. [Source: architecture.md#Complete Project Directory Structure]
- **Shared utility pattern:** `seed.py` is a top-level module consumed by both `scripts/train.py` and `scripts/evaluate.py` in later stories. Import pattern: `from seed import set_all_seeds` — works when scripts are run from project root (e.g., `python scripts/train.py`). No `sys.path` manipulation needed. [Source: architecture.md#Structure Patterns]
- **Seed propagation spec:** Architecture calls for 4 sources: numpy, torch, SB3, engine. Direct calls cover numpy and torch. Python `random` is also seeded (SB3's internal `set_random_seed` would have done this too). Engine source skipped (not available). [Source: architecture.md#Seed Propagation]
- **No SB3 dependency:** `seed.py` imports only `random`, `numpy`, `torch`, and `logging`. It does NOT import from `stable_baselines3`. This keeps the seed utility decoupled from the RL library — evaluation scripts can use it without implying SB3 is needed for seeding.
- **Naming conventions:** snake_case function (`set_all_seeds`), snake_case file (`seed.py`). [Source: architecture.md#Code Naming Conventions]
- **Error handling:** Use `logging` module, not print statements. [Source: architecture.md#Error Handling]
- **No unnecessary abstraction:** Single function, no class needed. Keep it minimal.

### Library / Framework Requirements

| Library | Version | Purpose | Notes |
|---|---|---|---|
| `random` | stdlib | `random.seed()` | Python stdlib, no install needed |
| `torch` | >=2.3.0 | `torch.manual_seed()`, CUDA determinism | Already in requirements.txt |
| `numpy` | (transitive) | `numpy.random.seed()` | Already available |
| `logging` | stdlib | Seed value logging | Python stdlib, no install needed |
| `pytest` | latest stable | Testing | Already in requirements.txt |

**No new dependencies needed.** `seed.py` imports only `random`, `logging`, `numpy`, and `torch` — does NOT import `stable-baselines3`.

### Project Structure Notes

Files to create in this story:
```
briscas_rl/
├── seed.py                    # NEW — set_all_seeds(seed) function
├── tests/
│   └── test_seed.py           # NEW — seed reproducibility tests
├── gym_env/                   # NO CHANGES
└── requirements.txt           # NO CHANGES
```

Alignment with architecture.md project structure: exact match. `seed.py` is at the project root as specified. [Source: architecture.md#Complete Project Directory Structure]

### Testing Standards

- Use `pytest` as testing framework [Source: architecture.md#Development Tooling]
- Test file mirrors source: `tests/test_seed.py` ↔ `seed.py` [Source: architecture.md#Structure Patterns]
- All Story 1.1 tests (23) and Story 1.2 tests (50) MUST continue passing — do not modify any existing files. Run full `pytest` suite as final verification.
- Tests verify reproducibility by calling `set_all_seeds` twice with the same seed and asserting identical random output from `random`, `numpy`, and `torch`
- No SB3 mocking needed — all seed calls are direct stdlib/numpy/torch calls, testable through outcomes
- **Expected test count:** ~6 new tests in `test_seed.py`, bringing project total to ~79 (73 existing + 6 new). Dev agent should verify full suite passes.
- **Known limitation:** `PYTHONHASHSEED` affects Python `set()` and `dict()` ordering. For fully deterministic runs, callers should set `PYTHONHASHSEED=0` in their environment. This is outside the scope of `set_all_seeds()` but should be documented in `seed.py` as a comment.

### Previous Story Intelligence (Story 1.2)

**What was built:**
- `gym_env/observation.py` — card encoding constants, observation space
- `gym_env/briscas_env.py` — `BriscasEnv(gymnasium.Env)` with full Gymnasium 1.2.x compliance
- 73 tests total (23 from Story 1.1 + 50 from Story 1.2)

**Patterns established:**
- `logging.getLogger(__name__)` for logger setup
- `__init__.py` exports all public types via `__all__`
- Single-line `"""..."""` docstrings
- Tests use `unittest.mock` for mocking

**Key insight:** `BriscasEnv.reset(seed=seed)` calls `super().reset(seed=seed)` which sets `self.np_random` — this is Gymnasium's internal RNG. Our `set_all_seeds()` is separate from this: it seeds the global RNG state for numpy, torch, and Python's random module. Both mechanisms coexist — `set_all_seeds()` for global reproducibility, `env.reset(seed=)` for env-specific RNG.

### Git Intelligence

**Recent commits (4 total):**
1. `d31f6d2` — Story 1.2: BriscasEnv, observation encoding, 48 tests, .gitignore
2. `98b8684` — Story 1.1: EngineAdapter, RESTAdapter, dataclasses, 23 tests
3. `719fda7` — Planning artifacts
4. `366f6ff` — First commit

**Code conventions observed:**
- Inline type hints, no `.pyi` stubs
- Single-line docstrings
- `logging.getLogger(__name__)` pattern
- Module-level functions (no unnecessary classes)

### Latest Tech Information

**PyTorch 2.3+ reproducibility:**
- `torch.manual_seed(seed)` seeds both CPU and CUDA generators
- `torch.backends.cudnn.deterministic = True` ensures deterministic convolutions
- `torch.backends.cudnn.benchmark = False` disables auto-tuning (which introduces non-determinism)
- `torch.use_deterministic_algorithms(True)` is the strictest option but may cause errors for some ops — NOT needed for DQN (fully-connected network)

**Engine limitation:**
- Game engine uses `random.shuffle()` with Python's global random state
- Our `set_all_seeds` calls `random.seed(seed)` which seeds the local process's global random module
- This means if the engine ran in the same Python process, card shuffles would also be deterministic
- However, the engine runs as a separate REST API process — its `random` state is independent
- Card deal randomness is NOT controllable from the RL training side

### References

- [Source: architecture.md#Seed Propagation]
- [Source: architecture.md#Complete Project Directory Structure]
- [Source: architecture.md#Structure Patterns — "Shared utilities: Top-level modules"]
- [Source: architecture.md#Code Naming Conventions]
- [Source: architecture.md#Error Handling]
- [Source: epics.md#Story 1.3 — Acceptance criteria and BDD scenarios]
- [Source: prd.md#Reproducibility — FR16, FR17]
- [Source: 1-2-gymnasium-environment-with-observation-and-actions.md — Previous story patterns]
- [Source: PyTorch docs — Reproducibility]
- [ADR: Direct seeding over SB3 wrapper — architecture traceability, no RL library coupling, explicit seed sources]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

None — clean implementation, no issues encountered.

### Completion Notes List

- Implemented `seed.py` at project root with `set_all_seeds(seed: int) -> None`
- Seeds `random`, `numpy`, and `torch`; conditionally seeds CUDA and sets deterministic flags
- Documented engine seeding limitation and `PYTHONHASHSEED` caveat as inline comments
- Added input validation: `TypeError` for non-int/bool seeds, `ValueError` for out-of-range ([0, 2^32-1])
- 12 tests in `tests/test_seed.py` covering reproducibility, different seeds, edge case (seed=0), logging, CUDA branch mocking, and input validation
- Full suite: 85/85 tests pass (73 existing + 12 new), 0 regressions

### Change Log

- 2026-03-01: Implemented story 1-3-reproducible-seeding — created seed.py and tests/test_seed.py
- 2026-03-01: Code review — fixed Dev Agent Record (test count 6→12, suite 79→85), documented input validation in Tasks/Subtasks, added .gitignore to File List, corrected file statuses from (NEW) to (MODIFIED)

### File List

- seed.py (MODIFIED — added input validation)
- tests/test_seed.py (MODIFIED — added 6 tests for CUDA mocking, validation, multi-element)
- .gitignore (MODIFIED — added lets-play-brisca/ exclusion)
