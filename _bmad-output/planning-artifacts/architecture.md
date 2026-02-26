---
stepsCompleted:
  - 1
  - 2
  - 3
  - 4
  - 5
  - 6
  - 7
  - 8
lastStep: 8
status: 'complete'
completedAt: '2026-02-25'
inputDocuments:
  - prd.md
  - prd-validation-report.md
workflowType: 'architecture'
project_name: 'briscas_rl'
user_name: 'Caleb'
date: '2026-02-24'
---

# Architecture Decision Document

_This document builds collaboratively through step-by-step discovery. Sections are appended as we work through each architectural decision together._

## Project Context Analysis

### Requirements Overview

**Functional Requirements:**
17 FRs across 4 categories: Environment Integration (5), Agent Training (6), Game Evaluation (5), Reproducibility (2). All map to a straightforward training-then-evaluation pipeline.

**Non-Functional Requirements:**
1 explicit NFR (API error handling). Implicit NFRs derived from PRD context:
- Training must converge in practical time on local hardware (RTX 3060Ti)
- 10,000-game evaluation runs must complete in reasonable time
- Trained models must be saveable/loadable across sessions

**Scale & Complexity:**

- Primary domain: ML training pipeline (local, single-user)
- Complexity level: Low
- Estimated architectural components: 4 (environment wrapper, training harness, evaluation runner, model persistence)

### Technical Constraints & Dependencies

- Existing Briscas game engine exposed via local REST API
- RTX 3060Ti for GPU-accelerated training
- PyTorch + Stable Baselines3 + Gymnasium stack
- DQN algorithm (discrete action space)
- **Action masking required:** Hand size varies (3 → 1 cards); env wrapper must mask invalid actions for SB3's DQN
- **State encoding:** Variable-length play history must be encoded to fixed-size observation vector (padding or summary strategy needed)
- **Architecture fork risk:** If REST API bottlenecks training, refactoring to direct Python calls changes the integration architecture from HTTP client to in-process import

### Cross-Cutting Concerns Identified

- **Reproducibility:** Consistent seeding across NumPy, PyTorch, and game engine shuffling
- **API reliability:** Environment wrapper must handle engine errors gracefully to avoid corrupting training data
- **Performance monitoring:** REST API latency tracking to detect training bottlenecks early
- **Worst agent validation:** Verify worst agent performs meaningfully below random before drawing skill-vs-luck conclusions -- this is a validation gate, not just a metric

### Architectural Principles (from First Principles)

- **Adapter pattern for engine integration:** Treat the game engine connection as a thin, swappable adapter. Whether REST or direct Python calls, the Gymnasium env interface stays the same -- only the adapter behind it changes.
- **Scripts, not services:** This is a training pipeline executed as scripts with shared components, not a long-running system. No service orchestration needed.
- **One agent architecture, parameterized:** Best and worst agents share identical DQN setup; only the reward signal differs. Don't duplicate agent code.

## Starter Template Evaluation

### Primary Technology Domain

Python ML training pipeline -- no web framework or starter template applicable.

### Technology Stack (from PRD)

- **Language & Runtime:** Python 3.10+ (SB3/Gymnasium compatibility)
- **ML Framework:** PyTorch (GPU training on RTX 3060Ti)
- **RL Library:** Stable Baselines3 (DQN implementation)
- **Environment Interface:** Gymnasium (env wrapper standard)
- **Game Engine Integration:** Existing Briscas engine via REST API (with direct-call fallback)

### Project Structure

```
briscas_rl/
├── gym_env/              # Gymnasium environment wrapper + engine adapter
├── training/             # Training scripts, hyperparameter configs
├── evaluation/           # Evaluation runner, matchup scripts
├── models/               # Saved trained models (.zip)
├── results/              # Evaluation CSVs and analysis output
├── tests/                # pytest: env wrapper, action masking, reward signal
├── scripts/              # Entry points: train.py, evaluate.py
└── requirements.txt      # Dependencies
```

### Game Engine Relationship

The existing Briscas game engine is an **external dependency** -- it runs as a separate local process exposing a REST API. It is not part of this repository. The `gym_env/` adapter layer abstracts this boundary, making a future switch to direct Python calls transparent to training and evaluation code.

### Development Tooling

- **Package management:** pip + requirements.txt
- **Testing:** pytest (env wrapper validation, action masking, state encoding)
- **Entry points:** CLI scripts in `scripts/` (e.g., `python scripts/train.py --agent best --episodes 50000`)
- **Reproducibility:** Seed management utility shared across scripts
- **No CI/CD needed** -- local personal project

### Note

No starter template or scaffolding tool used. Project structure follows standard Python ML conventions.

## Core Architectural Decisions

### Decision Priority Analysis

**Critical Decisions (Block Implementation):**
- State representation design
- Action space handling
- Reward signal design

**Important Decisions (Shape Architecture):**
- Training configuration management

**Deferred Decisions (Post-MVP):**
- Per-trick reward comparison (Phase 2)
- Alternative algorithm support (Phase 3)

### State Representation

- **Card encoding:** Numeric ID -- rank + suit as integers
- **Play history:** Summary -- cards seen per suit, cumulative points taken per player
- **Hand ordering:** Sorted by suit+rank for consistent representation
- **Rationale:** Compact vector, sufficient signal for DQN, avoids large one-hot encoding
- **Observation vector structure (in order):** [hand cards (3 values, padded with -1), trump card (1 value), trump suit (1 value), current trick cards (2 values, padded with -1), cards seen per suit (4 values), agent cumulative points (1 value), opponent cumulative points (1 value)]. Exact encoding of each value is implementation detail, but section order must be preserved.
- **Note:** If training struggles to converge, switching to one-hot card encoding is a diagnostic option -- numeric IDs impose false ordinal relationships that the network must learn past.

### Action Space & Masking

- **Action space:** Discrete(3) -- index into current hand
- **Invalid action handling:** `action % len(hand)` in env `step()` -- no special masking library needed
- **Rationale:** Invalid actions only occur in final tricks when hand < 3 cards (deck exhausted). Modulo wrap is trivial, keeps DQN as-is, negligible training distortion.

### Reward Signal

- **Signal:** Point differential at game end (agent points - opponent points), normalized to [-1, +1] by dividing by 120 for DQN training stability
- **Worst agent:** Negated normalized differential (maximizes opponent's advantage)
- **Timing:** End-of-game only; per-trick reward deferred to Phase 2
- **Rationale:** Differential is symmetric and cleanly invertible for worst agent. Normalization prevents reward scale issues with DQN.

### Training Configuration

- **Approach:** Hardcoded defaults with CLI overrides
- **Rationale:** Simplest approach for a personal learning project. No config files to manage.

## Implementation Patterns & Consistency Rules

### Code Naming Conventions

- **Files/modules:** snake_case (`briscas_env.py`, `train_agent.py`)
- **Classes:** PascalCase (`BriscasEnv`, `EngineAdapter`)
- **Functions/variables:** snake_case (`get_observation`, `reward_scale`)
- **Constants:** UPPER_SNAKE (`MAX_HAND_SIZE = 3`, `TOTAL_POINTS = 120`)

### Structure Patterns

- **Tests:** `tests/` directory mirroring source structure (`tests/test_briscas_env.py`)
- **Entry points:** `scripts/` directory with argparse-based CLI scripts
- **Shared utilities:** Top-level modules (e.g., `seed.py`, `config.py`) -- no `utils/` grab-bag

### Environment Wrapper Contract

The Gymnasium env is the central interface. All implementations must:
- Inherit from `gymnasium.Env`
- Define `observation_space` and `action_space` in `__init__`
- Engine communication goes through an adapter object passed to the env, never called directly
- `step()` handles action masking via `action % len(hand)`
- `reset()` starts a new game and returns initial observation
- Reward is only returned at game end (0 for intermediate steps)
- Observation encoding MUST be implemented as a single method (`_get_observation()`) on the env class. Training and evaluation both use this method -- never reimplement observation encoding outside the env.

### Agent Training Pattern

- Training function accepts: agent type (`best`/`worst`), number of episodes, seed, and output path
- Worst agent uses `reward * -1` -- no other differences from best agent
- SB3's `DQN` class used directly -- no custom subclasses unless action masking requires it
- Models saved as SB3 `.zip` files in `models/` with descriptive names (`best_agent_50k.zip`)
- When saving a model, also save a metadata JSON alongside it: `{seed, episodes, reward_type, timestamp}`. Example: `best_agent_50k.json`
- Use SB3's `CheckpointCallback` to save periodic snapshots to `models/checkpoints/` during training -- prevents total loss on crash or interruption
- Training progress monitoring: SB3's built-in logging to stdout (episode reward, loss). TensorBoard is not required for MVP but can be enabled via SB3's `TensorboardCallback` if visual training curves are desired (Phase 2).

### Evaluation Pattern

- Evaluation function accepts: two agent paths (or `"random"` for engine's random player), number of games, seed
- When agent is `"random"`, use the game engine's built-in random player -- do not implement a separate random policy. This ensures the random baseline matches what agents trained against.
- Evaluation MUST instantiate the same `BriscasEnv` class used in training
- Alternates first player every game: `first_player = game_id % 2`
- Outputs CSV with columns: `game_id, agent1_points, agent2_points, first_player, point_differential`
- Output CSV filename: `{agent1}_vs_{agent2}_{num_games}g_{seed}s.csv` (e.g., `best_vs_random_10000g_42s.csv`). CLI accepts optional `--output` flag to override.
- Summary statistics printed to stdout after completion

### Error Handling

- Engine adapter: catch connection errors, raise `EngineConnectionError` with clear message
- Training and evaluation scripts must NOT catch `EngineConnectionError` -- let exceptions propagate to the top level. Only the adapter layer catches and wraps raw connection errors.
- Use Python logging module, not print statements

### Seed Propagation

Seed must be set in all four locations:
- `numpy.random.seed()`
- `torch.manual_seed()`
- SB3's `set_random_seed()`
- Game engine's seed/shuffle endpoint (if available)

### Enforcement Guidelines

**All AI agents implementing this project MUST:**
- Follow the adapter pattern for engine communication -- never call the REST API from outside `gym_env/`
- Use the same observation encoding via `_get_observation()` across training and evaluation
- Never modify the reward signal outside the env wrapper
- Preserve seed propagation through all four random sources

## Project Structure & Boundaries

### Complete Project Directory Structure

```
briscas_rl/
├── gym_env/
│   ├── __init__.py
│   ├── briscas_env.py          # BriscasEnv(gymnasium.Env) -- observation, step, reset, reward
│   ├── engine_adapter.py       # EngineAdapter base class + RESTAdapter implementation
│   └── observation.py          # Card encoding constants, observation space definition
├── training/
│   ├── __init__.py
│   └── train.py                # train_agent(agent_type, episodes, seed, output_path)
├── evaluation/
│   ├── __init__.py
│   └── evaluate.py             # run_evaluation(agent1, agent2, num_games, seed)
├── models/
│   ├── checkpoints/            # Periodic saves during training (gitignored)
│   ├── best_agent_50k.zip      # Final trained models
│   └── best_agent_50k.json     # Metadata
├── results/                    # Evaluation CSVs (gitignored)
├── tests/
│   ├── test_briscas_env.py     # Env wrapper: observation shape, reward timing, reset
│   ├── test_engine_adapter.py  # Adapter: connection handling, error wrapping
│   ├── test_action_masking.py  # Action modulo with hand sizes 1, 2, 3
│   └── test_observation.py     # Card encoding, sorting, history summary
├── scripts/
│   ├── train.py                # CLI: python scripts/train.py --agent best --episodes 50000 --seed 42
│   └── evaluate.py             # CLI: python scripts/evaluate.py --agent1 models/best.zip --agent2 random --games 10000
├── seed.py                     # set_all_seeds(seed) -- numpy, torch, sb3, engine
├── requirements.txt
├── .gitignore                  # models/checkpoints/, results/ gitignored; final models gitignored by default (retrain from seeds or use git-lfs)
└── README.md
```

### FR-to-Structure Mapping

| FR Category | FRs | Location |
|---|---|---|
| Environment Integration | FR1-FR5 | `gym_env/briscas_env.py`, `gym_env/engine_adapter.py` |
| Agent Training | FR6-FR11 | `training/train.py`, `scripts/train.py` |
| Game Evaluation | FR12-FR15, FR18 | `evaluation/evaluate.py`, `scripts/evaluate.py` |
| Reproducibility | FR16-FR17 | `seed.py` (consumed by training + evaluation) |

### Architectural Boundaries

**Boundary 1: Game Engine ↔ Environment**
- `EngineAdapter` is the only code that touches the REST API (or future direct calls)
- `BriscasEnv` receives an adapter instance -- it never knows how the engine is accessed
- Error boundary: adapter catches raw HTTP/connection errors → raises `EngineConnectionError`

**Boundary 2: Environment ↔ Training/Evaluation**
- Training and evaluation interact ONLY through `BriscasEnv`'s Gymnasium interface (`reset`, `step`, `observation_space`, `action_space`)
- No direct access to adapter, engine state, or internal env state

**Boundary 3: Scripts ↔ Library Code**
- `scripts/` contains CLI entry points (argparse, logging setup, seed initialization)
- `training/` and `evaluation/` contain reusable functions with no CLI dependencies
- This separation allows importing training/evaluation functions in tests or notebooks

### Data Flow

```
Engine (REST API)
    ↕ (HTTP requests)
EngineAdapter
    ↕ (game state dicts)
BriscasEnv._get_observation()
    ↕ (numpy arrays)
SB3 DQN Agent
    ↕ (action integers)
BriscasEnv.step() → action % len(hand)
    ↕ (mapped action)
EngineAdapter → Engine
```

## Architecture Validation Results

### Coherence Validation

- Decision compatibility: All technology choices (Python 3.10+, PyTorch, SB3, Gymnasium, DQN) are proven compatible
- Pattern consistency: Python naming conventions, adapter pattern, and scripts-not-services principle are internally consistent
- Structure alignment: Directory structure directly maps to architectural boundaries and decisions

### Requirements Coverage

All 17 functional requirements mapped to specific architectural components. No orphan FRs. All implicit NFRs (training speed, model persistence, reproducibility) addressed through architectural decisions and patterns.

| FR | Covered | Location |
|---|---|---|
| FR1-FR5 | Yes | `gym_env/briscas_env.py`, `gym_env/engine_adapter.py` |
| FR6-FR11 | Yes | `training/train.py`, `scripts/train.py` |
| FR12-FR15, FR18 | Yes | `evaluation/evaluate.py`, `scripts/evaluate.py` |
| FR16-FR17 | Yes | `seed.py` |

### Implementation Readiness

- All critical decisions documented with clear rationale
- Implementation patterns are specific enough to prevent AI agent conflicts
- Project structure is complete with file-level detail
- Enforcement guidelines are explicit and testable

### Architecture Completeness Checklist

- [x] Project context analyzed and validated
- [x] Technology stack specified (Python 3.10+, PyTorch, SB3, Gymnasium)
- [x] Core decisions made (state representation, action masking, reward signal, config management)
- [x] Observation vector structure defined with section ordering
- [x] Implementation patterns defined (naming, structure, env contract, training, evaluation, error handling, seeds)
- [x] Project structure complete with FR mapping
- [x] Architectural boundaries defined (engine↔env, env↔training, scripts↔library)
- [x] Data flow documented
- [x] Checkpointing strategy included
- [x] Random opponent behavior specified
- [x] CSV output naming convention defined

### Architecture Readiness Assessment

**Overall Status:** READY FOR IMPLEMENTATION

**Confidence Level:** High

**Key Strengths:**
- Simple, flat architecture matching project complexity
- Clean adapter boundary enabling REST→direct-call migration
- Single observation encoding source of truth preventing train/eval drift
- Parameterized agent design preventing code duplication
- Observation vector ordering locked down to prevent implementation drift

**Areas for Future Enhancement:**
- Per-trick reward strategy (Phase 2)
- TensorBoard training curves (Phase 2)
- Algorithm comparison infrastructure (Phase 3)

### Implementation Handoff

**First implementation priority:** Build the `gym_env/` package -- `EngineAdapter` first (verify engine connectivity), then `BriscasEnv` (observation, step, reset, reward). Everything else depends on a working environment wrapper.
