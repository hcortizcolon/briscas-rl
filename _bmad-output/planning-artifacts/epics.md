---
stepsCompleted:
  - step-01-validate-prerequisites
  - step-02-design-epics
  - step-03-create-stories
  - step-04-final-validation
status: complete
completedAt: '2026-02-28'
inputDocuments:
  - prd.md
  - architecture.md
---

# briscas_rl - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for briscas_rl, decomposing the requirements from the PRD and Architecture into implementable stories.

## Requirements Inventory

### Functional Requirements

FR1: System can connect to the existing Briscas game engine via local REST API
FR2: System can translate game engine state into a fixed-size numerical observation vector (hand, trump, current trick, play history)
FR3: System can translate agent action (card index) into a valid game engine move
FR4: System can receive end-of-game point totals as reward signal from the engine
FR5: System can constrain agent actions to only legal moves given current game state
FR6: Researcher can train a best agent (point maximizer) against the engine's random player
FR7: Researcher can train a worst agent (point minimizer) against the engine's random player
FR8: Researcher can monitor training progress by inspecting win rate over episodes
FR9: Researcher can save a trained agent model to disk
FR10: Researcher can load a previously trained agent model from disk
FR11: System can output training summary statistics upon completion (final win rate, episodes trained)
FR12: Researcher can run N evaluation games between any two agents (best, worst, or engine's random)
FR13: Researcher can configure the number of evaluation games per matchup
FR14: System can alternate which agent plays first across evaluation games
FR15: System can record per-game results to CSV (both players' points, which agent went first, point differential)
FR16: Researcher can set random seeds for training and evaluation runs
FR17: System can produce reproducible results given the same seed
FR18: Researcher can view win/loss/draw rates and point differentials after running an evaluation matchup

### NonFunctional Requirements

NFR1: Environment must handle game engine API errors gracefully — retry or fail with clear error message, never silently corrupt training data

### Additional Requirements

- Action masking via `action % len(hand)` in env `step()` — no special masking library
- Fixed-size observation vector with defined section ordering: [hand(3), trump card(1), trump suit(1), trick cards(2), cards seen per suit(4), agent points(1), opponent points(1)]
- Reward normalization: point differential / 120 for [-1, +1] range; worst agent negates this
- Adapter pattern: `EngineAdapter` is sole REST API touchpoint; swappable to direct Python calls
- Checkpoint saving via SB3's `CheckpointCallback` during training
- Model metadata JSON saved alongside each `.zip` model file
- Seed propagation through 4 sources: NumPy, PyTorch, SB3, game engine
- Error boundary: adapter catches raw HTTP errors → raises `EngineConnectionError`
- CSV output naming: `{agent1}_vs_{agent2}_{num_games}g_{seed}s.csv`
- Training config: hardcoded defaults with CLI overrides (no config files)

### FR Coverage Map

| FR | Epic | Description |
|---|---|---|
| FR1 | Epic 1 | Connect to game engine via REST API |
| FR2 | Epic 1 | Translate state to observation vector |
| FR3 | Epic 1 | Translate action to game move |
| FR4 | Epic 1 | Receive end-of-game reward |
| FR5 | Epic 1 | Constrain to legal moves |
| FR6 | Epic 2 | Train best agent |
| FR7 | Epic 2 | Train worst agent |
| FR8 | Epic 2 | Monitor training progress |
| FR9 | Epic 2 | Save trained model |
| FR10 | Epic 2 | Load trained model |
| FR11 | Epic 2 | Training summary stats |
| FR12 | Epic 3 | Run N evaluation games |
| FR13 | Epic 3 | Configure game count |
| FR14 | Epic 3 | Alternate first player |
| FR15 | Epic 3 | Record results to CSV |
| FR16 | Epic 1 | Set random seeds |
| FR17 | Epic 1 | Reproducible results |
| FR18 | Epic 3 | View win rates and differentials |
| NFR1 | Epic 1 | Graceful API error handling |

## Epic List

### Epic 1: Play Briscas Programmatically
Researcher can connect to the game engine, observe game state, take actions, and receive rewards through a standard Gymnasium interface with reproducible seeding — the foundation everything else depends on.
**FRs covered:** FR1, FR2, FR3, FR4, FR5, FR16, FR17, NFR1
**Notes:** Seed propagation utility (`seed.py`) is foundational — used by both training and evaluation. Adapter pattern enables REST-to-direct-call migration if API bottleneck surfaces during training.

### Epic 2: Train Best and Worst Agents
Researcher can train a point-maximizing agent and a point-minimizing agent against random play, monitor convergence, and save/load trained models for later use.
**FRs covered:** FR6, FR7, FR8, FR9, FR10, FR11
**Notes:** Worst agent validation gate — verify worst agent's win rate vs random is meaningfully below 50% before proceeding to evaluation. This is a story-level validation step.

### Epic 3: Evaluate Matchups and Analyze Results
Researcher can pit any two agents against each other over thousands of games, collect per-game results to CSV, and view summary statistics to answer the skill-vs-luck question.
**FRs covered:** FR12, FR13, FR14, FR15, FR18

## Epic 1: Play Briscas Programmatically

Researcher can connect to the game engine, observe game state, take actions, and receive rewards through a standard Gymnasium interface with reproducible seeding — the foundation everything else depends on.

### Story 1.1: Connect to Game Engine via Adapter

As a researcher,
I want to connect to the Briscas game engine through a clean adapter interface,
So that I can programmatically start games, play cards, and receive game state without coupling to the REST API directly.

**Acceptance Criteria:**

**Given** the game engine is running locally
**When** the adapter connects to the engine
**Then** the adapter can start a new game and receive initial game state
**And** the adapter can submit a card play action and receive updated state
**And** the adapter can retrieve end-of-game point totals for both players

**Given** the game engine is not running or unreachable
**When** the adapter attempts to connect
**Then** an `EngineConnectionError` is raised with a clear error message
**And** no silent failures or corrupted state occur

**Given** the adapter pattern from Architecture
**When** implementing engine communication
**Then** `EngineAdapter` defines the base interface and `RESTAdapter` implements it
**And** all REST API calls are isolated within `gym_env/engine_adapter.py`
**And** no code outside `gym_env/` directly calls the engine API

### Story 1.2: Gymnasium Environment with Observation and Actions

As a researcher,
I want a standard Gymnasium environment that translates game state into numerical observations and my actions into valid moves,
So that I can use any Gymnasium-compatible RL library to train agents on Briscas.

**Acceptance Criteria:**

**Given** a new game is started via `env.reset()`
**When** the environment returns the initial observation
**Then** the observation is a fixed-size numerical vector following the defined ordering: [hand(3), trump card(1), trump suit(1), trick cards(2), cards seen per suit(4), agent points(1), opponent points(1)]
**And** hand cards are sorted by suit+rank for consistent representation
**And** empty slots are padded with -1

**Given** the agent selects an action (0, 1, or 2)
**When** `env.step(action)` is called
**Then** the action is mapped via `action % len(hand)` to handle hands smaller than 3
**And** the mapped action is translated to a valid game engine move via the adapter
**And** the environment returns (observation, reward, terminated, truncated, info)

**Given** a game is in progress
**When** intermediate steps are taken (not end-of-game)
**Then** reward is 0

**Given** a game reaches its final trick
**When** the last step completes
**Then** reward is the normalized point differential: (agent_points - opponent_points) / 120
**And** terminated is True

**Given** observation encoding
**When** any component needs the current observation
**Then** it is produced by a single `_get_observation()` method on `BriscasEnv`
**And** this method is the sole source of truth for observation encoding

### Story 1.3: Reproducible Seeding

As a researcher,
I want to set a single seed that propagates to all random sources,
So that my training and evaluation runs are reproducible.

**Acceptance Criteria:**

**Given** a seed value (e.g., 42)
**When** `set_all_seeds(seed)` is called from `seed.py`
**Then** `numpy.random.seed(seed)` is set
**And** `torch.manual_seed(seed)` is set
**And** SB3's `set_random_seed(seed)` is called
**And** the game engine's seed/shuffle endpoint is called (if available)

**Given** `set_all_seeds(42)` is called
**When** checking random state immediately after
**Then** `numpy.random.random()` produces the same value across separate invocations with the same seed

**Given** the same seed is used for two separate runs
**When** the same operations are performed
**Then** results are identical across both runs

## Epic 2: Train Best and Worst Agents

Researcher can train a point-maximizing agent and a point-minimizing agent against random play, monitor convergence, and save/load trained models for later use.

### Story 2.1: Train Best Agent with DQN

**Note:** This is the largest story in the project — covers training harness, CLI entry point, checkpointing, metadata, and summary output. Dev agent should plan for this accordingly.

As a researcher,
I want to train a point-maximizing agent against the engine's random player using DQN,
So that I can produce an agent that plays Briscas as well as possible given observable information.

**Acceptance Criteria:**

**Given** a working `BriscasEnv` and seed value
**When** `train_agent(agent_type="best", episodes=N, seed=S, output_path=P)` is called
**Then** SB3's DQN is trained against the engine's built-in random player
**And** the reward signal is the normalized point differential from `BriscasEnv`
**And** training uses hardcoded defaults with CLI overrides via `scripts/train.py`

**Given** training is in progress
**When** episodes complete
**Then** win rate over recent episodes is logged to stdout via SB3's built-in logging
**And** periodic checkpoint snapshots are saved to `models/checkpoints/` via SB3's `CheckpointCallback`

**Given** training completes
**When** the final model is saved
**Then** the model is saved as an SB3 `.zip` file to the specified output path (e.g., `models/best_agent_50k.zip`)
**And** a metadata JSON is saved alongside it containing: seed, episodes, reward_type, timestamp
**And** a training summary is printed to stdout: final win rate, total episodes trained

**Given** the CLI entry point `scripts/train.py`
**When** invoked with `--agent best --episodes 50000 --seed 42`
**Then** seeds are propagated via `set_all_seeds()`, training runs, and model is saved to default output path

### Story 2.2: Train and Validate Worst Agent

As a researcher,
I want to train a point-minimizing agent with negated reward and validate it performs worse than random,
So that I have a confirmed anti-optimal agent for the skill-vs-luck comparison.

**Acceptance Criteria:**

**Given** the same training harness from Story 2.1
**When** `train_agent(agent_type="worst", ...)` is called
**Then** the only difference is reward is negated: `reward * -1`
**And** all other training parameters, checkpointing, and save behavior are identical

**Given** training completes for the worst agent
**When** a quick validation evaluation runs against random (e.g., 1000 games)
**Then** the worst agent's win rate vs random is reported
**And** if win rate is not meaningfully below 50%, a warning is logged: "Worst agent may not be producing true anti-optimal play"

**Given** the CLI entry point
**When** invoked with `--agent worst`
**Then** training runs with negated reward and validation is performed automatically upon completion

### Story 2.3: Load and Resume Trained Models

As a researcher,
I want to load a previously saved agent model from disk,
So that I can evaluate it or continue training without retraining from scratch.

**Acceptance Criteria:**

**Given** a saved SB3 model `.zip` file exists at a known path
**When** the model is loaded via `DQN.load(path)`
**Then** the agent can be used for evaluation via `.predict(observation)` to select actions
**And** the loaded model can be passed back to `.learn()` to continue training from its saved state (SB3 native capability — no custom resume logic needed)

**Given** a model path that does not exist
**When** loading is attempted
**Then** a clear error message is raised indicating the file was not found

**Given** the metadata JSON alongside the model
**When** a model is loaded
**Then** the metadata (seed, episodes, reward_type, timestamp) is available for logging and verification

## Epic 3: Evaluate Matchups and Analyze Results

Researcher can pit any two agents against each other over thousands of games, collect per-game results to CSV, and view summary statistics to answer the skill-vs-luck question.

### Story 3.1: Run Evaluation Matchups

As a researcher,
I want to run N games between any two agents with alternating first player and record results to CSV,
So that I can collect the raw data needed to compare agent performance.

**Acceptance Criteria:**

**Given** two agents (any combination of trained model path or `"random"` for engine's built-in random player)
**When** `run_evaluation(agent1, agent2, num_games=N, seed=S)` is called
**Then** N games are played between the two agents
**And** first player alternates every game: `first_player = game_id % 2`

**Given** `"random"` is specified as an agent
**When** evaluation runs
**Then** the game engine's built-in random player is used — no separate random policy implemented

**Given** evaluation uses the same environment as training
**When** games are played
**Then** `BriscasEnv` is instantiated with the same observation encoding and action masking used during training

**Given** evaluation completes
**When** results are written
**Then** a CSV is created with columns: `game_id, agent1_points, agent2_points, first_player, point_differential`
**And** the default filename follows the convention: `{agent1}_vs_{agent2}_{num_games}g_{seed}s.csv`
**And** the file is saved to `results/`

**Given** the CLI entry point `scripts/evaluate.py`
**When** invoked with `--agent1 models/best.zip --agent2 random --games 10000 --seed 42`
**Then** seeds are propagated via `set_all_seeds()`, evaluation runs, and CSV is written
**And** an optional `--output` flag overrides the default filename

### Story 3.2: Summary Statistics and Analysis

As a researcher,
I want to see win rates, point differentials, and outcome variance after an evaluation run,
So that I can answer the skill-vs-luck question without manually analyzing the CSV.

**Acceptance Criteria:**

**Given** an evaluation run has completed
**When** summary statistics are computed
**Then** the following are printed to stdout:
- Win/loss/draw count and percentages for each agent (draw = equal points, i.e., agent1_points == agent2_points)
- Average point differential
- Point differential sample standard deviation (outcome variance)
- Upset percentage: percentage of games where the lower-rated agent wins (the key metric from the PRD measuring how often favorable deals overcome skill)

**Given** multiple matchups are run (best vs random, best vs worst, worst vs random)
**When** each evaluation completes
**Then** each prints its own summary, enabling side-by-side comparison of matchup statistics
