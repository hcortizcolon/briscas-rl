# Story 1.2: Gymnasium Environment with Observation and Actions

Status: done

## Story

As a researcher,
I want a standard Gymnasium environment that translates game state into numerical observations and my actions into valid moves,
So that I can use any Gymnasium-compatible RL library to train agents on Briscas.

## Acceptance Criteria

1. **Given** a new game is started via `env.reset()`, **When** the environment returns the initial observation, **Then** the observation is a fixed-size numpy array of 13 float32 values following the defined ordering: [hand(3), trump_card(1), trump_suit(1), trick_cards(2), cards_seen_per_suit(4), agent_points(1), opponent_points(1)] **And** hand cards are encoded as contiguous card IDs (0-39) sorted by suit+rank **And** empty slots are padded with -1 (AC: FR2)

2. **Given** the agent selects an action (0, 1, or 2), **When** `env.step(action)` is called, **Then** the action is mapped via `action % len(hand)` to handle hands smaller than 3 **And** the mapped action is translated to a valid game engine move via the adapter **And** the environment returns `(observation, reward, terminated, truncated, info)` (AC: FR3, FR5)

3. **Given** a game is in progress, **When** intermediate steps are taken (not end-of-game), **Then** reward is 0.0 (AC: FR4)

4. **Given** a game reaches its final trick, **When** the last step completes, **Then** reward is the normalized point differential: `(agent_points - opponent_points) / 120` **And** terminated is True **And** truncated is always False (AC: FR4)

5. **Given** observation encoding, **When** any component needs the current observation, **Then** it is produced by a single `_get_observation()` method on `BriscasEnv` **And** this method is the sole source of truth for observation encoding (AC: FR2, Architecture Enforcement)

6. **Given** the Gymnasium 1.2.x API, **When** `reset(seed=None, options=None)` is called, **Then** it returns `(observation, info)` **And** calls `super().reset(seed=seed)` for proper seeding **And** starts a new game via the adapter **And** clears all internal tracking state (cards_seen set) (AC: FR2)

## Tasks / Subtasks

- [x] Task 1: Create `gym_env/observation.py` — card encoding constants and observation space definition (AC: #1, #5)
  - [x] Define suit-to-index mapping: Oros=0, Copas=1, Espadas=2, Bastos=3
  - [x] Define rank-to-index mapping: [1,2,3,4,5,6,7,10,11,12] → [0..9]
  - [x] Define `encode_card(card: Card) -> int` returning `suit_index * 10 + rank_index` (range 0-39)
  - [x] Define `OBSERVATION_SIZE = 13` constant
  - [x] Define `build_observation_space()` returning `gymnasium.spaces.Box` with per-element low/high bounds and dtype=np.float32
- [x] Task 2: Create `gym_env/briscas_env.py` — `BriscasEnv(gymnasium.Env)` class (AC: #1-6)
  - [x] `__init__(self, adapter: EngineAdapter)` — store adapter, define `observation_space` and `action_space = gymnasium.spaces.Discrete(3)`
  - [x] `reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]` — if `_game_active`, call `adapter.delete_game()` first to clean up previous game. Then call `super().reset(seed=seed)`, call `adapter.new_game()`, set `_game_active=True`, clear `_cards_seen` set, return `(_get_observation(), {})`
  - [x] `step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]` — apply `action % len(hand)` masking, call `adapter.play_card()`, then loop `adapter.process_ai_turn()` until `is_your_turn=True` or `game_over=True` (handles opponent leading after winning a trick), update `_cards_seen` from trick cards, compute reward, return 5-tuple with `info={}`
  - [x] `_execute_turn(self, card_index: int) -> GameState` — private method encapsulating the turn-processing loop: `play_card(card_index)` → loop `process_ai_turn()` until `is_your_turn` or `game_over`. Returns final GameState. Called by `step()`.
  - [x] `_get_observation(self) -> np.ndarray` — single source of truth for observation encoding, builds 13-element float32 array from current GameState + internal tracking state
  - [x] `_update_cards_seen(self, state: GameState)` — add any trick cards to `_cards_seen` set, derive per-suit counts when building observation
- [x] Task 3: Update `gym_env/__init__.py` — export `BriscasEnv` and observation utilities (AC: #5)
- [x] Task 4: Write tests `tests/test_briscas_env.py` (AC: #1-6)
  - [x] Test observation shape is (13,) and dtype is float32
  - [x] Test observation values match expected encoding for known game state
  - [x] Test hand sorting produces consistent encoding regardless of input order
  - [x] Test hand padding with -1 for hands smaller than 3
  - [x] Test trick slot ordering: slot 0 = lead card, slot 1 = response card, -1 when empty
  - [x] Test action masking: action % len(hand) for hand sizes 1, 2, 3
  - [x] Test reward is 0.0 for intermediate steps
  - [x] Test reward is normalized point differential at game end
  - [x] Test terminated=True at game over, truncated=False always
  - [x] Test reset() clears cards_seen and returns fresh observation
  - [x] Test reset() returns (observation, info) tuple
  - [x] Test reset() calls delete_game() when a game is already active
  - [x] Test env.observation_space contains observation from reset(), env.action_space is Discrete(3)
  - [x] Test full game loop: mock multi-turn game sequence, verify observation updates each step, reward=0 until final, terminal reward=normalized differential, terminated=True at end
  - [x] Test step() handles opponent leading after winning trick (process_ai_turn called before agent plays)
- [x] Task 5: Write tests `tests/test_observation.py` (AC: #1)
  - [x] Test encode_card produces correct card IDs for known cards
  - [x] Test suit and rank index mappings are complete and correct
  - [x] Test observation_space bounds match expected low/high per element
  - [x] Test cards_seen_per_suit counts derived correctly from card ID set

## Dev Notes

### Technical Requirements

- **Gymnasium 1.2.x API compliance:** `reset()` returns `(observation, info)`, `step()` returns `(observation, reward, terminated, truncated, info)`. Must call `super().reset(seed=seed)` in reset.
- **SB3 DQN compatibility:** SB3 wraps envs in `DummyVecEnv` which passes `seed` argument to `reset()`. The env MUST accept `seed` and `options` keyword args.
- **Observation dtype:** `np.float32` — SB3's DQN expects float32 observations. Do NOT use int or float64.
- **Reward range:** [-1.0, +1.0] via `(agent_points - opponent_points) / 120`. The `TOTAL_POINTS = 120` constant should be defined in observation.py.
- **Action space:** `gymnasium.spaces.Discrete(3)` — always 3, even when hand has fewer cards. Masking via `action % len(hand)` in `step()`.
- **No external state leakage:** The env must not expose adapter, internal GameState, or `_cards_seen` publicly. All external interaction through Gymnasium interface only.

### Architecture Compliance

- **Boundary 2 (Environment ↔ Training/Evaluation):** Training and evaluation interact ONLY through `BriscasEnv`'s Gymnasium interface (`reset`, `step`, `observation_space`, `action_space`). No direct access to adapter or internal state. [Source: architecture.md#Architectural Boundaries]
- **Environment Wrapper Contract:** Must inherit `gymnasium.Env`, define `observation_space` and `action_space` in `__init__`, engine communication goes through adapter object only, `step()` handles action masking, `reset()` starts new game, reward only at game end, observation via single `_get_observation()` method. [Source: architecture.md#Environment Wrapper Contract]
- **Adapter pattern:** `BriscasEnv` receives an `EngineAdapter` instance in `__init__`. It never knows whether the adapter uses REST or direct calls. [Source: architecture.md#Boundary 1]
- **Error handling:** The env does NOT catch `EngineConnectionError`. Let it propagate per architecture rules. Only the adapter wraps raw errors. [Source: architecture.md#Error Handling]
- **Naming conventions:** snake_case for files/functions (`briscas_env.py`, `_get_observation`), PascalCase for classes (`BriscasEnv`), UPPER_SNAKE for constants (`OBSERVATION_SIZE`, `TOTAL_POINTS`). [Source: architecture.md#Code Naming Conventions]
- **Logging:** Use `logging` module, not print statements. [Source: architecture.md#Error Handling]

### Library / Framework Requirements

| Library | Version | Purpose | Notes |
|---|---|---|---|
| `gymnasium` | >=1.2.0,<2.0 | Env base class, spaces | Already in requirements.txt |
| `numpy` | (transitive via gymnasium) | Observation arrays | Use `np.float32` dtype |
| `requests` | latest stable | Used by adapter (Story 1.1) | No new dependency |
| `pytest` | latest stable | Testing | Already in requirements.txt |

**No new dependencies needed** — gymnasium and numpy are already pinned in requirements.txt from Story 1.1.

**Gymnasium 1.2.x critical notes:**
- `reset()` signature: `reset(self, seed=None, options=None) -> tuple[ObsType, dict]`
- `step()` signature: `step(self, action) -> tuple[ObsType, float, bool, bool, dict]`
- `terminated` vs `truncated`: terminated=True when game ends naturally, truncated=False always (no time limits in Brisca)
- `super().reset(seed=seed)` initializes `self.np_random` — the Gymnasium-managed RNG. Use this if env-internal randomness is ever needed.

### Project Structure Notes

Files to create/modify in this story:
```
briscas_rl/
├── gym_env/
│   ├── __init__.py           # MODIFY — add BriscasEnv, observation exports
│   ├── engine_adapter.py     # NO CHANGE — Story 1.1 (adapter + dataclasses)
│   ├── observation.py        # NEW — card encoding, observation space, constants
│   └── briscas_env.py        # NEW — BriscasEnv(gymnasium.Env)
├── tests/
│   ├── test_engine_adapter.py # NO CHANGE — Story 1.1
│   ├── test_briscas_env.py    # NEW — env wrapper tests
│   └── test_observation.py    # NEW — encoding utility tests
└── requirements.txt           # NO CHANGE — dependencies already pinned
```

Alignment with architecture.md project structure: exact match. [Source: architecture.md#Complete Project Directory Structure]

### Testing Standards

- Use `pytest` as testing framework [Source: architecture.md#Development Tooling]
- **Mock the adapter, NOT HTTP calls.** Story 1.1 mocked HTTP responses directly. This story should mock `EngineAdapter` methods (`new_game`, `play_card`, `process_ai_turn`, `get_state`, `delete_game`) returning `GameState` dataclass instances. This tests the env in isolation from the adapter.
- Test file mirrors source: `tests/test_briscas_env.py` ↔ `gym_env/briscas_env.py`, `tests/test_observation.py` ↔ `gym_env/observation.py` [Source: architecture.md#Structure Patterns]
- All Story 1.1 tests (23 tests) MUST continue passing — do not modify `engine_adapter.py` or its tests
- Use `unittest.mock.MagicMock` or `unittest.mock.create_autospec(EngineAdapter)` to mock the adapter
- Build `GameState` fixtures using the frozen dataclasses from Story 1.1 (`Card`, `TrickCard`, `PlayerInfo`, `GameState`)

### Previous Story Intelligence (Story 1.1)

**What was built:**
- `EngineAdapter` ABC + `RESTAdapter` in `gym_env/engine_adapter.py`
- Frozen dataclasses: `Card(rank, suit, suit_symbol, display_name, points)`, `TrickCard(player, card)`, `PlayerInfo(name, is_current, is_human, score, hand_size)`, `GameState(hand, trump, trick, players, deck_remaining, round_number, game_over, winner, is_your_turn)`
- `EngineConnectionError` custom exception
- 23 unit tests all passing

**Patterns established:**
- `logging.getLogger(__name__)` for logger setup
- Frozen dataclasses for immutable data objects
- `__init__.py` exports all public types via `__all__`
- Module-level private helper functions (`_parse_card`, `_parse_game_state`)
- `requests.Session()` for cookie persistence (adapter internals)

**Key data structures this story depends on:**
- `Card.rank` is raw engine value (1-7, 10-12) — observation.py must map these to contiguous indices
- `Card.suit` is a string ("Oros", "Copas", "Espadas", "Bastos") — observation.py must map to 0-3
- `GameState.hand` is `list[Card]` — variable length 0-3
- `GameState.trick` is `list[TrickCard]` — variable length 0-2, ordered by play sequence
- `GameState.players` is `list[PlayerInfo]` — always 2 entries; agent is `is_human=True`, opponent is `is_human=False`
- `GameState.game_over` signals terminal state
- `PlayerInfo.score` gives current point totals

**Critical game engine behaviors from Story 1.1 dev notes:**
- Turn flow: human → AI → human. After `play_card()`, must call `process_ai_turn()` to let opponent play.
- Trick resolution happens AFTER state serialization — the state returned from `play_card()` shows trick cards but hasn't resolved the winner yet.
- `ai_difficulty="basic"` plays first card in hand (closest to random available).
- Card point values: 1→11, 3→10, 12→4, 11→3, 10→2, rest→0. Total 120 points in deck.

### Git Intelligence

**Recent commits (3 total):**
1. `98b8684` — Story 1.1 implementation: EngineAdapter, RESTAdapter, dataclasses, 23 tests, requirements.txt
2. `719fda7` — Planning artifacts: story docs, sprint status, epics, implementation readiness
3. `366f6ff` — First commit

**Files established in codebase:**
- `gym_env/__init__.py`, `gym_env/engine_adapter.py` — DO NOT MODIFY
- `tests/__init__.py`, `tests/test_engine_adapter.py` — DO NOT MODIFY
- `requirements.txt` — NO CHANGES NEEDED

**Code conventions observed:**
- No type stubs or `.pyi` files — inline type hints only
- No docstring style enforced but Story 1.1 uses single-line `"""..."""` docstrings
- `__pycache__/` directories present — `.gitignore` may need updating (not this story's concern)

### Latest Tech Information

**Gymnasium 1.2.x (current stable):**
- `reset()` accepts `seed` and `options` kwargs, returns `(observation, info)` tuple
- `step()` returns 5-tuple: `(observation, reward, terminated, truncated, info)`
- `terminated` = task completed/failed (MDP terminal). `truncated` = episode ended by external boundary (time limit). For Brisca: terminated=True at game end, truncated=False always.
- `super().reset(seed=seed)` initializes `self.np_random` (numpy Generator-based RNG)
- `render_mode` must be set at init time, not dynamically. For this story: no rendering needed, skip render_mode.

**SB3 2.7.x compatibility:**
- `DummyVecEnv` passes `seed` kwarg to `reset()` — env MUST accept it
- DQN expects `observation_space` to be a `Box` with float32 dtype
- DQN expects `action_space` to be `Discrete`
- `check_env()` from `stable_baselines3.common.env_checker` can validate env compliance — useful for dev agent to run as a sanity check (not a required test, but recommended)

### Card Encoding ADRs (from Advanced Elicitation)

**ADR-1: Card Numeric Encoding**
- Contiguous mapping. Suits: Oros=0, Copas=1, Espadas=2, Bastos=3. Ranks: [1,2,3,4,5,6,7,10,11,12] → indices [0..9].
- Card ID = `suit_index * 10 + rank_index`, giving range 0-39. Empty slots = -1.

**ADR-2: Observation Space Bounds**
```
Index:  [0,  1,  2,   3,  4,  5,  6,  7,  8,  9, 10,  11,  12]
Field:  [h0, h1, h2, trp, ts, t0, t1, s0, s1, s2, s3, apt, opt]
Low:    [-1, -1, -1,   0,  0, -1, -1,  0,  0,  0,  0,   0,   0]
High:   [39, 39, 39,  39,  3, 39, 39, 10, 10, 10, 10, 120, 120]
```
- h0-h2: hand card IDs (sorted, -1 padded)
- trp: trump card ID
- ts: trump suit index (0-3)
- t0-t1: trick card IDs (slot 0=lead, slot 1=response, -1 when empty)
- s0-s3: cards seen per suit (Oros, Copas, Espadas, Bastos)
- apt: agent cumulative points
- opt: opponent cumulative points
- dtype: np.float32

**ADR-3: Hand Sorting**
- Sort by card ID ascending (suit_index * 10 + rank_index). Groups by suit first, then by rank within suit.

### Observation Stateful Tracking (from Advanced Elicitation)

- **`_cards_seen`**: `set[int]` of card IDs seen in completed tricks. Reset on `reset()`. Updated by `_update_cards_seen()` after each state change.
- **Scope:** Tracks cards played in completed tricks ONLY. Does NOT include cards in agent's hand or the trump card (those are already separate observation components).
- **Trick slot ordering:** Slot 0 = first card played in trick (lead card), Slot 1 = second card (response). When agent acts and trick is empty, agent is leading. When slot 0 is occupied, agent is responding.
- **Update strategy:** Maintain seen card set. After each state, add any trick card IDs not already in the set. Derive per-suit counts by iterating the set: `count_for_suit_i = sum(1 for cid in _cards_seen if cid // 10 == suit_index)`.

### References

- [Source: architecture.md#Environment Wrapper Contract]
- [Source: architecture.md#Architectural Boundaries — Boundary 2: Environment ↔ Training/Evaluation]
- [Source: architecture.md#Core Architectural Decisions — State Representation, Action Space, Reward Signal]
- [Source: architecture.md#Implementation Patterns — Naming, Error Handling, Seed Propagation]
- [Source: architecture.md#Complete Project Directory Structure]
- [Source: epics.md#Story 1.2 — Acceptance criteria and BDD scenarios]
- [Source: prd.md#Environment Integration — FR2, FR3, FR4, FR5]
- [Source: prd.md#Additional Requirements — Observation vector ordering, action masking, reward normalization]
- [Source: 1-1-connect-to-game-engine-via-adapter.md — Dataclass definitions, game loop, engine behaviors]
- [Source: lets-play-brisca/app/core/models/card.py — Rank values, point values]
- [Source: lets-play-brisca/app/core/models/suit.py — Suit enum: Oros, Copas, Espadas, Bastos]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

None required.

### Completion Notes List

- Task 1: Created `gym_env/observation.py` with `SUIT_INDEX`, `RANK_INDEX`, `encode_card()`, `OBSERVATION_SIZE=13`, `TOTAL_POINTS=120`, and `build_observation_space()` with per-element Box bounds.
- Task 2: Created `gym_env/briscas_env.py` with `BriscasEnv(gymnasium.Env)`. Implements `__init__`, `reset`, `step`, `_execute_turn`, `_get_observation`, `_update_cards_seen`, `_compute_reward`. Follows Gymnasium 1.2.x API, action masking via `action % len(hand)`, reward only at terminal state, `_cards_seen` tracking.
- Task 3: Updated `gym_env/__init__.py` to export `BriscasEnv`, `OBSERVATION_SIZE`, `TOTAL_POINTS`, `build_observation_space`, `encode_card`.
- Task 4: Created `tests/test_briscas_env.py` — 29 tests covering observation shape/dtype/values, hand sorting/padding, trick slots, action masking, reward (intermediate/terminal), terminated/truncated, reset behavior, spaces, full game loop, opponent lead handling. All mocking at EngineAdapter level.
- Task 5: Created `tests/test_observation.py` — 19 tests covering encode_card, suit/rank mappings, observation space bounds, cards_seen_per_suit derivation, constants.
- Refactored `_get_observation` cards_seen counting to avoid off-by-one from np.full(-1) initialization.

### Change Log

- 2026-03-01: Implemented Story 1.2 — Gymnasium environment with observation and actions. 48 new tests (71 total), 3 new files, 1 modified file.
- 2026-03-01: Code review fixes — Fixed _cards_seen bug (intermediate states lost when opponent wins trick), added empty-hand guard in step(), added 2 new tests, created .gitignore, fixed sprint-status.yaml duplicate metadata. 73 tests total.

### File List

- gym_env/observation.py (NEW)
- gym_env/briscas_env.py (NEW)
- gym_env/__init__.py (MODIFIED)
- tests/test_observation.py (NEW)
- tests/test_briscas_env.py (NEW)
- .gitignore (NEW)
