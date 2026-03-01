# Story 1.1: Connect to Game Engine via Adapter

Status: ready-for-dev

## Story

As a researcher,
I want to connect to the Briscas game engine through a clean adapter interface,
So that I can programmatically start games, play cards, and receive game state without coupling to the REST API directly.

## Acceptance Criteria

1. **Given** the game engine is running locally, **When** the adapter connects to the engine, **Then** the adapter can start a new game and receive initial game state **And** the adapter can submit a card play action and receive updated state **And** the adapter can retrieve end-of-game point totals for both players (AC: FR1, FR3, FR4)

2. **Given** the game engine is not running or unreachable, **When** the adapter attempts to connect, **Then** an `EngineConnectionError` is raised with a clear error message **And** no silent failures or corrupted state occur (AC: NFR1)

3. **Given** the adapter pattern from Architecture, **When** implementing engine communication, **Then** `EngineAdapter` defines the base interface and `RESTAdapter` implements it **And** all REST API calls are isolated within `gym_env/engine_adapter.py` **And** no code outside `gym_env/` directly calls the engine API (AC: FR1, Architecture Boundary 1)

## Tasks / Subtasks

- [ ] Task 1: Create `gym_env/` package structure (AC: #3)
  - [ ] Create `gym_env/__init__.py`
  - [ ] Create `gym_env/engine_adapter.py`
- [ ] Task 2: Implement `EngineAdapter` abstract base class (AC: #3)
  - [ ] Define abstract methods: `new_game()`, `play_card(card_index)`, `process_ai_turn()`, `get_state()`, `delete_game()`
  - [ ] Define `EngineConnectionError` custom exception
  - [ ] Define return types/dataclasses for game state responses
- [ ] Task 3: Implement `RESTAdapter(EngineAdapter)` (AC: #1, #2)
  - [ ] Implement `__init__(self, base_url)` with configurable engine URL (default `http://localhost:5000`)
  - [ ] Implement `new_game()` → POST `/api/game/new` with `player_name` and `ai_difficulty="basic"` (random-equivalent)
  - [ ] Implement `play_card(card_index)` → POST `/api/game/play` with `card_index`
  - [ ] Implement `process_ai_turn()` → POST `/api/game/process-ai`
  - [ ] Implement `get_state()` → GET `/api/game/state`
  - [ ] Implement `delete_game()` → DELETE `/api/game/delete`
  - [ ] Handle HTTP session cookies (game_id stored server-side in Flask session)
  - [ ] Wrap all `requests` exceptions into `EngineConnectionError`
- [ ] Task 4: Write tests (AC: #1, #2, #3)
  - [ ] `tests/test_engine_adapter.py` — unit tests with mocked HTTP responses
  - [ ] Test successful game lifecycle: new_game → play_card → get_state → game_over
  - [ ] Test `EngineConnectionError` raised on connection failure
  - [ ] Test `EngineConnectionError` raised on non-200 responses
  - [ ] Test adapter interface contract (abstract methods defined)

## Dev Notes

### Critical Architecture Constraints

- **Adapter pattern is mandatory:** `EngineAdapter` is an abstract base class, `RESTAdapter` is the concrete implementation. This enables future migration from REST to direct Python calls without changing any code outside `gym_env/`. [Source: architecture.md#Architectural Boundaries]
- **Error boundary:** The adapter catches raw HTTP/connection errors and raises `EngineConnectionError`. Training and evaluation scripts must NOT catch this — let it propagate. Only the adapter layer wraps errors. [Source: architecture.md#Error Handling]
- **Use Python `logging` module, NOT print statements.** [Source: architecture.md#Error Handling]
- **No code outside `gym_env/` should directly call the engine API.** [Source: architecture.md#Enforcement Guidelines]

### Game Engine API Details

The existing game engine (`lets-play-brisca/`) exposes these REST endpoints via Flask:

| Endpoint | Method | Request Body | Response |
|---|---|---|---|
| `/api/game/new` | POST | `{player_name, ai_difficulty}` | `{success, game_id, state}` |
| `/api/game/state` | GET | — | `{success, state}` |
| `/api/game/play` | POST | `{card_index}` | `{success, state}` |
| `/api/game/process-ai` | POST | — | `{success, state}` |
| `/api/game/delete` | DELETE | — | `{success}` |

**Game state JSON structure returned by all state-bearing endpoints:**
```json
{
  "hand": [{"rank": int, "suit": str, "suit_symbol": str, "display_name": str, "points": int}],
  "trump": {"rank": int, "suit": str, "suit_symbol": str, "display_name": str, "points": int},
  "trick": [{"player": str, "card": {rank, suit, suit_symbol, display_name, points}}],
  "players": [{"name": str, "is_current": bool, "is_human": bool, "score": int, "hand_size": int}],
  "deck_remaining": int,
  "round_number": int,
  "game_over": bool,
  "winner": str|null,
  "is_your_turn": bool
}
```

**Critical game engine behaviors to account for:**
- **Session-based game tracking:** The engine uses Flask sessions (cookies) to track `game_id`. The adapter MUST use `requests.Session()` to maintain cookies across calls.
- **Turn flow is human → AI → human:** After calling `/api/game/play` (human plays), you MUST call `/api/game/process-ai` to let the AI opponent play. The RL agent is the "human" player from the engine's perspective.
- **AI difficulty for random play:** Use `ai_difficulty="basic"` — the `BasicStrategy` plays the first card in hand, which is the closest to random play available. Alternatively, implementing a truly random strategy in the engine could be a future enhancement.
- **Card encoding:** Ranks are integers: 1-7, 10-12 (no 8, 9 — Spanish deck). Suits are strings: "Oros", "Copas", "Espadas", "Bastos". Total 40 cards, 120 possible points (sum of all card point values).
- **Point values:** 1→11pts, 3→10pts, 12→4pts, 11→3pts, 10→2pts, rest→0pts.
- **Trick resolution timing:** The engine resolves tricks AFTER serializing state. When `play_card` is called and it completes a trick, the response shows the trick cards but the trick winner/new deal hasn't been processed yet. The next `get_state` or `process_ai_turn` call will show the resolved state.
- **No suit-following requirement:** In Brisca, any card from hand can be played at any time. No need for suit-following validation.

### Card Encoding for Observation Vector (Story 1.2 context)

While Story 1.2 will implement the full observation vector, the adapter should return raw game state data that Story 1.2 can encode. The adapter does NOT need to encode observations — it just provides the raw state.

Relevant for later: observation vector structure = [hand(3), trump_card(1), trump_suit(1), trick_cards(2), cards_seen_per_suit(4), agent_points(1), opponent_points(1)] = 13 values total.

### Game Loop for RL Training

A complete game from the adapter's perspective:
```
1. adapter.new_game() → get initial state (RL agent's hand, trump, etc.)
2. Loop until game_over:
   a. RL agent selects card_index (0, 1, or 2)
   b. adapter.play_card(card_index) → updated state
   c. If not game_over: adapter.process_ai_turn() → updated state (opponent played)
3. Extract final scores from state["players"] for reward calculation
4. adapter.delete_game() (cleanup)
```

### Project Structure Notes

Files to create in this story:
```
briscas_rl/
├── gym_env/
│   ├── __init__.py           # Export EngineAdapter, RESTAdapter, EngineConnectionError
│   └── engine_adapter.py     # EngineAdapter ABC + RESTAdapter + EngineConnectionError
├── tests/
│   └── test_engine_adapter.py
└── requirements.txt          # requests (+ pytest for dev)
```

Alignment with architecture.md project structure: exact match. [Source: architecture.md#Complete Project Directory Structure]

### Testing Standards

- Use `pytest` as testing framework [Source: architecture.md#Development Tooling]
- Mock HTTP responses — do NOT require a running game engine for unit tests
- Test file mirrors source: `tests/test_engine_adapter.py` for `gym_env/engine_adapter.py` [Source: architecture.md#Structure Patterns]
- Verify the abstract interface contract (all methods defined on ABC)
- Test error wrapping: `requests.ConnectionError` → `EngineConnectionError`
- Test cookie/session persistence across multiple API calls

### Library/Framework Requirements

| Library | Version | Purpose |
|---|---|---|
| `requests` | latest stable | HTTP client for REST API calls |
| `pytest` | latest stable | Testing framework |
| Python | 3.10+ | Required for SB3/Gymnasium compatibility |
| `stable-baselines3` | 2.7.x | Not needed yet but pin in requirements.txt for project setup |
| `gymnasium` | 1.2.x | Not needed yet but pin in requirements.txt for project setup |
| `torch` | 2.3+ | Not needed yet but pin in requirements.txt for project setup |

**SB3 2.7.x notes (for future stories):**
- Python 3.9 support dropped — use Python 3.10+
- DQN target network shares extractor with main network (correctness fix from 2.4.0+)
- DQN optimizer excludes target_q_network parameters (since 2.4.0)
- Gymnasium 0.29.1+ required (1.2.x is current stable)
- Custom envs must implement proper `reset()` signature — `DummyVecEnv` passes `seed` argument

### References

- [Source: architecture.md#Architectural Boundaries — Boundary 1: Game Engine ↔ Environment]
- [Source: architecture.md#Error Handling — EngineConnectionError pattern]
- [Source: architecture.md#Implementation Patterns — Adapter pattern]
- [Source: architecture.md#Enforcement Guidelines — No direct API calls outside gym_env/]
- [Source: architecture.md#Complete Project Directory Structure — gym_env/ layout]
- [Source: epics.md#Story 1.1 — Acceptance criteria and BDD scenarios]
- [Source: prd.md#Environment Integration — FR1, FR3, FR4, NFR1]
- [Source: lets-play-brisca/app/api/game_routes.py — REST API endpoint definitions]
- [Source: lets-play-brisca/app/services/game_service.py — State serialization format, trick resolution timing]
- [Source: lets-play-brisca/app/core/models/card.py — Card ranks, point values, Spanish deck structure]
- [Source: lets-play-brisca/app/core/models/suit.py — Suit enum values: Oros, Copas, Espadas, Bastos]
- [Source: lets-play-brisca/app/core/game/engine.py — Game lifecycle, trick resolution]
- [Source: lets-play-brisca/app/core/players/ai_player.py — AI difficulty strategies]

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
