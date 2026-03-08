# Story 3.2: Summary Statistics and Analysis

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a researcher,
I want to see win rates, point differentials, and outcome variance after an evaluation run,
So that I can answer the skill-vs-luck question without manually analyzing the CSV.

## Acceptance Criteria

1. **Given** an evaluation run has completed (via `run_evaluation()`), **When** results are returned, **Then** summary statistics are automatically printed to stdout via `print()` (not logger), including: win/loss/draw count and percentages for each agent (draw = `point_differential == 0`, i.e., 60-60), average point differential, point differential sample standard deviation (outcome variance, `ddof=1`), and upset rate (minority win % — percentage of games won by the agent with fewer total wins). (AC: FR18)

2. **Given** an existing CSV file from a previous evaluation run, **When** `compute_summary_statistics(csv_path)` is called, **Then** the same statistics are computed and printed to stdout — enabling analysis of historical results without re-running games. (AC: FR18, enhancement)

3. **Given** multiple matchups are run separately (e.g., best vs random, worst vs random), **When** each `run_evaluation()` call completes, **Then** each independently prints its own summary statistics. The researcher compares printed summaries side-by-side to answer the skill-vs-luck question — no multi-matchup orchestrator needed. **Note:** model-vs-model (e.g., best vs worst) is not supported by the engine; the comparison is done indirectly. (AC: FR18)

4. **Given** edge cases in game results, **When** all games are won by one agent, or all games are draws, or only a single game is played, **Then** statistics are computed correctly without errors (no division by zero, no undefined std dev crashes).

## Tasks / Subtasks

- [x] Task 1: Create `compute_summary_statistics()` and `print_summary_statistics()` in `evaluation/evaluate.py` (AC: #1, #2, #4)
  - [x] Implement `compute_summary_statistics(results: list[dict]) -> dict` — pure computation, no side effects
  - [x] Input `results` dicts must have keys: `agent1_points`, `agent2_points`, `point_differential` (same as CSV columns)
  - [x] Guard: `if len(results) == 0: raise ValueError("No results to analyze")`
  - [x] Win/loss/draw counts: agent1 wins when `point_differential > 0`, agent2 wins when `point_differential < 0`, draw when `point_differential == 0`
  - [x] Win/loss/draw percentages for each agent
  - [x] Average point differential (from agent1's perspective)
  - [x] Point differential sample std dev: `if len(differentials) < 2: stdev = 0.0` else `statistics.stdev(differentials)`
  - [x] Upset rate: `min(agent1_wins, agent2_wins) / total_games * 100`. When all draws (0 wins for both), report `0.0`
  - [x] Return dict with explicit keys:
    ```python
    {
        "agent1_wins": int, "agent1_losses": int, "agent1_draws": int,
        "agent1_win_pct": float, "agent1_loss_pct": float, "agent1_draw_pct": float,
        "agent2_wins": int, "agent2_losses": int, "agent2_draws": int,
        "agent2_win_pct": float, "agent2_loss_pct": float, "agent2_draw_pct": float,
        "avg_point_differential": float,
        "std_point_differential": float,
        "upset_rate": float,
        "total_games": int,
    }
    ```
  - [x] Implement `print_summary_statistics(stats: dict, agent1_name: str, agent2_name: str) -> None` — presentation only, prints to stdout via `print()`
  - [x] Float precision: percentages `:.1f`, avg differential `:+.1f` (always show sign), std dev `:.1f`
  - [x] Output format (no alignment padding — consistent per line):
    ```
    === Evaluation Summary ===
    Agent 1 (best_agent_1000k): Win 70.0% (7000) | Loss 25.0% (2500) | Draw 5.0% (500)
    Agent 2 (random): Win 25.0% (2500) | Loss 70.0% (7000) | Draw 5.0% (500)
    Avg Point Differential: +15.3 (agent1 perspective)
    Std Dev: 22.1
    Upset Rate: 25.0% (minority winner)
    Total Games: 10000
    ===========================
    ```

- [x] Task 2: Create `compute_summary_statistics_from_csv()` wrapper (AC: #2)
  - [x] Implement `compute_summary_statistics_from_csv(csv_path: str, agent1_name: str | None = None, agent2_name: str | None = None) -> dict`
  - [x] Read CSV using `csv.DictReader`, parse rows into results list with `int()` conversion for `agent1_points`, `agent2_points`, and `point_differential` columns. Ignore `game_id` and `first_player`
  - [x] If `agent1_name`/`agent2_name` not provided, attempt filename parsing (split on `_vs_`). If parsing fails, fall back to `"Agent 1"` / `"Agent 2"`
  - [x] Call `compute_summary_statistics()` then `print_summary_statistics()` with parsed data and names
  - [x] Return the stats dict

- [x] Task 3: Integrate into `run_evaluation()` (AC: #1, #3)
  - [x] After CSV writing and before `return csv_path`, call `compute_summary_statistics(results)` then `print_summary_statistics(stats, agent1_name, agent2_name)`
  - [x] Keep return type as `str` (csv_path only) — no API break
  - [x] Update `evaluation/__init__.py` `__all__` to include `compute_summary_statistics`, `print_summary_statistics`, `compute_summary_statistics_from_csv`

- [x] Task 4: Write tests in `tests/test_evaluation.py` (AC: #1, #2, #3, #4)
  - [x] New class `TestSummaryStatistics` — add to existing test file
  - [x] Test win/loss/draw counts and percentages with known data
  - [x] Test average point differential calculation
  - [x] Test sample standard deviation (`ddof=1` via `statistics.stdev`)
  - [x] Test upset rate calculation
  - [x] Test edge case: all games won by agent1 (0% upset rate, 0% agent2 win rate)
  - [x] Test edge case: all draws (0% upset rate, 0% win rate for both, stdev = 0.0)
  - [x] Test edge case: single game (stdev = 0.0, no crash)
  - [x] Test edge case: empty results raises `ValueError("No results to analyze")`
  - [x] Test `compute_summary_statistics_from_csv()` reads CSV correctly
  - [x] Test CSV name parsing fallback when filename doesn't follow convention
  - [x] Test `print_summary_statistics()` output format (capture with `capsys`)
  - [x] Test `run_evaluation()` integration prints stats after CSV write (capture with `capsys`)

## Dev Notes

### Technical Requirements

- **Summary statistics are stdout, not logging.** Use `print()` for all stats output. The logger remains for operational messages (e.g., "Evaluation complete: 10000 games"). Stats go to stdout so the researcher can pipe, redirect, or copy-paste cleanly. [Source: architecture.md#Evaluation Pattern — "Summary statistics printed to stdout after completion"]
- **Pure compute + separate print.** `compute_summary_statistics()` is a pure function returning a dict. `print_summary_statistics()` handles formatting and `print()`. This separation keeps compute testable without capsys.
- **`statistics.stdev()` for sample std dev.** Uses `ddof=1` by default (unlike `numpy.std()` which defaults to `ddof=0`). Guard `len < 2` to avoid `StatisticsError`.
- **No new dependencies.** Only `statistics` (stdlib) and `csv` (stdlib) are needed. No numpy/pandas for this story.
- **Upset rate definition:** `min(agent1_wins, agent2_wins) / total_games * 100`. This is the percentage of games won by whichever agent won fewer games overall — the "minority winner." When both agents have 0 wins (all draws), upset rate is `0.0`.

### Architecture Compliance

- **File location:** All new functions go in existing `evaluation/evaluate.py`. No new files created in `evaluation/`. [Source: architecture.md#Complete Project Directory Structure]
- **Boundary 3 preserved:** `compute_summary_statistics()` and `print_summary_statistics()` are library code in `evaluation/`, not in `scripts/`. [Source: architecture.md#Architectural Boundaries]
- **`run_evaluation()` return type unchanged:** Returns `str` (csv_path). No API break. [Source: architecture.md#Evaluation Pattern]
- **Error handling:** Use Python `logging` module for operational messages only. Stats use `print()`. Exceptions propagate to top level. [Source: architecture.md#Error Handling]
- **Export new functions:** Add `compute_summary_statistics`, `print_summary_statistics`, `compute_summary_statistics_from_csv` to `evaluation/__init__.py` `__all__`.

### Library / Framework Requirements

No new dependencies. All from stdlib and existing packages:

| Library | Version | Purpose | Notes |
|---|---|---|---|
| `statistics` | stdlib | `stdev()` for sample std dev | No install needed |
| `csv` | stdlib | Read CSV for `from_csv` wrapper | Already used in evaluate.py |
| `os` | stdlib | Path/filename parsing | Already used in evaluate.py |

### File Structure Requirements

Files to create/modify in this story:
```
briscas_rl/
├── evaluation/
│   ├── __init__.py                 # MODIFIED — add new exports
│   └── evaluate.py                 # MODIFIED — add 3 new functions, integrate into run_evaluation()
├── tests/
│   └── test_evaluation.py          # MODIFIED — add TestSummaryStatistics class (~12 new tests)
```

### Testing Standards

- Add to existing `tests/test_evaluation.py` — same module, related functionality. [Source: architecture.md#Development Tooling]
- New class `TestSummaryStatistics` for compute/print tests
- New class `TestSummaryStatisticsFromCsv` for CSV wrapper tests
- Use `capsys` pytest fixture for stdout capture (no mocking needed for `print()`)
- Use `tmp_path` for CSV file creation in `from_csv` tests
- All existing 190 tests MUST continue passing
- Expected new tests: ~12 tests

### Previous Story Intelligence (Story 3.1)

**What was built:**
- `evaluation/evaluate.py` — `run_evaluation()` plays games, writes CSV, returns path
- `_extract_agent_name()` — extracts name from path or literal ("advanced", "random")
- `evaluation/__init__.py` — exports `run_evaluation`
- 190 tests passing (170 env/training + 20 evaluation)

**Patterns confirmed:**
- `logging.getLogger(__name__)` for logger setup
- `__init__.py` with `__all__` exports
- Single-line docstrings
- `unittest.mock` for mocking in tests
- `csv.DictWriter` for CSV output
- `os.makedirs(dir, exist_ok=True)` for directory creation
- `_extract_agent_name()` reusable for name parsing

**Code patterns from `run_evaluation()` (integration point):**
```python
# ... after writer.writerows(results) ...
logger.info(
    "Evaluation complete: %d games | %s vs %s | Output: %s",
    num_games, agent1_name, agent2_name, csv_path,
)
return csv_path
```
New code inserts between `logger.info()` and `return csv_path`.

**In-memory results format (from run_evaluation):**
```python
results.append({
    "game_id": game_id,
    "agent1_points": a1_points,
    "agent2_points": a2_points,
    "first_player": csv_first_player,
    "point_differential": a1_points - a2_points,
})
```

### Git Intelligence

**Recent commits (last 5):**
1. `4246875` — Refactor BriscasEnv reset() for dynamic first player
2. `9c03bea` — Finalize evaluation matchups (run_evaluation, CLI, tests)
3. `d6227b0` — Add LocalAdapter for in-process game engine
4. `91ace4c` — Add loss/draw rates to WinRateCallback
5. `5425094` — Refactor observation structure to 50 features

**Code conventions confirmed:**
- `logging.getLogger(__name__)` for logger setup
- `__init__.py` with `__all__` exports
- Single-line docstrings
- `unittest.mock` for mocking in tests
- `@pytest.mark.integration` for integration tests
- CLI pattern: argparse, logging.basicConfig, sys.path insertion

**In-progress changes (unstaged):**
- `evaluation/evaluate.py` and `scripts/evaluate.py` refactored to support `"advanced"` engine strategy alongside `"random"`. Tests updated to match. These changes are NOT yet committed — the dev agent should work on top of the current working tree state.

### Project Context Reference

- PRD: `_bmad-output/planning-artifacts/prd.md` — FR18, upset percentage as key metric
- Architecture: `_bmad-output/planning-artifacts/architecture.md` — Evaluation Pattern, stdout stats
- Epics: `_bmad-output/planning-artifacts/epics.md` — Epic 3, Story 3.2 acceptance criteria
- Previous story: `_bmad-output/implementation-artifacts/3-1-run-evaluation-matchups.md` — run_evaluation(), 190 tests

### References

- [Source: architecture.md#Evaluation Pattern — Summary statistics printed to stdout after completion]
- [Source: architecture.md#Evaluation Pattern — CSV naming convention]
- [Source: architecture.md#Complete Project Directory Structure — evaluation/evaluate.py]
- [Source: architecture.md#Architectural Boundaries — Boundary 3: scripts vs library code]
- [Source: architecture.md#Error Handling — logging module, exceptions propagate]
- [Source: epics.md#Story 3.2 — Acceptance Criteria, upset percentage definition]
- [Source: prd.md#Success Criteria — "The key number: percentage of games where worst beats best"]
- [Source: 3-1-run-evaluation-matchups.md — run_evaluation(), results format, 190 tests]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

None — clean implementation, no debug issues.

### Completion Notes List

- Task 1: Implemented `compute_summary_statistics()` as pure function returning dict with all specified keys. Implemented `print_summary_statistics()` with exact output format per spec. Added `import statistics` for `stdev()`. Guards: empty list raises `ValueError`, single-game stdev = 0.0, all-draws upset rate = 0.0.
- Task 2: Implemented `compute_summary_statistics_from_csv()` — reads CSV via `DictReader`, parses `_vs_` filename convention for agent names, falls back to "Agent 1"/"Agent 2".
- Task 3: Integrated into `run_evaluation()` — calls compute then print between `logger.info()` and `return csv_path`. Return type unchanged (`str`). Updated `evaluation/__init__.py` exports.
- Task 4: 13 new tests across `TestSummaryStatistics` (9 tests), `TestSummaryStatisticsFromCsv` (3 tests), `TestRunEvaluationPrintsStats` (1 test).

### File List

- `evaluation/evaluate.py` — MODIFIED: added `compute_summary_statistics()`, `print_summary_statistics()`, `compute_summary_statistics_from_csv()`, integrated stats into `run_evaluation()`. Also refactored to support `"advanced"` engine strategy alongside `"random"`.
- `evaluation/__init__.py` — MODIFIED: added 3 new exports to `__all__`
- `gym_env/local_adapter.py` — MODIFIED: added `_random_choose_card_index()` function, `STRATEGIES` dict, and `strategy` parameter to `BriscasGame` and `LocalAdapter` for configurable AI strategy
- `scripts/evaluate.py` — MODIFIED: updated help text for `"advanced"`/`"random"` strategies, removed redundant `set_all_seeds()` call (already called inside `run_evaluation()`)
- `tests/test_evaluation.py` — MODIFIED: added `TestSummaryStatistics`, `TestSummaryStatisticsFromCsv`, `TestRunEvaluationPrintsStats` classes. Added tests for `"random"` strategy, mixed engine strategies validation, `_extract_agent_name("random")`, explicit CSV name overrides.
- `tests/test_local_adapter.py` — MODIFIED: added `TestRandomStrategy` class (4 tests) and strategy parameterization tests to `TestBriscasGame` (3 tests)

## Change Log

- 2026-03-08: Code review — 9 fixes (1 HIGH, 5 MEDIUM, 3 LOW). Added 12 new tests: 7 in `test_local_adapter.py` (random strategy, strategy parameterization), 5 in `test_evaluation.py` (both_random_raises, mixed_strategies_raises, random_literal, random_strategy_runs, explicit_name_overrides). Removed redundant `set_all_seeds()` from `scripts/evaluate.py`. Updated File List with 3 previously undocumented files. All 218 tests passing.
