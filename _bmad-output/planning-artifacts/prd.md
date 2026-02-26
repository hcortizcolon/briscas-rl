---
stepsCompleted:
  - step-01-init
  - step-02-discovery
  - step-02b-vision
  - step-02c-executive-summary
  - step-03-success
  - step-04-journeys
  - step-05-domain
  - step-06-innovation
  - step-07-project-type
  - step-08-scoping
  - step-09-functional
  - step-10-nonfunctional
  - step-11-polish
date: '2026-02-23'
inputDocuments: []
workflowType: 'prd'
classification:
  projectType: scientific_ml
  domain: scientific
  complexity: low
  projectContext: brownfield
---

# Product Requirements Document - briscas_rl

**Author:** Caleb
**Date:** 2026-02-23

## Executive Summary

A personal RL learning project that trains two opposing agents to play the Spanish card game Briscas: one maximizing points, one minimizing points. Built on an existing game engine, the project measures how much skill (optimal vs. anti-optimal strategy) influences outcomes in a game with significant hidden information. A random-play baseline completes the comparison. Results are measured over thousands of games using win rate, point differential, and outcome variance.

### What Makes This Special

The dual-agent approach (best + worst + random baseline) turns a simple card game AI into an experiment about the nature of Briscas itself. By comparing trained extremes against random play, the project answers: is Briscas skill-dominated or luck-dominated? An interesting sub-question emerges — can a trained worst-agent actually perform worse than random, or does the hidden information floor limit how bad deliberate play can be?

## Project Classification

- **Type:** Scientific/ML — Python RL training pipeline
- **Domain:** AI/ML experimentation (personal learning)
- **Complexity:** Low
- **Context:** Brownfield — building on existing Briscas game engine
- **Observation model:** Agent's hand, trump card/suit, current trick, full play history

## Success Criteria

### Technical Success

- Best agent's win rate vs random has plateaued (converged) — practical optimality given observable information
- Worst agent's win rate vs random has plateaued at a low point — converged to anti-optimal play
- Training runs complete in reasonable time on local hardware
- Results are reproducible across runs

### Measurable Outcomes

- Win rate matrix: best vs. worst, best vs. random, worst vs. random
- Average point differential for each matchup
- Outcome variance for each matchup
- **The key number:** percentage of games where the worst agent beats the best agent — this measures how often deliberate worst-play still wins due to favorable deals and hidden information

## Product Scope

### MVP (Phase 1)

- Environment wrapper around existing game engine
- Best agent (maximize points via RL)
- Worst agent (minimize points via RL)
- Training loop with manual convergence inspection
- Model save/load for trained agents
- Evaluation script: configurable N games per matchup, alternating first player, results to CSV

### Growth Features (Phase 2)

- Compare reward strategies (end-of-game vs per-trick points)
- Agent decision visualization
- Training curves / convergence plots
- Worst vs random evaluation
- Blog post writeup

### Vision (Phase 3)

- Self-play training
- Algorithm comparison (DQN vs PPO vs others)
- Multiplayer Briscas variant

## User Journeys

### Researcher Journey: Train and Evaluate Briscas RL Agents

**Actor:** Caleb (sole user, running locally)

1. **Environment Setup:** Wire environment wrapper around existing Briscas game engine — define state space, action space, reward signal (points)
2. **Train Best Agent:** Train maximizer agent against the engine's random player until convergence (win rate plateaus)
3. **Train Worst Agent:** Train minimizer agent against random player with flipped reward signal until convergence
4. **Evaluate — Best vs Random:** Run 10,000 games (alternating first player), collect win rates, per-game point totals, and point differential distribution
5. **Evaluate — Best vs Worst:** Run 10,000 games (alternating first player), collect same metrics — produces the key unwinnable percentage
6. **Analyze Results:** Compare matchup statistics, point differential distributions, answer the skill-vs-luck question
7. **(Optional) Evaluate — Worst vs Random:** Test whether trained worst-play is actually worse than random

### Design Decisions

- Worst agent trains against random (not against best) — simpler, and the evaluation against best still answers the core question
- Post-training agents are deterministic — 10,000 games purely measures the deal's influence
- Alternate first player across games to eliminate lead-trick bias
- Track full per-game point totals, not just win/lose — distribution matters

## Technical Architecture

### Stack

- **RL Algorithm:** DQN — discrete action space (1 of up to 3 cards), well-documented, GPU-friendly, beginner-approachable
- **Framework:** PyTorch + Stable Baselines3
- **Hardware:** RTX 3060Ti, local execution
- **Game engine integration:** Local REST API. Monitor training speed — if HTTP overhead bottlenecks training, refactor to direct Python calls

### Implementation Details

- **Environment:** Gymnasium interface wrapping game engine's local REST API
- **State representation:** Fixed-size numerical vector — hand (sorted consistently), trump card/suit, current trick, play history. Encode card IDs, not positions.
- **Action space:** Discrete(3) — index of card in hand to play
- **Reward:** End-of-game points. Per-trick reward comparison deferred to Phase 2.
- **Worst agent:** Same DQN setup, negated reward signal
- **Training opponent:** Engine's built-in random player
- **Reproducibility:** Set random seeds for NumPy, PyTorch, and game engine shuffling
- **Model persistence:** Save/load trained models so evaluation doesn't require retraining
- **Worst agent validation:** Verify worst agent's win rate vs random is meaningfully below 50% — if not, the negated reward approach isn't producing true anti-optimal play

### Risk Mitigation

- **Training speed:** Monitor REST API overhead; refactor to direct Python calls if bottleneck
- **Experiment validity:** Validate worst agent is meaningfully worse than random before drawing conclusions

## Functional Requirements

### Environment Integration

- FR1: System can connect to the existing Briscas game engine via local REST API
- FR2: System can translate game engine state into a fixed-size numerical observation vector (hand, trump, current trick, play history)
- FR3: System can translate agent action (card index) into a valid game engine move
- FR4: System can receive end-of-game point totals as reward signal from the engine
- FR5: System can constrain agent actions to only legal moves given current game state

### Agent Training

- FR6: Researcher can train a best agent (point maximizer) against the engine's random player
- FR7: Researcher can train a worst agent (point minimizer) against the engine's random player
- FR8: Researcher can monitor training progress by inspecting win rate over episodes
- FR9: Researcher can save a trained agent model to disk
- FR10: Researcher can load a previously trained agent model from disk
- FR11: System can output training summary statistics upon completion (final win rate, episodes trained)

### Game Evaluation

- FR12: Researcher can run N evaluation games between any two agents (best, worst, or engine's random)
- FR13: Researcher can configure the number of evaluation games per matchup
- FR14: System can alternate which agent plays first across evaluation games
- FR15: System can record per-game results to CSV (both players' points, which agent went first, point differential)
- FR18: Researcher can view win/loss/draw rates and point differentials after running an evaluation matchup

### Reproducibility

- FR16: Researcher can set random seeds for training and evaluation runs
- FR17: System can produce reproducible results given the same seed

## Non-Functional Requirements

### Integration Reliability

- Environment must handle game engine API errors gracefully — retry or fail with clear error message, never silently corrupt training data
