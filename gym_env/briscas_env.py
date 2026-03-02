"""BriscasEnv — Gymnasium environment wrapper for the Briscas card game."""

import logging

import gymnasium
import numpy as np

from gym_env.engine_adapter import EngineAdapter, GameState
from gym_env.observation import (
    OBSERVATION_SIZE,
    TOTAL_POINTS,
    SUIT_INDEX,
    build_observation_space,
    encode_card,
)

logger = logging.getLogger(__name__)


class BriscasEnv(gymnasium.Env):
    """Gymnasium environment for two-player Briscas via an EngineAdapter."""

    metadata = {"render_modes": []}

    def __init__(self, adapter: EngineAdapter, reward_scale: float = 1.0) -> None:
        super().__init__()
        self._adapter = adapter
        self.reward_scale = reward_scale
        self.observation_space = build_observation_space()
        self.action_space = gymnasium.spaces.Discrete(3)
        self._state: GameState | None = None
        self._cards_seen: set[int] = set()
        self._game_active: bool = False

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if self._game_active:
            self._adapter.delete_game()
        self._state = self._adapter.new_game()
        self._game_active = True
        self._cards_seen = set()
        return self._get_observation(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        hand_size = len(self._state.hand)
        if hand_size == 0:
            raise ValueError("Cannot call step() with an empty hand (game is over)")
        masked_action = action % hand_size
        self._state = self._execute_turn(masked_action)

        info = {}
        if self._state.game_over:
            raw_reward = self._compute_reward()
            # game_result based on raw (unscaled) reward
            if raw_reward > 0:
                info["game_result"] = "win"
            elif raw_reward < 0:
                info["game_result"] = "loss"
            else:
                info["game_result"] = "draw"
            reward = raw_reward * self.reward_scale
            terminated = True
        else:
            reward = 0.0
            terminated = False

        return self._get_observation(), reward, terminated, False, info

    def _execute_turn(self, card_index: int) -> GameState:
        """Play card and loop process_ai_turn until it's our turn or game over."""
        state = self._adapter.play_card(card_index)
        self._update_cards_seen(state)
        while not state.is_your_turn and not state.game_over:
            state = self._adapter.process_ai_turn()
            self._update_cards_seen(state)
        return state

    def _get_observation(self) -> np.ndarray:
        """Single source of truth for observation encoding."""
        obs = np.full(OBSERVATION_SIZE, -1.0, dtype=np.float32)
        state = self._state

        # Hand cards (sorted by card ID, padded with -1)
        hand_ids = sorted(encode_card(c) for c in state.hand)
        for i, cid in enumerate(hand_ids):
            obs[i] = cid

        # Trump card and suit
        obs[3] = encode_card(state.trump)
        obs[4] = SUIT_INDEX[state.trump.suit]

        # Trick cards
        for i, tc in enumerate(state.trick):
            obs[5 + i] = encode_card(tc.card)

        # Cards seen per suit (indices 7-10, default 0)
        for i in range(4):
            obs[7 + i] = sum(1 for cid in self._cards_seen if cid // 10 == i)

        # Agent and opponent points
        for p in state.players:
            if p.is_human:
                obs[11] = p.score
            else:
                obs[12] = p.score

        return obs

    def _update_cards_seen(self, state: GameState) -> None:
        """Add trick cards to _cards_seen set."""
        for tc in state.trick:
            self._cards_seen.add(encode_card(tc.card))

    def close(self) -> None:
        """Clean up active game on the engine."""
        if self._game_active:
            self._adapter.delete_game()
            self._game_active = False
        super().close()

    def _compute_reward(self) -> float:
        """Compute normalized point differential reward."""
        agent_score = 0
        opponent_score = 0
        for p in self._state.players:
            if p.is_human:
                agent_score = p.score
            else:
                opponent_score = p.score
        return (agent_score - opponent_score) / TOTAL_POINTS
