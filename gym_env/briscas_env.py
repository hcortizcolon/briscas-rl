"""BriscasEnv — Gymnasium environment wrapper for the Briscas card game."""

import logging

import gymnasium
import numpy as np

from gym_env.engine_adapter import EngineAdapter, GameState
from gym_env.observation import (
    TOTAL_POINTS,
    build_observation,
    build_observation_space,
    encode_card,
    sorted_hand_index,
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
        info = {"first_player": 0 if self._state.is_your_turn else 1}

        # If AI goes first, let it play before returning
        if not self._state.is_your_turn:
            self._state = self._adapter.process_ai_turn()
            self._update_cards_seen(self._state)

        return self._get_observation(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        hand_size = len(self._state.hand)
        if hand_size == 0:
            raise ValueError("Cannot call step() with an empty hand (game is over)")
        masked_action = action % hand_size
        # The observation encodes the hand sorted by card ID, so the agent's
        # action index refers to the sorted order.  Map it back to the engine's
        # hand index so the intended card is actually played.
        engine_index = self._sorted_hand_index(masked_action)
        self._state = self._execute_turn(engine_index)

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
            for p in self._state.players:
                if p.is_human:
                    info["agent_points"] = p.score
                else:
                    info["opponent_points"] = p.score
            reward = raw_reward * self.reward_scale
            terminated = True
        else:
            reward = 0.0
            terminated = False

        return self._get_observation(), reward, terminated, False, info

    def _sorted_hand_index(self, sorted_idx: int) -> int:
        """Map an index in the sorted hand back to the engine's hand index."""
        return sorted_hand_index(self._state.hand, sorted_idx)

    def _execute_turn(self, card_index: int) -> GameState:
        """Play card and loop until it's our turn or game over.

        The engine returns unresolved state when a trick is complete (2 cards).
        Normally process_ai_turn resolves the trick, but on the final trick
        (empty hand, empty deck) the AI has no cards left so we must use
        get_state instead.
        """
        state = self._adapter.play_card(card_index)
        self._update_cards_seen(state)
        while not state.game_over:
            if len(state.trick) >= 2:
                if len(state.hand) == 0 and state.deck_remaining == 0:
                    # Final trick — AI has no cards; resolve via get_state
                    state = self._adapter.get_state()
                else:
                    state = self._adapter.process_ai_turn()
                self._update_cards_seen(state)
            elif not state.is_your_turn:
                state = self._adapter.process_ai_turn()
                self._update_cards_seen(state)
            else:
                break
        return state

    def _get_observation(self) -> np.ndarray:
        """Single source of truth for observation encoding."""
        state = self._state
        agent_score = 0
        opponent_score = 0
        for p in state.players:
            if p.is_human:
                agent_score = p.score
            else:
                opponent_score = p.score
        return build_observation(
            hand=state.hand,
            trump=state.trump,
            trick=state.trick,
            cards_seen=self._cards_seen,
            deck_remaining=state.deck_remaining,
            agent_score=agent_score,
            opponent_score=opponent_score,
        )

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
