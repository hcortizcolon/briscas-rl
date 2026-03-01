"""Engine adapter interface and REST implementation for Briscas game engine."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)


class EngineConnectionError(Exception):
    """Raised when the adapter cannot communicate with the game engine."""


@dataclass(frozen=True)
class Card:
    """A single card in the Briscas deck."""

    rank: int
    suit: str
    suit_symbol: str
    display_name: str
    points: int


@dataclass(frozen=True)
class TrickCard:
    """A card played in the current trick, with the player who played it."""

    player: str
    card: Card


@dataclass(frozen=True)
class PlayerInfo:
    """Public information about a player."""

    name: str
    is_current: bool
    is_human: bool
    score: int
    hand_size: int


@dataclass(frozen=True)
class GameState:
    """Complete game state returned by the engine."""

    hand: list[Card]
    trump: Card
    trick: list[TrickCard]
    players: list[PlayerInfo]
    deck_remaining: int
    round_number: int
    game_over: bool
    winner: str | None
    is_your_turn: bool


def _parse_card(data: dict) -> Card:
    return Card(
        rank=data["rank"],
        suit=data["suit"],
        suit_symbol=data["suit_symbol"],
        display_name=data["display_name"],
        points=data["points"],
    )


def _parse_game_state(data: dict) -> GameState:
    return GameState(
        hand=[_parse_card(c) for c in data["hand"]],
        trump=_parse_card(data["trump"]),
        trick=[
            TrickCard(player=t["player"], card=_parse_card(t["card"]))
            for t in data["trick"]
        ],
        players=[
            PlayerInfo(
                name=p["name"],
                is_current=p["is_current"],
                is_human=p["is_human"],
                score=p["score"],
                hand_size=p["hand_size"],
            )
            for p in data["players"]
        ],
        deck_remaining=data["deck_remaining"],
        round_number=data["round_number"],
        game_over=data["game_over"],
        winner=data["winner"],
        is_your_turn=data["is_your_turn"],
    )


class EngineAdapter(ABC):
    """Abstract base class for game engine communication."""

    @abstractmethod
    def new_game(self) -> GameState:
        """Start a new game and return initial state."""

    @abstractmethod
    def play_card(self, card_index: int) -> GameState:
        """Play a card from hand by index and return updated state."""

    @abstractmethod
    def process_ai_turn(self) -> GameState:
        """Let the AI opponent play and return updated state."""

    @abstractmethod
    def get_state(self) -> GameState:
        """Get current game state."""

    @abstractmethod
    def delete_game(self) -> None:
        """Delete the current game (cleanup)."""


class RESTAdapter(EngineAdapter):
    """Concrete adapter that communicates with the Briscas engine via REST API."""

    DEFAULT_TIMEOUT = 10

    def __init__(self, base_url: str = "http://localhost:5000") -> None:
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def new_game(self) -> GameState:
        data = self._post("/api/game/new", json={"player_name": "rl_agent", "ai_difficulty": "basic"})
        return _parse_game_state(data["state"])

    def play_card(self, card_index: int) -> GameState:
        data = self._post("/api/game/play", json={"card_index": card_index})
        return _parse_game_state(data["state"])

    def process_ai_turn(self) -> GameState:
        data = self._post("/api/game/process-ai")
        return _parse_game_state(data["state"])

    def get_state(self) -> GameState:
        data = self._get("/api/game/state")
        return _parse_game_state(data["state"])

    def delete_game(self) -> None:
        self._request("DELETE", "/api/game/delete", parse_json=False)

    def _post(self, path: str, **kwargs) -> dict:
        return self._request("POST", path, **kwargs)

    def _get(self, path: str) -> dict:
        return self._request("GET", path)

    def _request(self, method: str, path: str, parse_json: bool = True, **kwargs) -> dict | None:
        url = f"{self._base_url}{path}"
        try:
            response = self._session.request(method, url, timeout=self.DEFAULT_TIMEOUT, **kwargs)
            response.raise_for_status()
            if not parse_json:
                return None
            data = response.json()
        except requests.ConnectionError as exc:
            raise EngineConnectionError(
                f"Cannot connect to game engine at {self._base_url}: {exc}"
            ) from exc
        except requests.HTTPError as exc:
            raise EngineConnectionError(
                f"Game engine returned error {response.status_code} for {method} {path}: {exc}"
            ) from exc
        except requests.RequestException as exc:
            raise EngineConnectionError(
                f"Request failed for {method} {path}: {exc}"
            ) from exc
        except ValueError as exc:
            raise EngineConnectionError(
                f"Invalid JSON response for {method} {path}: {exc}"
            ) from exc
        if not data.get("success", True):
            raise EngineConnectionError(
                f"Game engine returned failure for {method} {path}: {data}"
            )
        return data
