"""คลาสที่เกี่ยวกับเกมโป๊กเกอร์: Player, HandEvaluator และ PokerTable"""
import itertools
from collections import Counter

from core.cards import Card


class DuplicateCardError(ValueError):
    """เกิดขึ้นเมื่อมีไพ่ใบเดียวกันถูกใช้มากกว่า 1 ครั้งบนโต๊ะ"""


class Player:
    """ผู้เล่น 1 คน ถือไพ่ในมือ 2 ใบ"""

    def __init__(self, name: str, hole_cards):
        hole_cards = tuple(hole_cards)
        if len(hole_cards) != 2:
            raise ValueError("A player must hold exactly 2 cards")
        self.__name = name
        self.__hole_cards = hole_cards

    @property
    def name(self) -> str:
        return self.__name

    @property
    def hole_cards(self) -> tuple:
        return self.__hole_cards

    def __repr__(self) -> str:
        return f"Player({self.__name!r}, {[c.code for c in self.__hole_cards]})"


class HandEvaluator:
    """ประเมินความแรงของไพ่ คืนค่าเป็น tuple (ระดับมือ, ไพ่ตัดสิน) ที่นำมาเปรียบเทียบกันได้โดยตรง"""

    HAND_NAMES = [
        "High Card", "One Pair", "Two Pair", "Three of a Kind", "Straight",
        "Flush", "Full House", "Four of a Kind", "Straight Flush",
    ]
    WHEEL = {12, 3, 2, 1, 0}  # A-2-3-4-5

    def evaluate_5(self, cards) -> tuple:
        values = [c.value for c in cards]
        counts = Counter(values)
        unique = sorted(counts, reverse=True)

        is_flush = len({c.suit for c in cards}) == 1
        straight_high = self.__straight_high(unique)

        # เรียงไพ่ตัดสินตาม (จำนวนใบ, แต้ม) เช่น แพร์ K มาก่อนไพ่เดี่ยว A
        kickers = [v for v, _ in sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]
        shape = sorted(counts.values(), reverse=True)

        if is_flush and straight_high is not None:
            return (8, [straight_high])
        if shape[0] == 4:
            return (7, kickers)
        if shape == [3, 2]:
            return (6, kickers)
        if is_flush:
            return (5, kickers)
        if straight_high is not None:
            return (4, [straight_high])
        if shape[0] == 3:
            return (3, kickers)
        if shape == [2, 2, 1]:
            return (2, kickers)
        if shape[0] == 2:
            return (1, kickers)
        return (0, kickers)

    def evaluate_7(self, cards) -> tuple:
        """เลือกไพ่ 5 ใบที่ดีที่สุดจากไพ่ 7 ใบ"""
        return max(self.evaluate_5(combo) for combo in itertools.combinations(cards, 5))

    def hand_name(self, score: tuple) -> str:
        return self.HAND_NAMES[score[0]]

    def __straight_high(self, unique_values):
        if len(unique_values) != 5:
            return None
        if unique_values[0] - unique_values[4] == 4:
            return unique_values[0]
        if set(unique_values) == self.WHEEL:
            return 3  # สเตรทที่ต่ำที่สุด สูงสุดที่ไพ่ 5
        return None


class PokerTable:
    """โต๊ะโป๊กเกอร์ 1 สถานการณ์ ประกอบด้วยผู้เล่นหลายคนและไพ่กลาง (Composition)"""

    def __init__(self, players, board):
        self.__players = tuple(players)
        self.__board = tuple(board)
        self.__validate()

    @property
    def players(self) -> tuple:
        return self.__players

    @property
    def board(self) -> tuple:
        return self.__board

    @property
    def turn_board(self) -> tuple:
        """ไพ่กลาง 4 ใบแรก (Flop 3 + Turn 1)"""
        return self.__board[:4]

    @property
    def has_river(self) -> bool:
        return len(self.__board) == 5

    def all_cards(self) -> list:
        cards = [c for p in self.__players for c in p.hole_cards]
        return cards + list(self.__board)

    def __validate(self):
        if len(self.__board) not in (4, 5):
            raise ValueError("Board must contain 4 (turn) or 5 (river) cards")
        cards = self.all_cards()
        if len(cards) != len(set(cards)):
            raise DuplicateCardError("ห้ามเลือกไพ่ซ้ำกัน")

    @classmethod
    def from_codes(cls, hands, board) -> "PokerTable":
        """สร้างโต๊ะจากรหัสไพ่ เช่น hands=[['AS','KD'], ...], board=['2C', ...]"""
        players = [
            Player(f"Player {i + 1}", [Card.from_code(c) for c in hand])
            for i, hand in enumerate(hands)
        ]
        return cls(players, [Card.from_code(c) for c in board])
