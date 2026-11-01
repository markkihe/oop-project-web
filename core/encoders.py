"""ตัวแปลงไพ่บนโต๊ะให้เป็นเวกเตอร์ตัวเลขสำหรับโมเดล Machine Learning"""
from abc import ABC, abstractmethod

import numpy as np

from core.cards import Deck
from core.poker import PokerTable


class CardEncoder(ABC):
    """โครงสร้างกลางของตัวแปลงไพ่ คลาสลูกต้องกำหนดจำนวน feature และวิธี encode เอง"""

    def __init__(self, deck: Deck = None):
        self._deck = deck or Deck()

    @property
    @abstractmethod
    def n_features(self) -> int:
        ...

    @abstractmethod
    def encode(self, players, board) -> np.ndarray:
        """คืนเวกเตอร์ 1 มิติ ความยาว n_features"""

    def encode_table(self, table: PokerTable) -> np.ndarray:
        """แปลงโต๊ะ (ใช้เฉพาะไพ่ช่วง Turn) เป็นอาร์เรย์รูป (1, n_features) พร้อมส่งเข้าโมเดล"""
        return self.encode(table.players, table.turn_board).reshape(1, -1)


class PresenceEncoder(CardEncoder):
    """One-Hot 52 ช่อง: 1 = ไพ่ใบนี้อยู่บนโต๊ะ (ใช้กับโมเดล .pkl ชุดปัจจุบัน)"""

    @property
    def n_features(self) -> int:
        return len(self._deck)

    def encode(self, players, board) -> np.ndarray:
        vec = np.zeros(self.n_features)
        for player in players:
            for card in player.hole_cards:
                vec[self._deck.index_of(card)] = 1
        for card in board:
            vec[self._deck.index_of(card)] = 1
        return vec


class PerPlayerEncoder(CardEncoder):
    """One-Hot แยกกลุ่ม: 52 ช่องต่อผู้เล่นแต่ละคน + 52 ช่องสำหรับไพ่กลาง

    โมเดลจึงรู้ว่าไพ่ใบไหนเป็นของผู้เล่นคนใด (ต้องเทรนโมเดลใหม่ก่อนใช้งาน)
    """

    def __init__(self, n_players: int = 3, deck: Deck = None):
        super().__init__(deck)
        self.__n_players = n_players

    @property
    def n_features(self) -> int:
        return len(self._deck) * (self.__n_players + 1)

    def encode(self, players, board) -> np.ndarray:
        size = len(self._deck)
        vec = np.zeros(self.n_features)
        for p, player in enumerate(players):
            for card in player.hole_cards:
                vec[p * size + self._deck.index_of(card)] = 1
        board_offset = self.__n_players * size
        for card in board:
            vec[board_offset + self._deck.index_of(card)] = 1
        return vec
