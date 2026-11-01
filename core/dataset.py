"""การสร้างและโหลดชุดข้อมูลโป๊กเกอร์"""
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from core.cards import Card, Deck
from core.encoders import CardEncoder, PresenceEncoder
from core.poker import HandEvaluator, Player, PokerTable

COLUMNS = [
    "P1_1", "P1_2", "P2_1", "P2_2", "P3_1", "P3_2",
    "F1", "F2", "F3", "Turn", "River",
    "P1_prob", "P2_prob", "P3_prob",
]
LABEL_COLUMNS = ["P1_prob", "P2_prob", "P3_prob"]


class PokerDatasetGenerator:
    """จำลองเกมโป๊กเกอร์ 3 ผู้เล่นแบบสุ่ม แล้วคำนวณความน่าจะเป็นที่แต่ละคนจะชนะ"""

    MODES = ("river", "turn")

    def __init__(self, n_samples: int = 100_000, mode: str = "river",
                 evaluator: HandEvaluator = None, seed: int = None):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.__n_samples = n_samples
        self.__mode = mode
        self.__evaluator = evaluator or HandEvaluator()
        self.__deck = Deck()
        self.__rng = random.Random(seed)

    def generate_game(self) -> list:
        cards = self.__deck.shuffled(self.__rng)
        hands = [cards[0:2], cards[2:4], cards[4:6]]

        if self.__mode == "river":
            board = cards[6:11]
            probs = self.__showdown(hands, board)
            board_codes = [c.code for c in board]
        else:
            board = cards[6:10]
            probs = self.__exact_turn_probability(hands, board)
            board_codes = [c.code for c in board] + ["NA"]

        hand_codes = [c.code for hand in hands for c in hand]
        return hand_codes + board_codes + probs

    def generate(self) -> pd.DataFrame:
        rows = [self.generate_game() for _ in range(self.__n_samples)]
        return pd.DataFrame(rows, columns=COLUMNS)

    def save(self, path: str = "poker_dataset100k.csv") -> pd.DataFrame:
        df = self.generate()
        df.to_csv(path, index=False)
        return df

    def __showdown(self, hands, board) -> list:
        scores = [self.__evaluator.evaluate_7(hand + board) for hand in hands]
        best = max(scores)
        winners = [i for i, s in enumerate(scores) if s == best]
        return [1 / len(winners) if i in winners else 0.0 for i in range(len(hands))]

    def __exact_turn_probability(self, hands, board4) -> list:
        used = set(c for hand in hands for c in hand) | set(board4)
        remaining = [c for c in self.__deck if c not in used]
        wins = np.zeros(len(hands))
        for river in remaining:
            wins += self.__showdown(hands, board4 + [river])
        return list(wins / len(remaining))


class PokerDataset:
    """โหลดไฟล์ CSV แล้วแปลงเป็น X (feature) และ y (ผู้ชนะ 0/1/2) ด้วย encoder ที่กำหนด"""

    def __init__(self, csv_path: str = "poker_dataset100k.csv", encoder: CardEncoder = None):
        self.__csv_path = csv_path
        self.__encoder = encoder or PresenceEncoder()
        self.__df = None

    @property
    def dataframe(self) -> pd.DataFrame:
        if self.__df is None:
            self.__df = pd.read_csv(self.__csv_path)
        return self.__df

    def row_to_table(self, row) -> PokerTable:
        """แถวข้อมูล 1 แถว -> PokerTable ช่วง Turn (ไพ่ผู้เล่น 6 ใบ + ไพ่กลาง 4 ใบ)"""
        cards = [Card.from_code(code) for code in row[:10]]
        players = [Player(f"Player {i + 1}", cards[2 * i:2 * i + 2]) for i in range(3)]
        return PokerTable(players, cards[6:10])

    def features(self) -> np.ndarray:
        return np.vstack([
            self.__encoder.encode_table(self.row_to_table(row))
            for row in self.dataframe.values
        ])

    def labels(self) -> np.ndarray:
        return self.dataframe[LABEL_COLUMNS].values.argmax(axis=1)

    def split(self, test_size: float = 0.2, random_state: int = 42):
        return train_test_split(self.features(), self.labels(),
                                test_size=test_size, random_state=random_state)
