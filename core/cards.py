"""คลาสพื้นฐานของไพ่: Card (ไพ่ 1 ใบ) และ Deck (สำรับ 52 ใบ)"""
import random


class Card:
    """ไพ่ 1 ใบ เก็บแต้ม (rank) และดอก (suit) เป็นข้อมูลแบบ private"""

    RANKS = "23456789TJQKA"
    SUITS = "SHDC"
    SUIT_SYMBOLS = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}

    def __init__(self, rank: str, suit: str):
        rank, suit = rank.upper(), suit.upper()
        if rank not in self.RANKS:
            raise ValueError(f"Invalid rank: {rank}")
        if suit not in self.SUITS:
            raise ValueError(f"Invalid suit: {suit}")
        self.__rank = rank
        self.__suit = suit

    @classmethod
    def from_code(cls, code: str) -> "Card":
        """สร้าง Card จากรหัสสองตัวอักษร เช่น 'AS', 'TD'"""
        if len(code) != 2:
            raise ValueError(f"Invalid card code: {code}")
        return cls(code[0], code[1])

    @property
    def rank(self) -> str:
        return self.__rank

    @property
    def suit(self) -> str:
        return self.__suit

    @property
    def value(self) -> int:
        """ค่าแต้มเป็นตัวเลข 0 (2) ถึง 12 (A)"""
        return self.RANKS.index(self.__rank)

    @property
    def code(self) -> str:
        return self.__rank + self.__suit

    @property
    def symbol(self) -> str:
        """ข้อความสำหรับแสดงบนหน้าจอ เช่น 'A♠'"""
        return f"{self.__rank}{self.SUIT_SYMBOLS[self.__suit]}"

    def __eq__(self, other) -> bool:
        return isinstance(other, Card) and self.code == other.code

    def __hash__(self) -> int:
        return hash(self.code)

    def __repr__(self) -> str:
        return f"Card('{self.code}')"

    def __str__(self) -> str:
        return self.code


class Deck:
    """สำรับไพ่ 52 ใบ เรียงลำดับ rank ก่อน suit (ตรงกับการ encode ของโมเดลเดิม)"""

    def __init__(self):
        self.__cards = [Card(r, s) for r in Card.RANKS for s in Card.SUITS]
        self.__index = {card: i for i, card in enumerate(self.__cards)}

    def __len__(self) -> int:
        return len(self.__cards)

    def __iter__(self):
        return iter(self.__cards)

    @property
    def cards(self) -> list:
        return list(self.__cards)

    def index_of(self, card: Card) -> int:
        """ตำแหน่งของไพ่ในสำรับ (0-51) ใช้ทำ One-Hot Encoding"""
        return self.__index[card]

    def shuffled(self, rng: random.Random = None) -> list:
        """คืนสำเนาไพ่ที่สับแล้ว โดยไม่แก้ลำดับของสำรับต้นฉบับ"""
        cards = list(self.__cards)
        (rng or random).shuffle(cards)
        return cards

    def display_map(self) -> dict:
        """{'A♠': Card('AS'), ...} ใช้ทำตัวเลือกใน selectbox"""
        return {card.symbol: card for card in self.__cards}
