"""ตัวทำนายผู้ชนะ: แบบกฎ (River) และแบบ Machine Learning (Turn)"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from core.encoders import CardEncoder, PresenceEncoder
from core.models import EnsembleClassifier
from core.poker import HandEvaluator, PokerTable


@dataclass(frozen=True)
class PredictionResult:
    """ผลการทำนาย: ความน่าจะเป็นที่ผู้เล่นแต่ละคนจะชนะ และวิธีที่ใช้"""

    probabilities: tuple
    method: str

    @property
    def percentages(self) -> np.ndarray:
        return np.round(np.array(self.probabilities) * 100, 2)

    @property
    def winner_index(self) -> int:
        return int(np.argmax(self.probabilities))


class WinnerPredictor(ABC):
    """โครงสร้างกลางของตัวทำนาย ทุกคลาสลูกต้องมี predict(table)"""

    @property
    @abstractmethod
    def method_name(self) -> str:
        ...

    @abstractmethod
    def predict(self, table: PokerTable) -> PredictionResult:
        ...


class RuleBasedPredictor(WinnerPredictor):
    """เมื่อเปิด River แล้ว ผลแพ้ชนะแน่นอน จึงคำนวณด้วยกฎของโป๊กเกอร์"""

    def __init__(self, evaluator: HandEvaluator = None):
        self.__evaluator = evaluator or HandEvaluator()

    @property
    def method_name(self) -> str:
        return "Rule-Based (River)"

    def predict(self, table: PokerTable) -> PredictionResult:
        if not table.has_river:
            raise ValueError("Rule-based prediction needs the river card")
        scores = [self.__evaluator.evaluate_7(list(p.hole_cards) + list(table.board))
                  for p in table.players]
        best = max(scores)
        winners = [i for i, s in enumerate(scores) if s == best]
        probs = [1 / len(winners) if i in winners else 0.0 for i in range(len(scores))]
        return PredictionResult(tuple(probs), self.method_name)


class MLPredictor(WinnerPredictor):
    """ช่วง Turn ยังไม่รู้ไพ่ใบสุดท้าย จึงให้โมเดล Ensemble ประมาณความน่าจะเป็น"""

    def __init__(self, ensemble: EnsembleClassifier, encoder: CardEncoder = None):
        self.__ensemble = ensemble
        self.__encoder = encoder or PresenceEncoder()

    @property
    def method_name(self) -> str:
        return "Ensemble ML (Turn)"

    def predict(self, table: PokerTable) -> PredictionResult:
        X = self.__encoder.encode_table(table)
        probs = self.__ensemble.predict_proba(X)[0]
        return PredictionResult(tuple(float(p) for p in probs), self.method_name)


class PokerAIService:
    """เลือกตัวทำนายที่เหมาะกับสถานการณ์บนโต๊ะ

    ml_loader เป็นฟังก์ชันที่คืน MLPredictor ใช้โหลดโมเดลเฉพาะตอนจำเป็น (lazy loading)
    """

    def __init__(self, rule_predictor: WinnerPredictor, ml_loader):
        self.__rule_predictor = rule_predictor
        self.__ml_loader = ml_loader

    def choose_predictor(self, table: PokerTable) -> WinnerPredictor:
        return self.__rule_predictor if table.has_river else self.__ml_loader()

    def predict(self, table: PokerTable) -> PredictionResult:
        # Polymorphism: ไม่ว่าได้ตัวทำนายชนิดใด ก็เรียก predict() เหมือนกัน
        return self.choose_predictor(table).predict(table)
