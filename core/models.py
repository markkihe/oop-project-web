"""โมเดล Machine Learning แต่ละชนิด และโมเดลรวม (Ensemble)"""
from abc import ABC, abstractmethod

import joblib
import numpy as np
from sklearn.metrics import accuracy_score


class BaseClassifier(ABC):
    """คลาสแม่ของโมเดลทุกชนิด ห่อหุ้มโมเดลจริงของไลบรารีไว้ใน self._model"""

    DEFAULT_PATH = None

    def __init__(self, model=None):
        self._model = model if model is not None else self._create_model()

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def _create_model(self):
        """สร้างโมเดลเปล่าพร้อมไฮเปอร์พารามิเตอร์ของแต่ละชนิด"""

    def fit(self, X, y) -> "BaseClassifier":
        self._model.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self._model.predict_proba(X)

    def predict(self, X) -> np.ndarray:
        return self._model.predict(X)

    def score(self, X, y) -> float:
        return accuracy_score(y, self.predict(X))

    def save(self, path: str = None):
        # บันทึกเฉพาะโมเดลของไลบรารี ไฟล์ .pkl จึงใช้ร่วมกับโค้ดเดิมได้
        joblib.dump(self._model, path or self.DEFAULT_PATH)

    @classmethod
    def load(cls, path: str = None) -> "BaseClassifier":
        return cls(model=joblib.load(path or cls.DEFAULT_PATH))


class LogisticModel(BaseClassifier):
    DEFAULT_PATH = "log_model.pkl"

    @property
    def name(self) -> str:
        return "Logistic Regression"

    def _create_model(self):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000)


class RandomForestModel(BaseClassifier):
    DEFAULT_PATH = "rf_model.pkl"

    @property
    def name(self) -> str:
        return "Random Forest"

    def _create_model(self):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )


class XGBoostModel(BaseClassifier):
    DEFAULT_PATH = "xgb_model.pkl"

    @property
    def name(self) -> str:
        return "XGBoost"

    def _create_model(self):
        from xgboost import XGBClassifier
        return XGBClassifier(eval_metric="mlogloss")


class EnsembleClassifier:
    """รวมโมเดลหลายตัว (Composition) แล้วหาค่าเฉลี่ยความน่าจะเป็น"""

    def __init__(self, members):
        self.__members = list(members)
        if not self.__members:
            raise ValueError("Ensemble needs at least one model")

    @property
    def members(self) -> list:
        return list(self.__members)

    def predict_proba(self, X) -> np.ndarray:
        # Polymorphism: เรียก predict_proba() ของโมเดลแต่ละชนิดผ่านชื่อเมธอดเดียวกัน
        return np.mean([m.predict_proba(X) for m in self.__members], axis=0)

    @classmethod
    def load_default(cls) -> "EnsembleClassifier":
        return cls([LogisticModel.load(), RandomForestModel.load(), XGBoostModel.load()])
