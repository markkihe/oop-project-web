"""หน้าทดสอบโมเดล: Poker (ML) และ Sports Image Classifier (NN)"""
import streamlit as st
from PIL import Image

from core.cards import Deck
from core.models import EnsembleClassifier
from core.poker import DuplicateCardError, HandEvaluator, Player, PokerTable
from core.predictors import MLPredictor, PokerAIService, RuleBasedPredictor
from core.sports import SportsImageClassifier
from ui.base_page import BasePage


@st.cache_resource(show_spinner="🔄 Loading ML models...")
def load_ml_predictor() -> MLPredictor:
    return MLPredictor(EnsembleClassifier.load_default())


@st.cache_resource(show_spinner="🤖 Loading AI model...")
def load_sports_classifier() -> SportsImageClassifier:
    return SportsImageClassifier()


class CardSelector:
    """ชุด selectbox สำหรับเลือกไพ่ แปลงข้อความที่เลือกกลับเป็น Card"""

    NOT_OPENED = "🚫 Not Opened"

    def __init__(self, deck: Deck):
        self.__display_map = deck.display_map()
        self.__options = list(self.__display_map)

    def select(self, label: str, optional: bool = False):
        options = [self.NOT_OPENED] + self.__options if optional else self.__options
        choice = st.selectbox(label, options)
        return None if choice == self.NOT_OPENED else self.__display_map[choice]


class PokerTestPage(BasePage):
    N_PLAYERS = 3

    def __init__(self):
        super().__init__(title="Poker 3 Players AI System", icon="♠", page_title="Poker AI")
        self.__selector = CardSelector(Deck())
        self.__service = PokerAIService(RuleBasedPredictor(HandEvaluator()), load_ml_predictor)

    def render_header(self):
        st.markdown(f"# {self._icon} {self._title}")

    def render_body(self):
        hands = self.__select_hands()
        board = self.__select_board()

        if st.button("🎯 Evaluate / Predict"):
            try:
                players = [Player(f"Player {i + 1}", hand) for i, hand in enumerate(hands)]
                table = PokerTable(players, board)
                result = self.__service.predict(table)
            except DuplicateCardError:
                st.error("❌ ห้ามเลือกไพ่ซ้ำกัน")
                return
            except FileNotFoundError as e:
                st.error(f"❌ {e}")
                return
            self.__show_result(result)

    def __select_hands(self) -> list:
        st.markdown("## 🎴 Players")
        hands = []
        for i, col in enumerate(st.columns(self.N_PLAYERS)):
            with col:
                hands.append([self.__selector.select(f"P{i + 1} Card {k}") for k in (1, 2)])
        return hands

    def __select_board(self) -> list:
        st.markdown("## 🃏 Board")
        labels = ["Flop 1", "Flop 2", "Flop 3", "Turn"]
        cols = st.columns(5)
        board = []
        for label, col in zip(labels, cols):
            with col:
                board.append(self.__selector.select(label))
        with cols[4]:
            river = self.__selector.select("River", optional=True)
        return board + ([river] if river is not None else [])

    def __show_result(self, result):
        st.markdown("## 🏆 Result")
        st.caption(f"Method: {result.method}")
        percents = result.percentages
        for i, col in enumerate(st.columns(self.N_PLAYERS)):
            with col:
                text = f"Player {i + 1}\n\n{percents[i]} %"
                if i == result.winner_index:
                    st.success(f"🎉 {text}")
                else:
                    st.write(text)


class SportsClassifierPage(BasePage):
    def __init__(self):
        super().__init__(title="AI Sports Image Classifier", icon="⚽", page_title="AI Sports Vision")

    def render_header(self):
        st.title(f"🧠{self._icon} {self._title}")

    def render_body(self):
        uploaded = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])
        if not uploaded:
            return

        image = Image.open(uploaded).convert("RGB")
        try:
            classifier = load_sports_classifier()
            with st.spinner("🤖 AI is analyzing the image..."):
                prediction = classifier.predict(image, top_k=3)
        except Exception as e:  # ไฟล์โมเดลหาย / TensorFlow โหลดไม่ได้
            st.error(f"❌ Model Load Error: {e}")
            return

        _, center, _ = st.columns([1, 2, 1])
        with center:
            st.image(image, caption="Uploaded Image", width=350)
        st.markdown("---")
        st.success(f"🎯 Predicted Sport: {prediction.label}")
        if not prediction.is_confident():
            st.warning("⚠ Model is not confident about this prediction.")
        st.info(f"Confidence: {prediction.confidence:.2f}%")

        st.markdown("### 🔎 Top 3 Predictions")
        for label, percent in prediction.top_k:
            st.write(f"{label} — {percent:.2f}%")
