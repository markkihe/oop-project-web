"""หน้าที่แสดงข้อมูลอย่างเดียว: หน้าแรก, หน้าอธิบาย ML และหน้าอธิบาย NN"""
from ui.base_page import Section, SectionPage


class HomePage(SectionPage):
    def __init__(self):
        super().__init__(
            title="Sports / Poker AI Project",
            icon="🏆",
            divider_between_sections=True,
            footer_message="🚀 Use the sidebar to explore model explanations and testing pages.",
            sections=[
                Section("🎯 Project Objective", """
This project compares two types of Artificial Intelligence models:

- Machine Learning (Ensemble Model)
- Neural Network (Deep Learning)

The objective is to demonstrate how different AI approaches
handle structured and unstructured datasets.
"""),
                Section("📂 Datasets Used", """
### 🎴 Poker Dataset
- Type: Structured Data
- Source: Generated using ChatGPT
- Task: Predict winning player

### 🏟 Sports Image Dataset
- Type: Unstructured Data (Images)
- Source: Kaggle
- Task: Classify sports category
"""),
                Section("🧠 Models Implemented", """
### 1️⃣ Ensemble Machine Learning
- Logistic Regression
- Random Forest
- XGBoost

### 2️⃣ Neural Network
- EfficientNetB0 (Transfer Learning)
"""),
                Section("⚖ Model Comparison", """
| Model Type | Data Type | Strength |
|------------|-----------|----------|
| Ensemble ML | Structured | Fast, stable, interpretable |
| Neural Network | Image | High accuracy, feature extraction |
"""),
            ],
        )


class MLExplainPage(SectionPage):
    def __init__(self):
        super().__init__(
            title="Machine Learning Model Explanation",
            icon="📊",
            page_title="ML Explain",
            sections=[
                Section("1️⃣ Dataset Information", """
### 🎴 Poker Simulation Dataset

- **Type:** Structured Dataset
- **Source:** Artificially generated using ChatGPT for simulation purposes
- **Domain:** Texas Hold'em Poker

The dataset was generated to simulate multiple poker game scenarios
including different player hands and community card combinations.

Each record contains:

- 2 cards per player (3 players total)
- Community cards (Flop, Turn, River)
- Target label: Winning player (Player 1, 2, or 3)

The dataset includes intentionally diverse and imperfect combinations
to simulate real-world variability.
"""),
                Section("2️⃣ Feature Description", """
Each card is encoded using One-Hot Encoding:

- Total features: 52 binary features
- Each index represents a unique card in the deck
- Value = 1 if card is present, 0 otherwise

This structured format allows compatibility with traditional ML models.
"""),
                Section("3️⃣ Data Preparation Process", """
Steps performed:

1. Card indexing (0–51 mapping)
2. One-hot encoding transformation
3. Duplicate checking
4. Train-test splitting
5. Label encoding for classification
"""),
                Section("4️⃣ Ensemble Model Development", """
This project uses an Ensemble Machine Learning approach:

- Logistic Regression
- Random Forest
- XGBoost

Final prediction is computed by averaging probabilities:

Final Probability = (P_log + P_rf + P_xgb) / 3

The player with the highest averaged probability is selected as the winner.
"""),
                Section("5️⃣ Algorithm Theory", """
- **Logistic Regression:** Linear probabilistic classifier
- **Random Forest:** Ensemble of decision trees
- **XGBoost:** Gradient boosting framework

Ensemble learning improves model stability and reduces overfitting.
"""),
                Section("6️⃣ References", """
- Dataset generated using ChatGPT (OpenAI)
- Scikit-learn Documentation
- XGBoost Documentation
"""),
            ],
        )


class NNExplainPage(SectionPage):
    def __init__(self):
        super().__init__(
            title="Neural Network Model Explanation",
            icon="🧠",
            page_title="NN Explain",
            footer_message="🚀 This Neural Network model applies transfer learning for image classification.",
            sections=[
                Section("1️⃣ Dataset Information", """
### 🏟 Sports Image Classification Dataset

- **Type:** Unstructured Dataset (Image Data)
- **Source:** Kaggle
- **Link:** https://www.kaggle.com/datasets/gpiosenka/sports-classification

The dataset contains multiple sports categories
with labeled images for supervised learning.

Each class contains several hundred images.
"""),
                Section("2️⃣ Data Preparation Process", """
Steps performed:

1. Image resizing to 224x224 pixels
2. RGB conversion
3. Normalization using EfficientNet preprocess_input
4. Train-validation split
5. Optional data augmentation
"""),
                Section("3️⃣ Model Architecture", """
The model is based on EfficientNetB0 (Pretrained on ImageNet).

Architecture:

- EfficientNetB0 base model
- Global Average Pooling
- Dense layer with Softmax activation

Transfer learning was applied to leverage pretrained features.
"""),
                Section("4️⃣ Neural Network Theory", """
Convolutional Neural Networks (CNN) extract spatial features
from image data using convolution and pooling layers.

Softmax activation is used for multi-class classification.
"""),
                Section("5️⃣ References", """
- Kaggle Sports Classification Dataset
  https://www.kaggle.com/datasets/gpiosenka/sports-classification
- TensorFlow Documentation
- EfficientNet Paper (Tan & Le, 2019)
"""),
            ],
        )
