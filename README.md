# Sports / Poker AI Project (OOP Version)

LINK: [https://isproject-jzzcpzkkchfikr6k8lnndm.streamlit.app/](https://isproject-jzzcpzkkchfikr6k8lnndm.streamlit.app/)

เว็บแอป Streamlit เปรียบเทียบ AI 2 แบบ
- **Machine Learning (Ensemble)** ทำนายผู้ชนะโป๊กเกอร์ 3 ผู้เล่น (Logistic Regression + Random Forest + XGBoost)
- **Neural Network** จำแนกประเภทกีฬาจากรูปภาพ 100 ประเภท (EfficientNetB0 Transfer Learning)

## โครงสร้างโปรเจกต์

```
app.py                  หน้าแรก (HomePage)
pages/                  หน้าย่อยของ Streamlit (ไฟล์ละ 2 บรรทัด เรียกคลาสจาก ui/)
core/                   ตรรกะหลัก ไม่ขึ้นกับ Streamlit
  cards.py              Card, Deck
  poker.py              Player, HandEvaluator, PokerTable, DuplicateCardError
  encoders.py           CardEncoder (abstract), PresenceEncoder, PerPlayerEncoder
  models.py             BaseClassifier (abstract), LogisticModel, RandomForestModel,
                        XGBoostModel, EnsembleClassifier
  predictors.py         WinnerPredictor (abstract), RuleBasedPredictor, MLPredictor,
                        PokerAIService, PredictionResult
  dataset.py            PokerDatasetGenerator, PokerDataset
  sports.py             ImagePreprocessor, SportsImageClassifier, SportsModelTrainer,
                        SportsPrediction
ui/                     หน้าจอในรูปแบบคลาส
  base_page.py          BasePage (abstract), Section, SectionPage
  info_pages.py         HomePage, MLExplainPage, NNExplainPage
  test_pages.py         CardSelector, PokerTestPage, SportsClassifierPage
gen.py                  สร้างชุดข้อมูล
train_models.py         เทรนโมเดล ML
train_nn.py             เทรนโมเดล NN
tests/                  Unit test ของคลาสหลัก
```

## การรัน

```bash
pip install -r requirements.txt
streamlit run app.py
python -m pytest tests     # ต้องติดตั้ง pytest
```

ไฟล์โมเดลเดิม (`*.pkl`, `nn_model.h5`) ใช้ได้ทันทีโดยไม่ต้องเทรนใหม่
