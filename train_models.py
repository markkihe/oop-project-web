"""เทรนโมเดล ML ทั้ง 3 ตัว: python train_models.py"""
from core.dataset import PokerDataset
from core.encoders import PresenceEncoder
from core.models import LogisticModel, RandomForestModel, XGBoostModel


def main():
    dataset = PokerDataset("poker_dataset100k.csv", encoder=PresenceEncoder())
    X_train, X_test, y_train, y_test = dataset.split(test_size=0.2, random_state=42)

    for model in (LogisticModel(), RandomForestModel(), XGBoostModel()):
        model.fit(X_train, y_train)
        print(f"{model.name} Accuracy: {model.score(X_test, y_test):.4f}")
        model.save()

    print("Models saved successfully!")


if __name__ == "__main__":
    main()
