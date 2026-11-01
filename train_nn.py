"""เทรนโมเดล Neural Network: python train_nn.py"""
from core.sports import SportsModelTrainer


def main():
    trainer = SportsModelTrainer(data_dir=".", img_size=224, batch_size=32)
    trainer.load_data().build_model().train(epochs=10).fine_tune(epochs=5)
    accuracy = trainer.evaluate_and_save()
    print("Final Test Accuracy:", accuracy)


if __name__ == "__main__":
    main()
