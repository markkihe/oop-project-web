"""สร้างชุดข้อมูลโป๊กเกอร์ใหม่: python gen.py"""
from core.dataset import PokerDatasetGenerator

if __name__ == "__main__":
    generator = PokerDatasetGenerator(n_samples=100_000, mode="river")
    generator.save("poker_dataset100k.csv")
    print("Dataset Done!")
