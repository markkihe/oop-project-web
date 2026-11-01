"""ส่วน Neural Network: จำแนกประเภทกีฬาจากรูปภาพด้วย EfficientNetB0

TensorFlow ถูก import ภายในเมธอดเท่านั้น เพื่อให้แอปเปิดเร็วและไม่โหลดโมเดลจนกว่าจะใช้งานจริง
"""
import json
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SportsPrediction:
    label: str
    confidence: float          # เปอร์เซ็นต์ 0-100
    top_k: tuple               # ((label, percent), ...)

    def is_confident(self, threshold: float = 60.0) -> bool:
        return self.confidence >= threshold


class ImagePreprocessor:
    """เตรียมรูปให้อยู่ในรูปแบบที่ EfficientNet ต้องการ"""

    def __init__(self, img_size: int = 224):
        self.__img_size = img_size

    @property
    def img_size(self) -> int:
        return self.__img_size

    def prepare(self, image: Image.Image) -> np.ndarray:
        from tensorflow.keras.applications.efficientnet import preprocess_input
        image = image.convert("RGB").resize((self.__img_size, self.__img_size))
        array = preprocess_input(np.array(image))
        return np.expand_dims(array, axis=0)


class SportsImageClassifier:
    """โหลดโมเดล .h5 และรายชื่อคลาส แล้วทำนายประเภทกีฬาของรูป"""

    def __init__(self, model_path: str = "nn_model.h5",
                 class_index_path: str = "class_indices.json",
                 preprocessor: ImagePreprocessor = None):
        self.__model_path = model_path
        self.__class_index_path = class_index_path
        self.__preprocessor = preprocessor or ImagePreprocessor()
        self.__model = None
        self.__class_names = None

    @property
    def class_names(self) -> list:
        if self.__class_names is None:
            if not os.path.exists(self.__class_index_path):
                raise FileNotFoundError(f"class_indices.json not found: {self.__class_index_path}")
            with open(self.__class_index_path, "r") as f:
                class_indices = json.load(f)
            names = [None] * len(class_indices)
            for name, index in class_indices.items():
                names[index] = name
            self.__class_names = names
        return self.__class_names

    def predict(self, image: Image.Image, top_k: int = 3) -> SportsPrediction:
        scores = self.__get_model().predict(self.__preprocessor.prepare(image), verbose=0)[0]
        order = np.argsort(scores)[::-1][:top_k]
        top = tuple((self.class_names[i], float(scores[i]) * 100) for i in order)
        return SportsPrediction(label=top[0][0], confidence=top[0][1], top_k=top)

    def __get_model(self):
        if self.__model is None:
            if not os.path.exists(self.__model_path):
                raise FileNotFoundError(f"Model file not found: {self.__model_path}")
            import tensorflow as tf
            self.__model = tf.keras.models.load_model(self.__model_path, compile=False)
        return self.__model


class SportsModelTrainer:
    """ขั้นตอนเทรนโมเดล: เตรียมข้อมูล -> สร้างโมเดล -> เทรน -> Fine-tune -> ประเมินและบันทึก"""

    def __init__(self, data_dir: str = ".", img_size: int = 224, batch_size: int = 32,
                 checkpoint_path: str = "best_model.h5", output_path: str = "nn_model.h5"):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.output_path = output_path
        self._train = self._valid = self._test = None
        self._model = self._base_model = None

    def load_data(self, class_index_path: str = "class_indices.json"):
        from tensorflow.keras.applications.efficientnet import preprocess_input
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        augment = ImageDataGenerator(
            preprocessing_function=preprocess_input, rotation_range=20, zoom_range=0.2,
            horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1,
        )
        plain = ImageDataGenerator(preprocessing_function=preprocess_input)
        self._train = self.__flow(augment, "train")
        self._valid = self.__flow(plain, "valid")
        self._test = self.__flow(plain, "test")

        with open(class_index_path, "w") as f:
            json.dump(self._train.class_indices, f)
        return self

    def build_model(self):
        from tensorflow.keras.applications import EfficientNetB0
        from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
        from tensorflow.keras.models import Model

        self._base_model = EfficientNetB0(weights="imagenet", include_top=False,
                                          input_shape=(self.img_size, self.img_size, 3))
        self._base_model.trainable = False
        x = GlobalAveragePooling2D()(self._base_model.output)
        x = Dropout(0.5)(x)
        x = Dense(256, activation="relu")(x)
        output = Dense(self._train.num_classes, activation="softmax")(x)
        self._model = Model(inputs=self._base_model.input, outputs=output)
        return self

    def train(self, epochs: int = 10):
        self.__compile(1e-4)
        self._model.fit(self._train, validation_data=self._valid, epochs=epochs,
                        callbacks=self.__callbacks())
        return self

    def fine_tune(self, epochs: int = 5, unfreeze_last: int = 20):
        self._base_model.trainable = True
        for layer in self._base_model.layers[:-unfreeze_last]:
            layer.trainable = False
        self.__compile(1e-5)
        self._model.fit(self._train, validation_data=self._valid, epochs=epochs,
                        callbacks=self.__callbacks())
        return self

    def evaluate_and_save(self) -> float:
        import tensorflow as tf
        best = tf.keras.models.load_model(self.checkpoint_path, compile=False)
        best.compile(loss="categorical_crossentropy", metrics=["accuracy"])
        _, acc = best.evaluate(self._test)
        best.save(self.output_path)
        return acc

    def __flow(self, generator, folder):
        return generator.flow_from_directory(
            os.path.join(self.data_dir, folder),
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode="categorical",
        )

    def __compile(self, learning_rate):
        from tensorflow.keras.optimizers import Adam
        self._model.compile(optimizer=Adam(learning_rate), loss="categorical_crossentropy",
                            metrics=["accuracy"])

    def __callbacks(self):
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        return [
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, verbose=1),
            ModelCheckpoint(self.checkpoint_path, monitor="val_accuracy",
                            save_best_only=True, verbose=1, save_format="h5"),
        ]
