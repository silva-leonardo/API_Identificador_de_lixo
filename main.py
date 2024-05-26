from fastapi import FastAPI, UploadFile, File
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import io
import os

app = FastAPI()

current_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.getenv('MODEL_PATH', os.path.join(current_dir, 'keras_model.h5'))
labels_path = os.getenv('LABELS_PATH', os.path.join(current_dir, 'labels.txt'))

model = load_model(model_path, compile=False)
class_names = open(labels_path, "r").readlines()


@app.get("/")
def read_root():
    return {"ESTADO": "A API esta funcionando corretamente"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    np.set_printoptions(suppress=True)

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data[0] = normalized_image_array

    prediction = model.predict(data)

    localLimpo = round(float(prediction[0][0]), 4)
    localSujo = round(float(prediction[0][1]), 4)

    return {"Local Sujo": localSujo, "Local Limpo": localLimpo}