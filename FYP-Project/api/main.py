from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from io import BytesIOsome
from PIL import Image
import tensorflow as tf


MODEL = tf.keras.models.load_model("../models/my_model.h5")
CLASS_NAMES = ["Early Blight","Late Blight","Healthy"]


app = FastAPI()


@app.get("/ping")
async def ping():
    return "hello I'm Alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image,0)
    prediction = MODEL.predict(image_batch)
    pass


if __name__ == "__main__":
    uvicorn.run(app=app, host="localhost", port=8000)
