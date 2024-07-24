import numpy as np
from fastapi import FastAPI, File, UploadFile
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf

MODEL = tf.keras.models.load_model("../models/1")
CLASS_NAME = ["Early Blight", "Late Blight", "Healthy"]


app = FastAPI()


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.get("/ping")
async def ping():
    return "hello"


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    prediction = MODEL.predict(img_batch)
    predicted_class = CLASS_NAME[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return {
        "Class": predicted_class,
        "Confidence":float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
