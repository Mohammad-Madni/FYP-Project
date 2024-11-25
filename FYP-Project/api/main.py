import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from io import BytesIO
from PIL import Image
import requests

CLASS_NAME = ["Cordana", "Healthy", "Panama Disease", "Yellow and Black Sigatoka"]


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


end_point = "http://localhost:8501/v1/models/banana_model:predict"

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
    json_data = {
        "instances":img_batch.tolist()
    }
    response = requests.post(end_point,json=json_data)
    prediction = np.array(response.json()["predictions"][0])
    predicted_class = CLASS_NAME[np.argmax(prediction)]
    confidence = np.max(prediction)

    return {
        "Class": predicted_class,
        "Confidence": float(confidence)
    }



if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
