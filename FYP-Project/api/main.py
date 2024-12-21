import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from io import BytesIO
from PIL import Image
import requests

# class names for each model
CLASS_NAME = ["Banana", "Cotton", "Mango", "Rice", "Sugarcane", "Wheat"]
DISEASE_CLASSES_RICE = ["Bacterial Leaf Blight", "Brown Spot", "Healthy", "Leaf Blast", "Leaf Scald", "Narrow Brown Spot", "Neck Blast", "Rice Hispa", "Sheath Blight", "Tungro"]
DISEASE_CLASSES_COTTON = ["Alphids", "Army worm", "Bacterial blight", "Healthy", "Powdery mildew", "Target spot"]
DISEASE_CLASSES_SUGARCANE = ["Banded Chlorosis", "Brown Spot", "Brown Rust", "Dried Leaves", "Grassy Shoot", "Healthy Leaves", "Pokkah Boeng", "Sett Rot", "Smut", "Viral Disease", "Yellow Leaf"]
DISEASE_CLASSES_WHEAT = ["Healthy", "Septoria", "Stripe Rust"]
DISEASE_CLASSES_BANANA = [ "Cordana", "Healthy", "Panama Disease", "Yellow and Black Sigatoka"]
DISEASE_CLASSES_MANGO = ["Anthracnose", "Bacterial Canker", "Cutting Weevil","Die Back", "Gall Midge", "Healthy", "Powdery Mildew", "Sooty Mould",]

DISEASE_CLASSES = {
    "Rice": DISEASE_CLASSES_RICE,
    "Cotton": DISEASE_CLASSES_COTTON,
    "Sugarcane": DISEASE_CLASSES_SUGARCANE,
    "Wheat": DISEASE_CLASSES_WHEAT,
    "Banana": DISEASE_CLASSES_BANANA,
    "Mango": DISEASE_CLASSES_MANGO,
}

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

# Endpoints for models
end_point_leaf_classifier = "http://localhost:8501/v1/models/leaf_classifier_model:predict"
end_point_rice = "http://localhost:8501/v1/models/rice_model:predict"
end_point_cotton = "http://localhost:8501/v1/models/cotton_model:predict"
end_point_sugarcane = "http://localhost:8501/v1/models/sugarcane_model:predict"
end_point_wheat = "http://localhost:8501/v1/models/wheat_model:predict"
end_point_banana = "http://localhost:8501/v1/models/banana_model:predict"
end_point_mango = "http://localhost:8501/v1/models/mango_model:predict"

# For Reading the image
def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

# Ping route for testing purpose
@app.get("/ping")
async def ping():
    return "hello"

# Predict route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    json_data = {"instances": img_batch.tolist()}

    # Step 1: Predict plant type using leaf classifier
    response = requests.post(end_point_leaf_classifier, json=json_data)
    leaf_prediction = np.array(response.json()["predictions"][0])
    predicted_leaf_class = CLASS_NAME[np.argmax(leaf_prediction)]

    if predicted_leaf_class not in CLASS_NAME:
        return {"error": "Leaf type not recognized"}

    if predicted_leaf_class == "Rice":
        endpoint = end_point_rice
        disease_classes = DISEASE_CLASSES_RICE
    elif predicted_leaf_class == "Cotton":
        endpoint = end_point_cotton
        disease_classes = DISEASE_CLASSES_COTTON
    elif predicted_leaf_class == "Sugarcane":
        endpoint = end_point_sugarcane
        disease_classes = DISEASE_CLASSES_SUGARCANE
    elif predicted_leaf_class == "Wheat":
        endpoint = end_point_wheat
        disease_classes = DISEASE_CLASSES_WHEAT
    elif predicted_leaf_class == "Banana":
        endpoint = end_point_banana
        disease_classes = DISEASE_CLASSES_BANANA
    elif predicted_leaf_class == "Mango":
        endpoint = end_point_mango
        disease_classes = DISEASE_CLASSES_MANGO

    response = requests.post(endpoint, json=json_data)
    prediction = np.array(response.json()["predictions"][0])
    predicted_class = disease_classes[np.argmax(prediction)]
    confidence = np.max(prediction)

    return {
        "Plant Type": predicted_leaf_class,
        "Class": predicted_class,
        "Confidence": float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
