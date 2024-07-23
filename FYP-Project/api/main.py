from fastapi import FastAPI, UploadFile, File
import uvicorn
app = FastAPI()
@app.get("/ping")
async def ping():
    return "hello I'm Alive"

@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    pass


if __name__ == "__main__":
    uvicorn.run(app=app, host="localhost", port=8000)
