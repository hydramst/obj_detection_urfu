import image_detection_model as fl
from fastapi import FastAPI
from pydantic import BaseModel

class Item(BaseModel):
    link: str

app = FastAPI()

@app.get("/")
async def root():
    return "This is a image detection model"

@app.post("/predict/")
def predict(item: Item):
    return fl.detect(item.link)

