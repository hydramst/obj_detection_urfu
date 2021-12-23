#model - 'https://tfhub.dev/tensorflow/centernet/resnet101v1_fpn_512x512/1'

import image_detection_model as fl
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow_hub as hub

hub_model = hub.load('model/')

class Item(BaseModel):
    link: str

app = FastAPI()

@app.get("/")
async def root():
    return "This is an image detection model"

@app.post("/predict/")
def predict(item: Item):
    return fl.detect(item.link, hub_model)
