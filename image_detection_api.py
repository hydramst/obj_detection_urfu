import image_detection_model as fl
from fastapi import FastAPI
from pydantic import BaseModel

import requests
import io
import scipy.misc
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pickle

MODELS_path = 'https://tfhub.dev/tensorflow/centernet/resnet101v1_fpn_512x512/1'
hub_model = hub.load(MODELS_path) 

class Item(BaseModel):
    link: str

app = FastAPI()

@app.get("/")
async def root():
    return "This is an image detection model"

@app.post("/predict/")
def predict(item: Item):
    return fl.detect(item.link, hub_model)

