from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
# import keras as keras
from keras.models import load_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]

)

@app.get("/form")
def fromData(bedRooms,pin,houseType,furnishing,facing):
    features= np.array([[int(bedRooms),int(pin),int(houseType),int(furnishing),int(facing)]])
    m = load_model('my_model.h5')
    result=m.predict(features)
    return {'predictedPrice': str(result[0][0])}