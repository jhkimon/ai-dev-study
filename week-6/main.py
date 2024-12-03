from enum import Enum
from fastapi import FastAPI 
from pydantic import BaseModel
from model.model import SentimentModel

app = FastAPI()
model = SentimentModel()

class TextInput(BaseModel):
    text: str

# 예측할 때 /predict로 많이 씀
@app.post("/predict")
async def predict_sentiment(input: TextInput):
    prediction = model.predict(input.text)
    return {"text": input.text, "sentiment": prediction}

@app.get("/")
def root():
    return {"message" : "Hello World"}

@app.get("/home")
def home():
    return {"message" : "home"}

@app.get("/home/{name}")
def read_name(name: str):
    return {"name" : name}

@app.post("/")
def home_posst(msg: str):
    return {"Hello" : "POST", "msg" : msg}

