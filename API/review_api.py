from fastapi import FastAPI
# test

# Souhaiebqsdsdssdssdsdsd
# welyeysdsdsdsdssdsdsdsdsd

from pydantic import BaseModel
from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
import tensorflow as tf
import uvicorn

# Creating FastApi instance
app = FastAPI()


# Initiliazing input
class Item(BaseModel):
    text: str


@app.post("/")
async def root(item: Item):
    text = item.text

    tokenizer, model = import_model("../Model/finetuned_distilled_bert")
    prediction = classify_text(text, tokenizer, model)

    return prediction


def import_model(save_directory):
    """
    Function takes directory and returns tokenizer and model
    """
    # Loadind model

    tokenizer = DistilBertTokenizer.from_pretrained(save_directory)
    model = TFDistilBertForSequenceClassification.from_pretrained(
        save_directory)

    return (tokenizer, model)


def classify_text(text, tokenizer, model):
    """
    Function classifys given text into Positive/Neutral/Negative
    """
    # Predicting

    # Tokenizing text
    predict_input = tokenizer.encode(
        text,
        truncation=True,
        padding=True,
        return_tensors="tf"
    )

    output = model(predict_input)[0]

    prediction_value = tf.argmax(output, axis=1).numpy()[0]

    if prediction_value == 0:
        return "Negative"
    if prediction_value == 1:
        return "Neutral"
    return "Positive"


# Instead of running each time from console
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
