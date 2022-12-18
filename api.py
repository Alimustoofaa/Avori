'''''
@author     : Ali Mustofa HALOTEC
@module     : Service Models OCR Container Number
@Created on : 29 Nov 2022
'''
import os
import io
import json
import uvicorn
from PIL import Image
import numpy as np
from src.process import main_process
from fastapi import FastAPI,File

app = FastAPI()

@app.get("/")
def home():
	return {"msg": "Avocado ripe classification"}

@app.post('/fruit')
def fruit_ripe_classification(file: bytes = File(...)):
    image = Image.open(io.BytesIO(file)).convert("RGB")
    image = np.array(image)
    image = image[:,:,::-1].copy()
    result = main_process(
        image
    )
    result_json = result.json()
    return json.loads(result_json)

if __name__ == '__main__':
	uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=True)


# not_ripe = ["Medium Sea Green", "Viridian", "Medium Jungle Green", "Hooker'S Green", ]
# almost_ripe = ["Aurometalsaurus", "Cadet Blue", "Viridian", "Payne'S Grey"]
# almost/ripe = Hunter Green, Slate Gray
# ripe = ["Ucla Blue", "Cadet Grey", "Dark Slate Gray", "Aurometalsaurus", "Cadet Blue", "Rackley"]
# overripe = ["Dark Jungle Green", "Payne'S Grey", "Aurometalsaurus", "Purple Taupe", "Dim Gray", ]