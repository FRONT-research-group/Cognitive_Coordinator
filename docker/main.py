import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from src.inference import ClassificationComponentWrapper

app = FastAPI(docs_url='/cc-docs')
cc = ClassificationComponentWrapper()

# Data storage (for demonstration purposes, replace with a database if needed)
stored_data = []
calculating = False
clotw_score = None
nlotw_score = None

class TextInput(BaseModel):
    text: str
    label: str

class DataInput(BaseModel):
    text: List[TextInput]

@app.post("/submit_data/")
def submit_data(data: DataInput):
    global stored_data, calculating
    stored_data.extend(data.text)
    calculating = True  
    return {"message": "Data received successfully", "count": len(stored_data)}

@app.post("/calculate_clotw/")
def calculate_clotw():
    global calculating, clotw_score, nlotw_score
    if not stored_data:
        raise HTTPException(status_code=404, detail="No data available")

    nlotw_score = cc.caclulate_nlotw(stored_data)
    clotw_score = cc.calculate_clotw(stored_data)

    calculating = False  # Set to False only after computing cLoTw
    return {"message": "cLoTw calculated successfully", "cLoTw": clotw_score}

@app.get("/get_clotw/")
def get_clotw():
    if clotw_score is None:
        raise HTTPException(status_code=404, detail="cLoTw not yet calculated")
    return {"cLoTw": clotw_score}

@app.get("/get_nlotw/")
def get_nlotw():
    if nlotw_score is None:
        raise HTTPException(status_code=404, detail="nLoTw not yet calculated")
    return {"cLoTw": nlotw_score}

@app.get("/status/")
def get_status():
    return {"calculating": calculating, "data_count": len(stored_data)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6464)