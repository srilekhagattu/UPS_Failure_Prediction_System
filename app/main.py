# app/main.py
from typing import List
from app.schemas import IsolationForestResultSchema, KMeansResultSchema
import traceback
from fastapi import HTTPException
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app import models
print(models.IsolationForestResult)


app = FastAPI()


@app.get("/")
def root():
    return {"message": "UPS Anomaly Detection API"}


@app.get("/isolation-forest/", response_model=List[IsolationForestResultSchema])
def get_isolation_forest_results(db: Session = Depends(get_db)):
    try:
        return db.query(models.IsolationForestResult).all()
    except Exception as e:
        traceback.print_exc()  # shows full error in terminal
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/kmeans/", response_model=List[KMeansResultSchema])
def get_kmeans_results(db: Session = Depends(get_db)):
    try:
        return db.query(models.KMeansResult).all()
    except Exception as e:
        import traceback
        traceback.print_exc()  # ✅ This will show the actual error in the terminal
        raise HTTPException(status_code=500, detail=str(e))
