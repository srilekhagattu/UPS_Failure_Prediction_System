# app/schemas.py
from pydantic import BaseModel
from datetime import datetime


class IsolationForestResultSchema(BaseModel):
    id: int
    last_modified: datetime
    battery_voltage: float
    input_voltage: float
    output_voltage: float
    input_frequency: float
    max_output_current_percentage: float
    temp: float
    iso_anomaly: int
    iso_score: float
    failure_iso: str

    class Config:
        from_attributes = True


class KMeansResultSchema(BaseModel):
    id: int
    last_modified: datetime
    battery_voltage: float
    input_voltage: float
    output_voltage: float
    input_frequency: float
    max_output_current_percentage: float
    temp: float
    kmeans_anomaly: int  # ✅ Added
    kmeans_score: float
    failure_kmeans: str

    class Config:
        from_attributes = True
