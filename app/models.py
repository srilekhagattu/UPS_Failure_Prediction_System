# app/models.py
from sqlalchemy import Column, Integer, Float, String, DateTime
from app.database import Base


class IsolationForestResult(Base):
    __tablename__ = 'isolation_forest_results'

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    last_modified = Column(DateTime)
    battery_voltage = Column(Float)
    input_voltage = Column(Float)
    output_voltage = Column(Float)
    input_frequency = Column(Float)
    max_output_current_percentage = Column(Float)
    temp = Column(Float)
    iso_anomaly = Column(Integer)
    iso_score = Column(Float)
    failure_iso = Column(String)


class KMeansResult(Base):
    __tablename__ = 'kmeans_results'

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    last_modified = Column(DateTime)
    battery_voltage = Column(Float)
    input_voltage = Column(Float)
    output_voltage = Column(Float)
    input_frequency = Column(Float)
    max_output_current_percentage = Column(Float)
    temp = Column(Float)
    kmeans_anomaly = Column(Integer)  # ✅ Added
    kmeans_score = Column(Float)
    failure_kmeans = Column(String)
