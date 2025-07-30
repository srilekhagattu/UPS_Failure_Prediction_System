# app/config.py

# PostgreSQL Configuration

DB_USERNAME = "postgres"
DB_PASSWORD = "Srilu@05"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "ups_anamoly_db"

# Construct database URL
DATABASE_URL = f"postgresql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
