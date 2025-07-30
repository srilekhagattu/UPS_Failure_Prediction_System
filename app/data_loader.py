import pandas as pd
from sqlalchemy import create_engine
import traceback
import urllib.parse
from datetime import datetime

# -------------------- PostgreSQL connection config --------------------
db_user = 'postgres'
db_password = urllib.parse.quote_plus('Srilu@05')  # URL-safe encoding
db_host = 'localhost'
db_port = '5432'
db_name = 'ups_anamoly_db'

# Create SQLAlchemy engine
engine = create_engine(
    f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
)


# -------------------- Load Isolation Forest Results --------------------
try:
    iso_csv_path = r'C:/Users/malle/Downloads/ups_isolation_forest_results.csv'
    df_iso = pd.read_csv(iso_csv_path)

    # ✅ Type conversion
    df_iso = df_iso.astype({
        'id': int,
        'battery_voltage': float,
        'input_voltage': float,
        'output_voltage': float,
        'input_frequency': float,
        'max_output_current_percentage': float,
        'temp': float,
        'iso_anomaly': int,
        'iso_score': float,
    })

    # ✅ Handle datetime
    if 'last_modified' in df_iso.columns:
        df_iso['last_modified'] = pd.to_datetime(df_iso['last_modified'])
    else:
        df_iso['last_modified'] = datetime.now()

    # ✅ Add fallback for missing column
    if 'failure_iso' not in df_iso.columns:
        df_iso['failure_iso'] = 'Unknown'

    # ✅ Insert into database
    df_iso.to_sql('isolation_forest_results', engine,
                  if_exists='replace', index=False)
    print("✅ Isolation Forest data inserted successfully!")

except Exception:
    print("❌ Error inserting Isolation Forest data:")
    traceback.print_exc()

# -------------------- Load KMeans Results --------------------
try:
    kmeans_csv_path = r'C:/Users/malle/Downloads/ups_kmeans_results.csv'
    df_kmeans = pd.read_csv(kmeans_csv_path)

    # ✅ Drop 'id' to let PostgreSQL autogenerate it
    if 'id' in df_kmeans.columns:
        df_kmeans = df_kmeans.drop(columns=['id'])

    # ✅ Convert data types
    df_kmeans['battery_voltage'] = df_kmeans['battery_voltage'].astype(float)
    df_kmeans['input_voltage'] = df_kmeans['input_voltage'].astype(float)
    df_kmeans['output_voltage'] = df_kmeans['output_voltage'].astype(float)
    df_kmeans['input_frequency'] = df_kmeans['input_frequency'].astype(float)
    df_kmeans['max_output_current_percentage'] = df_kmeans['max_output_current_percentage'].astype(
        float)
    df_kmeans['temp'] = df_kmeans['temp'].astype(float)

    # ✅ Ensure datetime
    df_kmeans['last_modified'] = pd.to_datetime(df_kmeans['last_modified'])

    # ✅ Add fallback column
    if 'failure_kmeans' not in df_kmeans.columns:
        df_kmeans['failure_kmeans'] = 'Unknown'

    # ✅ Insert into DB
    df_kmeans.to_sql('kmeans_results', engine, if_exists='append', index=False)
    print("✅ KMeans data inserted successfully!")

except Exception:
    print("❌ Error inserting KMeans data:")
    traceback.print_exc()
