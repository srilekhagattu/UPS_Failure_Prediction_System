# UPS Failure Prediction System
This project uses KMeans and Isolation Forest (unsupervised ML) to detect and classify failures in UPS (Uninterruptible Power Supply) systems. It features:

Unsupervised ML Techniques for anomaly detection
A Streamlit Dashboard for real-time monitoring
A FastAPI Backend
PostgreSQL database integration

Setup Instructions
# Create virtual environment
      python -m venv venv
      venv\Scripts\activate
# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn
pip install sqlalchemy psycopg2-binary streamlit plotly
pip install fastapi uvicorn
# How to Run
      python run_model.py
# Load results to PostgreSQL
      python app/data_loader.py
# Start FastAPI
      uvicorn app.main:app --reload
# Launch Dashboard
      streamlit run dashboard/dashboard.py
Open your browser at: http://localhost:8501
