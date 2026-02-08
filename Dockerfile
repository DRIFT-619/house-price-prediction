# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run FastAPI + Streamlit together
CMD bash -c "uvicorn api:app --host 0.0.0.0 --port 8000 & streamlit run House_Price_Prediction_App.py --server.port 8501 --server.address 0.0.0.0"
