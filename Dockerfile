# Stage 1: Build Stage
FROM python:3.10-slim AS build

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and MLflow run logs (for model loading)
COPY src/ /app/src/
COPY mlruns.db /app/
COPY mlruns/ /app/mlruns/

# Stage 2: Run Stage (keeping it simple for this demo)
# Using the same image for simplicity, but a lighter one is better for production
FROM build

# Expose the port for the FastAPI application
EXPOSE 8000

# Set MLflow tracking URI (important for loading the model)
ENV MLFLOW_TRACKING_URI=sqlite:///mlruns.db
ENV REGISTERED_MODEL_NAME=Churn_RF_Model
ENV MODEL_STAGE=Production # The stage we want to load

# Command to run the prediction API using uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
