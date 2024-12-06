FROM python:3.12-slim

# Ensure necessary system updates
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Ensure /tmp permissions for Vast.ai
RUN mkdir -p /tmp && chmod -R 1777 /tmp

# Set environment variables for Vast.ai compatibility
ENV DATA_DIRECTORY=/workspace

# Working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r app/requirements.txt && pip install streamlit

# Expose ports for FastAPI and Streamlit
EXPOSE 8000
EXPOSE 8501

# Install process manager
RUN pip install gunicorn

# Command to start both FastAPI and Streamlit
CMD ["sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 app.main:app & streamlit run frontend/app.py --server.port=8501 --server.address=0.0.0.0"]