FROM python:3.12-slim

# Ensure /tmp permissions for Vast.ai
RUN mkdir -p /tmp && chmod 1777 /tmp

# Working directory
WORKDIR /app 

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose port 8000
EXPOSE 8000

# Entry point to ensure compatibility with Vast.ai
ENTRYPOINT ["/bin/sh", "-c", "mkdir -p /tmp && chmod 1777 /tmp && exec \"$@\"", "--"]

# Command for FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]