FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required by GitPython and tree-sitter
RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables for data and logs
ENV DATA_DIR=/app/data
ENV LOGS_DIR=/app/logs

# Expose backend port
EXPOSE 8000

# Start the application
CMD ["python", "main.py"]
