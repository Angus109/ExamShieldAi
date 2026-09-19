# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y pkg-config default-libmysqlclient-dev build-essential libgl1 libglib2.0-0 libsm6 libxrender1 libxext6

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install \
    --no-cache-dir \
    --prefer-binary \
    --timeout 1000 \
    -r requirements.txt
    
# Copy the rest of the application code
COPY . .

# Expose the port Flask runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]