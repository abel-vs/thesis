# Use the official NVIDIA CUDA image as the base image
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000
EXPOSE 8000

# Copy the rest of the project into the container
COPY . .

ENV PYTHONPATH=/app

# Run api.py when the container launches
CMD ["python3", "src/api.py"]