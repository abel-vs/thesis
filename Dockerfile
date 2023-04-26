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
WORKDIR /workspace

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Install Jupyter Notebook
RUN pip install jupyter

# Expose the Jupyter Notebook port
EXPOSE 8888

# Copy the rest of the project into the container
COPY . .

# # Set the default command to run the Jupyter Notebook server
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
