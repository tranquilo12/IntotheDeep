FROM python:3.12.0-slim

# Set Environment Variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    git \
    ssh \
    build-essential \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working dir
WORKDIR /app

# Copy requirements and install packages
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current dir content into the container at /code
COPY . /app/

# Expose Jupyter port
EXPOSE 8888

# Command to run the kernel gateway
CMD [\
    "jupyter",\
    "kernelgateway",\
    "--KernelGatewayApp.ip=0.0.0.0",\
    "--KernelGatewayApp.port=8888",\
    "--KernelGatewayApp.allow_origin='*'",\
    "--KernelGatewayApp.api='kernel_gateway.notebook_http'",\
    "--KernelGatewayApp.seed_uri='example.ipynb'"\
    ]
