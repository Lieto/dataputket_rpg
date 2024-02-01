FROM nvidia/cuda:12.3.1-devel-ubuntu22.04


# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0


# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch and torchvision
RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html


# Set the working directory
WORKDIR /app


# Copy application code into the container 
COPY . /app 

RUN python3 -m pip install --upgrade setuptools 

RUN python3 -m pip install -r requirements.txt


# Set the entrypoint
#ENTRYPOINT [ "python3" ]



# Expose port 8080 
EXPOSE 8080 
# Set the default command to run when the container starts 
#CMD ["python", "hello.py"]

