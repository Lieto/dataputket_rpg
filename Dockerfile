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

RUN mkdir -p /app/repositories
RUN mkdir -p /app/generated_images/demo_imgs 

WORKDIR /app/repositories 

RUN git clone https://github.com/Stability-AI/generative-models
RUN git clone https://github.com/Stability-AI/stablediffusion
RUN git clone https://github.com/sczhou/CodeFormer
RUN git clone https://github.com/crowsonkb/k-diffusion
RUN git clone https://github.com/salesforce/BLIP
WORKDIR /app 

RUN export PYTHONPATH=/app/repositories/stablediffusion

RUN python3 -m pip install --upgrade setuptools 

RUN python3 -m pip install -r requirements.txt

RUN python3 -m pip install -e /app/repositories/stablediffusion
RUN python3 -m pip install -e /app/repositories/k-diffusion
RUN python3 -m pip install -e /app/repositories/generative-models 


# Set the entrypoint
#ENTRYPOINT [ "python3" ]



# Expose port 8080 
EXPOSE 8000
# Set the default command to run when the container starts 
#CMD ["python3", "hello.py"]

