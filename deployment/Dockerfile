# Start from the Hugging Face PyTorch GPU image
FROM huggingface/transformers-pytorch-gpu@sha256:b4b27504b33cce83235c1c330417dd667751512a5df6f3bae3a3bf6cff60d869

# Set the working directory
WORKDIR /workspace

# Copy the requirements.txt file from the "deployment" folder to the container
COPY ./deployment/requirements.txt /workspace

# Install any additional dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the packages to the container
COPY . /workspace/
