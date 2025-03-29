#!/bin/bash

# Install Git LFS
git lfs install

# Create directory for reference data if it doesn't exist
mkdir -p reference_data

# Clone datasets from Hugging Face
echo "Downloading SciQ dataset..."
git clone https://huggingface.co/datasets/allenai/sciq

echo "Downloading OpenBookQA dataset..."
git clone https://huggingface.co/datasets/allenai/openbookqa

echo "Downloading ARC dataset..."
git clone https://huggingface.co/datasets/allenai/ai2_arc

echo "All datasets downloaded successfully!"
