#!/usr/bin/env python3
"""
Simple example script for using the converted Pi0 checkpoint.

Usage:
    python3 test_simple.py
"""

import pickle
import torch
from pi0_infer import Pi0Inference

# Load converted checkpoint
print("Loading checkpoint...")
converted_checkpoint = pickle.load(open('converted_checkpoint.pkl', 'rb'))

# Initialize inference
number_of_images = 2  # Number of camera views
length_of_trajectory = 10  # Number of action steps to predict

infer = Pi0Inference(converted_checkpoint, number_of_images, length_of_trajectory)
print("Model initialized")

# Create dummy inputs (replace with real data in practice)
normalized_observation_image_bfloat16 = torch.randn(
    number_of_images, 224, 224, 3, 
    dtype=torch.bfloat16, 
    device="cpu"
)

observation_state_bfloat16 = torch.randn(
    32, 
    dtype=torch.bfloat16, 
    device="cpu"
)

diffusion_input_noise_bfloat16 = torch.randn(
    length_of_trajectory, 32, 
    dtype=torch.bfloat16, 
    device="cpu"
)

# Run inference
print("Running inference...")
output_actions = infer.forward(
    normalized_observation_image_bfloat16,  # (number_of_images, 224, 224, 3)
    observation_state_bfloat16,              # (32,)
    diffusion_input_noise_bfloat16,          # (length_of_trajectory, 32)
)

print(f"Output shape: {output_actions.shape}")
print(f"Output range: [{output_actions.min():.4f}, {output_actions.max():.4f}]")
print("Done!")
