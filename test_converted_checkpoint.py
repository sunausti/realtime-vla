import pickle
import torch
import numpy as np
from pi0_infer import Pi0Inference

def test_converted_checkpoint():
    """Test the converted checkpoint with Pi0Inference"""
    
    # Load the converted checkpoint
    print("Loading converted checkpoint...")
    with open('converted_checkpoint.pkl', 'rb') as f:
        converted_checkpoint = pickle.load(f)
    
    print(f"Checkpoint loaded. Keys: {list(converted_checkpoint.keys())}")
    
    # Configuration
    number_of_images = 2  # Default is 2 views
    length_of_trajectory = 10  # Default trajectory length
    
    print(f"\nInitializing Pi0Inference with:")
    print(f"  number_of_images: {number_of_images}")
    print(f"  length_of_trajectory: {length_of_trajectory}")
    
    # Initialize inference
    infer = Pi0Inference(converted_checkpoint, number_of_images, length_of_trajectory)
    
    # Create dummy inputs
    print("\nCreating dummy inputs...")
    
    # Normalized observation images: (number_of_images, 224, 224, 3) in bfloat16
    normalized_observation_image_bfloat16 = torch.randn(
        number_of_images, 224, 224, 3, 
        dtype=torch.bfloat16, 
        device="cpu"
    )
    print(f"  observation images shape: {normalized_observation_image_bfloat16.shape}")
    
    # Observation state: (32,) in bfloat16
    observation_state_bfloat16 = torch.randn(
        32, 
        dtype=torch.bfloat16, 
        device="cpu"
    )
    print(f"  observation state shape: {observation_state_bfloat16.shape}")
    
    # Diffusion input noise: (length_of_trajectory, 32) in bfloat16
    diffusion_input_noise_bfloat16 = torch.randn(
        length_of_trajectory, 32, 
        dtype=torch.bfloat16, 
        device="cpu"
    )
    print(f"  diffusion input noise shape: {diffusion_input_noise_bfloat16.shape}")
    
    # Run inference
    print("\nRunning inference...")
    try:
        output_actions = infer.forward(
            normalized_observation_image_bfloat16,  # (number_of_images, 224, 224, 3)
            observation_state_bfloat16,              # (32,)
            diffusion_input_noise_bfloat16,          # (length_of_trajectory, 32)
        )
        
        print(f"\nInference successful!")
        print(f"  output_actions shape: {output_actions.shape}")
        print(f"  output_actions dtype: {output_actions.dtype}")
        print(f"  output_actions range: [{output_actions.min():.4f}, {output_actions.max():.4f}]")
        
        return output_actions
        
    except Exception as e:
        print(f"\nInference failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("="*60)
    print("Testing Converted Checkpoint")
    print("="*60)
    
    output_actions = test_converted_checkpoint()
    
    if output_actions is not None:
        print("\n" + "="*60)
        print("Test PASSED!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("Test FAILED!")
        print("="*60)
