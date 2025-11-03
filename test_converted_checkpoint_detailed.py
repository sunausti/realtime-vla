import pickle
import torch
import numpy as np
from pi0_infer import Pi0Inference

def create_normalized_images(number_of_images=2, seed=42):
    """Create normalized observation images (simulating preprocessed camera inputs)"""
    torch.manual_seed(seed)
    # Simulate normalized images (mean 0, std 1)
    images = torch.randn(number_of_images, 224, 224, 3, dtype=torch.bfloat16, device="cpu")
    return images

def create_observation_state(seed=42):
    """Create observation state (robot proprioceptive state)"""
    torch.manual_seed(seed)
    # Simulate normalized robot state (joint positions, velocities, etc.)
    state = torch.randn(32, dtype=torch.bfloat16, device="cpu")
    return state

def create_diffusion_noise(length_of_trajectory=10, seed=42):
    """Create diffusion input noise for action prediction"""
    torch.manual_seed(seed)
    # Standard normal noise for diffusion process
    noise = torch.randn(length_of_trajectory, 32, dtype=torch.bfloat16, device="cpu")
    return noise

def test_converted_checkpoint_detailed():
    """Detailed test of the converted checkpoint with Pi0Inference"""
    
    print("="*70)
    print("DETAILED TEST: Converted Checkpoint with Pi0Inference")
    print("="*70)
    
    # Load the converted checkpoint
    print("\n[1/5] Loading converted checkpoint...")
    try:
        with open('converted_checkpoint.pkl', 'rb') as f:
            converted_checkpoint = pickle.load(f)
        print(f"✓ Checkpoint loaded successfully")
        print(f"  Number of weight tensors: {len(converted_checkpoint)}")
        
        # Print some weight statistics
        print(f"\n  Sample weight statistics:")
        for key in list(converted_checkpoint.keys())[:5]:
            tensor = converted_checkpoint[key]
            print(f"    {key:40s}: shape={str(tuple(tensor.shape)):20s} dtype={tensor.dtype}")
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        return False
    
    # Configuration
    number_of_images = 2  # Typically 2 camera views
    length_of_trajectory = 10  # 10-step trajectory prediction
    
    print(f"\n[2/5] Initializing Pi0Inference...")
    print(f"  number_of_images: {number_of_images}")
    print(f"  length_of_trajectory: {length_of_trajectory}")
    
    try:
        infer = Pi0Inference(converted_checkpoint, number_of_images, length_of_trajectory)
        print(f"✓ Pi0Inference initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Create inputs
    print(f"\n[3/5] Creating test inputs...")
    
    normalized_observation_image_bfloat16 = create_normalized_images(number_of_images)
    print(f"  ✓ Observation images: {normalized_observation_image_bfloat16.shape} {normalized_observation_image_bfloat16.dtype}")
    
    observation_state_bfloat16 = create_observation_state()
    print(f"  ✓ Observation state: {observation_state_bfloat16.shape} {observation_state_bfloat16.dtype}")
    
    diffusion_input_noise_bfloat16 = create_diffusion_noise(length_of_trajectory)
    print(f"  ✓ Diffusion noise: {diffusion_input_noise_bfloat16.shape} {diffusion_input_noise_bfloat16.dtype}")
    
    # Run inference
    print(f"\n[4/5] Running inference...")
    try:
        output_actions = infer.forward(
            normalized_observation_image_bfloat16,  # (number_of_images, 224, 224, 3)
            observation_state_bfloat16,              # (32,)
            diffusion_input_noise_bfloat16,          # (length_of_trajectory, 32)
        )
        
        print(f"✓ Inference completed successfully")
        
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Analyze results
    print(f"\n[5/5] Analyzing results...")
    print(f"  Output shape: {output_actions.shape}")
    print(f"  Output dtype: {output_actions.dtype}")
    print(f"  Output device: {output_actions.device}")
    print(f"\n  Statistics:")
    print(f"    Min:    {output_actions.min().item():.6f}")
    print(f"    Max:    {output_actions.max().item():.6f}")
    print(f"    Mean:   {output_actions.float().mean().item():.6f}")
    print(f"    Std:    {output_actions.float().std().item():.6f}")
    
    # Check output shape
    expected_shape = (length_of_trajectory, 32)
    if output_actions.shape != expected_shape:
        print(f"\n✗ ERROR: Output shape {output_actions.shape} doesn't match expected {expected_shape}")
        return False
    
    # Check for NaN or Inf
    if torch.isnan(output_actions.float()).any():
        print(f"\n✗ ERROR: Output contains NaN values")
        return False
    if torch.isinf(output_actions.float()).any():
        print(f"\n✗ ERROR: Output contains Inf values")
        return False
    
    print(f"\n  ✓ Output validation passed")
    
    # Test multiple runs for consistency
    print(f"\n[Extra] Testing inference consistency...")
    output_actions_2 = infer.forward(
        normalized_observation_image_bfloat16,
        observation_state_bfloat16,
        diffusion_input_noise_bfloat16,
    )
    
    if torch.allclose(output_actions.float(), output_actions_2.float(), rtol=1e-3, atol=1e-3):
        print(f"  ✓ Consistent output across runs (deterministic)")
    else:
        print(f"  ⚠ Output varies across runs (non-deterministic or state issue)")
    
    # Test with different noise
    diffusion_input_noise_bfloat16_new = create_diffusion_noise(length_of_trajectory, seed=123)
    output_actions_3 = infer.forward(
        normalized_observation_image_bfloat16,
        observation_state_bfloat16,
        diffusion_input_noise_bfloat16_new,
    )
    
    if not torch.allclose(output_actions.float(), output_actions_3.float(), rtol=1e-3, atol=1e-3):
        print(f"  ✓ Different noise produces different output (expected behavior)")
    else:
        print(f"  ⚠ Same output despite different noise (potential issue)")
    
    return True

if __name__ == "__main__":
    success = test_converted_checkpoint_detailed()
    
    print("\n" + "="*70)
    if success:
        print("ALL TESTS PASSED! ✓")
        print("The converted checkpoint is working correctly with Pi0Inference.")
    else:
        print("TESTS FAILED! ✗")
        print("There are issues with the converted checkpoint or inference.")
    print("="*70)
