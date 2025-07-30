"""
Minimal test for multi_token_linear with single token and single operation
"""

import mirage as mi
import torch

def test_minimal_multi_token_linear():
    """Test multi_token_linear with the simplest possible configuration"""
    print("=" * 60)
    print("Minimal Multi-Token Linear Test")
    print("=" * 60)
    
    # Simplest configuration: single token
    num_tokens = 1
    hidden_dim = 128
    output_dim = 128
    max_tokens = 32
    blocks_per_token = 1
    
    print(f"Configuration:")
    print(f"  num_tokens: {num_tokens}")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  output_dim: {output_dim}")
    print(f"  max_tokens: {max_tokens}")
    print(f"  blocks_per_token: {blocks_per_token}")
    print(f"\nThis matches the hardcoded configuration in task_register.cc")
    
    # Initialize device
    mi.set_gpu_device_id(0)
    num_workers, num_schedulers = mi.get_configurations_from_gpu(0)
    print(f"\nGPU configuration: num_workers={num_workers}, num_schedulers={num_schedulers}")
    
    # Create simple test data
    torch.manual_seed(42)
    input_data = torch.randn(1, num_tokens * hidden_dim, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(hidden_dim, output_dim, dtype=torch.bfloat16, device="cuda") * 0.02
    weight_transposed = weight.t().contiguous()  # Transpose for column-major format
    residual = torch.zeros(1, num_tokens * output_dim, dtype=torch.bfloat16, device="cuda")
    output_buffer = torch.zeros(1, num_tokens * output_dim, dtype=torch.bfloat16, device="cuda")
    
    print(f"\nTensor shapes:")
    print(f"  input: {input_data.shape}")
    print(f"  weight (original): {weight.shape}")
    print(f"  weight (transposed): {weight_transposed.shape}")
    print(f"  residual: {residual.shape}")
    print(f"  output: {output_buffer.shape}")
    
    # Create MPK with minimal configuration
    mpk = mi.PersistentKernel(
        world_size=1,
        mpi_rank=0,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        num_remote_schedulers=0,
        max_seq_length=512,
        eos_token_id=128001,
        meta_tensors=[
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            torch.zeros(512, dtype=torch.int64, device="cuda")
        ],
        profiler_tensor=None,
    )
    
    # Attach tensors
    input_tensor = mpk.attach_input(input_data, "input")
    weight_tensor = mpk.attach_input(weight_transposed, "weight")
    residual_tensor = mpk.attach_input(residual, "residual")
    output_tensor = mpk.attach_input(output_buffer, "output")
    
    print("\nCreating multi_token_linear operation...")
    
    # Single multi_token_linear operation with single block
    mpk.multi_token_linear_layer(
        input=input_tensor,
        weight=weight_tensor,
        residual=residual_tensor,
        output=output_tensor,
        grid_dim=(1, 1, 1),  # Single block for single token
        block_dim=(128, 1, 1),
        max_tokens=max_tokens,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        blocks_per_token=blocks_per_token,
        num_tokens=num_tokens,  # Explicitly pass num_tokens
    )
    
    print("Compiling...")
    mpk.compile()
    
    print("Executing...")
    try:
        mpk()
        print("✓ Execution successful!")
        
        # Compute expected output
        expected = torch.matmul(input_data.view(num_tokens, hidden_dim), weight) + residual
        expected = expected.view(1, num_tokens * output_dim)
        
        # Compare results
        max_diff = torch.max(torch.abs(output_buffer - expected)).item()
        print(f"\nMax difference: {max_diff}")
        
        if max_diff < 1e-2:  # BFloat16 tolerance
            print("✓ Test PASSED!")
        else:
            print("✗ Test FAILED - incorrect output")
            print(f"  Output first 5 values: {output_buffer.flatten()[:5]}")
            print(f"  Expected first 5 values: {expected.flatten()[:5]}")
            
    except Exception as e:
        print(f"✗ Execution FAILED with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_minimal_multi_token_linear()