import torch
import runtime_kernel
import numpy as np

torch.set_printoptions(sci_mode=False)

# Test configurations
test_configs = [
    {"num_tokens": 1, "hidden_dim": 128, "output_dim": 128},
    {"num_tokens": 2, "hidden_dim": 128, "output_dim": 128},
    {"num_tokens": 4, "hidden_dim": 128, "output_dim": 128},
]

for config in test_configs:
    num_tokens = config["num_tokens"]
    hidden_dim = config["hidden_dim"]
    output_dim = config["output_dim"]
    max_tokens = 32
    blocks_per_token = 1
    
    print(f"\n=== Testing num_tokens={num_tokens}, hidden_dim={hidden_dim}, output_dim={output_dim} ===")
    
    # Create test data
    x = torch.randn((1, num_tokens * hidden_dim), device="cuda", dtype=torch.bfloat16)
    # Weight needs to be transposed to column-major for the kernel
    w = torch.randn((hidden_dim, output_dim), device="cuda", dtype=torch.bfloat16)
    w_transposed = w.t().contiguous()  # Now it's (output_dim, hidden_dim)
    residual = torch.randn((1, num_tokens * output_dim), device="cuda", dtype=torch.bfloat16)
    output = torch.empty(1, num_tokens * output_dim, device="cuda", dtype=torch.bfloat16)
    
    # Call the kernel
    runtime_kernel.multi_token_linear(
        x, w_transposed, residual, output,
        max_tokens, hidden_dim, output_dim, blocks_per_token, num_tokens
    )
    
    # Compute reference
    # Reshape input to [num_tokens, hidden_dim] for matmul
    x_reshaped = x.view(num_tokens, hidden_dim)
    # Use original weight (not transposed) for PyTorch matmul
    torch_out = torch.matmul(x_reshaped, w)  # [num_tokens, output_dim]
    torch_out = torch_out.view(1, num_tokens * output_dim) + residual
    
    # Compare results
    print("Output shape:", output.shape)
    print("Expected shape:", torch_out.shape)
    
    # Check correctness
    max_diff = torch.max(torch.abs(output - torch_out)).item()
    print(f"Max difference: {max_diff}")
    
    if max_diff < 1e-2:  # BFloat16 tolerance
        print("✓ Test PASSED!")
    else:
        print("✗ Test FAILED!")
        print("First few output values:", output.flatten()[:5])
        print("First few expected values:", torch_out.flatten()[:5])
        
        # For debugging, try with identity matrix
        print("\nTrying with identity matrix for debugging...")
        w_identity = torch.eye(hidden_dim, output_dim, device="cuda", dtype=torch.bfloat16)
        w_identity_transposed = w_identity.t().contiguous()
        residual_zero = torch.zeros((1, num_tokens * output_dim), device="cuda", dtype=torch.bfloat16)
        output_identity = torch.empty(1, num_tokens * output_dim, device="cuda", dtype=torch.bfloat16)
        
        runtime_kernel.multi_token_linear(
            x, w_identity_transposed, residual_zero, output_identity,
            max_tokens, hidden_dim, output_dim, blocks_per_token, num_tokens
        )
        
        # Expected: just the input (since weight is identity and residual is zero)
        expected_identity = x if output_dim >= hidden_dim else x[:, :num_tokens * output_dim]
        
        print("Identity test - first few values:")
        print("  Output:", output_identity.flatten()[:5])
        print("  Expected:", expected_identity.flatten()[:5])

print("\n=== Performance Test ===")
# Test with larger dimensions
num_tokens = 4
hidden_dim = 128
output_dim = 128
max_tokens = 32
blocks_per_token = 1

x = torch.randn((1, num_tokens * hidden_dim), device="cuda", dtype=torch.bfloat16)
w = torch.randn((output_dim, hidden_dim), device="cuda", dtype=torch.bfloat16)
residual = torch.randn((1, num_tokens * output_dim), device="cuda", dtype=torch.bfloat16)
output = torch.empty(1, num_tokens * output_dim, device="cuda", dtype=torch.bfloat16)

# Warm-up
for _ in range(16):
    runtime_kernel.multi_token_linear(
        x, w, residual, output,
        max_tokens, hidden_dim, output_dim, blocks_per_token, num_tokens
    )

torch.cuda.synchronize()
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 1000
starter.record()
for rep in range(repetitions):
    runtime_kernel.multi_token_linear(
        x, w, residual, output,
        max_tokens, hidden_dim, output_dim, blocks_per_token, num_tokens
    )
ender.record()
torch.cuda.synchronize()
total_time = starter.elapsed_time(ender)
avg_time = total_time / repetitions
print(f"Average time over {repetitions} runs: {avg_time:.6f} ms")