import mirage as mi
import torch

types = torch.bfloat16


def test_multi_token_linear_as_fc():
    """
    Test linear_with_residual_layer with multi-token inputs
    This demonstrates multi-token support in the existing linear kernel
    """
    print("Testing Multi-Token Linear Layer (as FC)")
    print(f"Using type: {types}")
    print("=" * 60)

    # Model parameters - ensure divisible by 128 for kernel constraints
    vocab_size = 4096       # Vocabulary size (for softmax)
    output_dim = 4096       # Output dimension

    # Initialize
    mi.set_gpu_device_id(0)

    # Create model weights
    # For linear layer (vocab_size → output_dim)
    linear_weight = torch.randn(output_dim, vocab_size, dtype=types, device="cuda")

    # Get proper configuration from GPU
    num_workers, num_schedulers = mi.get_configurations_from_gpu(0)
    print(f"GPU configuration: num_workers={num_workers}, "
          f"num_schedulers={num_schedulers}")

    print("\nModel parameters:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  output_dim: {output_dim}")

    # Test different token counts
    test_cases = [1, 4, 8, 16, 32]

    for num_tokens in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing with {num_tokens} tokens")
        print(f"{'-'*40}")

        # Create random logits data
        logits = torch.randn(num_tokens, vocab_size, dtype=types, 
                             device="cuda")
        print(f"Logits shape: {logits.shape}")

        # PyTorch reference: Softmax → Linear
        print("\n1. PyTorch reference (Softmax → Linear):")
        # Softmax
        pytorch_probs = torch.nn.functional.softmax(logits, dim=-1)
        # Linear
        pytorch_output = torch.nn.functional.linear(pytorch_probs, linear_weight,
                                                     bias=None)
        print(f"   Softmax output shape: {pytorch_probs.shape}")
        print(f"   Linear output shape: {pytorch_output.shape}")

        # MPK: Softmax → Linear
        print("\n2. MPK pipeline (Softmax → Linear):")

        # Create output buffers
        softmax_output = torch.zeros_like(logits)
        linear_output = torch.zeros(num_tokens, output_dim, dtype=types,
                                    device='cuda:0')
        
        # Create zero residual for linear layer
        zero_residual = torch.zeros_like(linear_output)

        # Create PersistentKernel
        mpk = mi.PersistentKernel(
            world_size=1,
            mpi_rank=0,
            num_workers=num_workers,
            num_local_schedulers=num_schedulers,
            num_remote_schedulers=0,
            max_seq_length=512,
            eos_token_id=128001,
            meta_tensors=[torch.zeros(1, dtype=torch.int32, device="cuda"),
                          torch.zeros(512, dtype=torch.int64, device="cuda")],
            profiler_tensor=None,
        )

        # Attach tensors
        logits_tensor = mpk.attach_input(logits, "logits")
        softmax_output_tensor = mpk.attach_input(softmax_output, "softmax_output")
        
        linear_weight_tensor = mpk.attach_input(linear_weight, "linear_weight")
        linear_output_tensor = mpk.attach_input(linear_output, "linear_output")
        zero_residual_tensor = mpk.attach_input(zero_residual, "zero_residual")

        # Step 1: Softmax (first operation, single block)
        mpk.softmax_layer(
            input=logits_tensor,
            output=softmax_output_tensor,
            grid_dim=(1, 1, 1),  # Single block for first operation
            block_dim=(128, 1, 1),
            temperature=1.0
        )
        
        # Step 2: Linear layer with residual=0 (acts as pure FC)
        grid_blocks = output_dim // 64  # 64 blocks for 4096 dim
        
        mpk.linear_with_residual_layer(
            input=softmax_output_tensor,  # Connected to softmax output
            weight=linear_weight_tensor,
            residual=zero_residual_tensor,
            output=linear_output_tensor,
            grid_dim=(grid_blocks, 1, 1),
            block_dim=(128, 1, 1)
        )
        
        # Compile and execute
        mpk.compile()
        mpk()
        
        print(f"   Softmax output shape: {softmax_output.shape}")
        print(f"   Linear output shape: {linear_output.shape}")
        
        # Compare outputs
        print("\n3. Comparison:")
        max_diff = torch.max(torch.abs(linear_output - pytorch_output)).item()
        avg_diff = torch.mean(torch.abs(linear_output - pytorch_output)).item()
        
        print(f"   Maximum difference: {max_diff}")
        print(f"   Average difference: {avg_diff}")
        
        # Check if outputs are close
        tolerance = 1e-2  # BFloat16 precision, slightly relaxed for matmul
        if max_diff < tolerance:
            print(f"   ✓ Test PASSED: Outputs match within tolerance ({tolerance})")
        else:
            # Test failed - raise error immediately
            raise AssertionError(
                f"Test FAILED: Maximum difference {max_diff:.6f} exceeds tolerance {tolerance}\n"
                f"   First token comparison:\n"
                f"   MPK output: {linear_output[0, :5]}\n"
                f"   PyTorch output: {pytorch_output[0, :5]}"
            )
    
    print("\n" + "=" * 60)
    print("Multi-token linear test completed!")
    print("\nKey insights:")
    print("1. linear_with_residual_layer CAN handle multi-token inputs")
    print("2. Setting residual=0 makes it behave as a pure FC layer")
    print("3. The kernel automatically adapts based on tensor dimensions")
    print("4. Batch size comes from tensor shape, not hardcoded")
    print("5. No need for a separate multi-token linear kernel!")
    print("\nImportant MPK constraints:")
    print("- Input/output dimensions must be divisible by 128")
    print("- Grid dimensions should be output_dim // 64")
    print("- First operation must use grid_dim=(1,1,1)")
    print("- Operations must be chained via shared tensors")


if __name__ == "__main__":
    test_multi_token_linear_as_fc()