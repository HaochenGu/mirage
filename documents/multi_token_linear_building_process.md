# New operation building process, multi-token linear as an example




# Kernel Design Process

```cpp
template <typename T>
__device__ void multi_token_linear_kernel_v1(...) {
}
```
The kernel should be defined at block level. Block size is fixed at registration, don't use blockIdx.x to determine the block, offset and so on.

Don't forget to update kernel.h

## Registration Process

### 1. Task Register Implementation
```cpp
// File: src/kernel/task_register.cc

void TaskRegister::register_multi_token_linear_task(
    int reduction_size, int output_size, int max_tokens) {
    
    // Build kernel name
    std::ostringstream oss;
    oss << "multi_token_linear_kernel<cutlass::bfloat16_t,"
        << reduction_size << "," << output_size << "," << max_tokens << ">";
    std::string kernel_name = oss.str();
    
    // Register task
    task_type_to_kernel[TASK_MULTI_TOKEN_LINEAR] = kernel_name;
    
    // Generate code
    std::string code = generate_multi_token_linear_code(
        reduction_size, output_size, max_tokens);
    
    generated_code.push_back(code);
}
```

### 2. Code Generation
```cpp
std::string TaskRegister::generate_multi_token_linear_code(
    int reduction_size, int output_size, int max_tokens) {
    
    std::ostringstream oss;
    oss << "  case " << TASK_MULTI_TOKEN_LINEAR << ": {\n"
        << "    multi_token_linear_kernel<cutlass::bfloat16_t, "
        << reduction_size << ", " << output_size << ", " << max_tokens
        << ">((void const*)task.input_ptrs[0],\n"
        << "       (void const*)task.input_ptrs[1],\n"
        << "       (void const*)task.input_ptrs[2],\n"
        << "       (void*)task.output_ptrs[0],\n"
        << "       task.parameters[0]);\n"
        << "    break;\n"
        << "  }\n";
    
    return oss.str();
}
```

### 3. Header Updates
```cpp
// File: src/kernel/task_register.h
class TaskRegister {
public:
    // Add this method
    void register_multi_token_linear_task(
        int reduction_size, int output_size, int max_tokens);
    
private:
    // Add this helper
    std::string generate_multi_token_linear_code(
        int reduction_size, int output_size, int max_tokens);
};
```

## Integration with MPK

### 1. Runtime Mapping
```cpp
// File: src/kernel/runtime.cc

// In generate_attention_single_batch_multi_token:
if (node_name == "multi_token_linear") {
    op_type = TASK_MULTI_TOKEN_LINEAR;
}

// In profiler_get_task_name:
case TASK_MULTI_TOKEN_LINEAR:
    return "MULTI_TOKEN_LINEAR";
```

### 2. Graph Builder Integration
```cpp
// File: src/kernel/graph.cc

// In handle_graph method:
if (op_type == "multi_token_linear") {
    // Extract parameters from tb_graph
    int reduction_size = tb_graph->parameters[0];
    int output_size = tb_graph->parameters[1];
    int max_tokens = tb_graph->parameters[2];
    
    // Register the task
    task_register.register_multi_token_linear_task(
        reduction_size, output_size, max_tokens);
}
```

### 3. Python Bindings
```python
# File: python/mirage/persistent_kernel.py

def multi_token_linear_layer(
        self,
        input: DTensor,      # Shape: (1, num_tokens * reduction_size)
        weight: DTensor,     # Shape: (reduction_size, output_size)
        residual: DTensor,   # Shape: (1, num_tokens * output_size)
        output: DTensor,     # Shape: (1, num_tokens * output_size)
        grid_dim: tuple,     # Must be (num_tokens, 1, 1)
        block_dim: tuple,    # e.g., (128, 1, 1)
        max_tokens: int = 64,
    ):
        # Validate grid dimension
        num_tokens = grid_dim[0]
        assert grid_dim == (num_tokens, 1, 1), "Grid must be (num_tokens, 1, 1)"
        
        # Validate tensor shapes
        assert input.num_dims == 2 and input.dim(0) == 1
        assert weight.num_dims == 2
        assert residual.num_dims == 2 and residual.dim(0) == 1
        assert output.num_dims == 2 and output.dim(0) == 1
        
        # Derive dimensions
        reduction_size = weight.dim(0)
        output_size = weight.dim(1)
        
        # Validate dimension alignment
        assert input.dim(1) == num_tokens * reduction_size
        assert residual.dim(1) == num_tokens * output_size
        assert output.dim(1) == num_tokens * output_size
        
        # Create threadblock graph
        # Calculate shared memory size needed
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        
        # Key mapping configuration:
        # - Input: Map dimension 1 to grid.x (partition by tokens)
        # - Weight: No mapping (shared across all blocks)
        # - Residual: Map dimension 1 to grid.x (partition by tokens)
        # - Output: Map dimension 1 to grid.x (partition by tokens)
        tb_graph.new_input(input, (1, -1, -1), 1, True)
        tb_graph.new_input(weight, (-1, -1, -1), 1, True)
        tb_graph.new_input(residual, (1, -1, -1), 1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        
        # Register with kernel graph
        self.kn_graph.customized([input, weight, residual, output], tb_graph)
        self.kn_graph.register_task(tb_graph, "multi_token_linear", 
                                   [output_size, reduction_size, max_tokens])
```
tb_graph is a wrapper of the graph, (1, -1, -1) means such tensor is splited by grid.x. For example, 

```python
input = [token_1, token_2, token_3, ...]
tb_graph.new_input(input, (1, -1, -1), 0, True)
grid_dim = (num_tokens, 1, 1)
```
mpk will automatically split the input tensor into [token_k] tensors, and register num_tokens tasks, each task is processed by one block(worker).


## Testing and Validation

For multi-block operation, we need to add a dummy operation beginning. Because MPK has the following constraint:

1. The entry of mpk must be single block operation.
2. Different layers of mpk must be chained by tensors, the output of one layer is the input of the next layer.

### 4. Integration Checklist
- Define task type in `runtime_header.h`
- Implement kernel in `tasks/` directory
- Include kernel in `tasks/kernel.h`
- Create registration methods in `task_register.cc/h`
- Add runtime mapping in `runtime.cc`
- Handle in graph builder (`graph.cc`)
- Create Python bindings
