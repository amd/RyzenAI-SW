# Operator interface

## Operator class

Each operator is enabled with C++ operator class derived from the operator interface [base class](../include/ops/op_interface.hpp).
Each operator must provide the following APIs:
1. API to format and copy weights for the operator into a buffer.
2. API to get transaction binary for a give input/output tensor shape.
3. API to get super kernel / runtime parameters for AIE cores, if any.
4. API to get buffer requirements for the operator.

Each operator class is queried for operator metdata required to create a single transaction binary using transction binaries from individual operators.
So dynamic dispatch looks for :
1. Buffer requirements
    * Buffer size for each input
    * Const parameter buffer size - Buffer size for constant inputs for the node. This provides the flexibility to send additional constants like LUTs to AIE kernels if needed.
    * Buffer size each output
    * Super kernel params size - Buffer size to store runtime parameters required for AIE core. Optionally, this can be 0 or combined with constant parameter buffer
    * `arg_map` shown in the code snippet below is for elementwise add operator. It captures the buffer requirement for two inputs, one constant parameter input, one output and one super kernel/runtime parameter input.
        ``` cpp
            struct OpArgMap {
              enum OpArgType {
                INPUT,
                OUTPUT,
                SCRATCH_PAD,
                CONST_INPUT,
                CONST_KERNEL_PARAM_INPUT,
              };
              OpArgType arg_type;
              size_t xrt_arg_idx;
              size_t onnx_arg_idx;
              size_t offset;
              size_t size; // in bytes
            };
            std::vector<OpArgMap> arg_map{
                {OpArgMap::OpArgType::INPUT, 1, 0, 0, input_1_bo_size},
                {OpArgMap::OpArgType::INPUT, 1, 1, input_1_bo_size, input_2_bo_size},
                {OpArgMap::OpArgType::CONST_INPUT, 2, 2, 0, const_params_bo_size},
                {OpArgMap::OpArgType::OUTPUT, 0, 3, 0, output_bo_size},
                {OpArgMap::OpArgType::CONST_KERNEL_PARAM_INPUT, 3, 0, 0,
                 super_kernel_size}};
            return arg_map;
        ```

2. XRT arg idx - ONNX operator mapping
    * Each operator may have a different xrt run argument order. A mapping to xrt idx to ONNX operator idx must be provided. For the above elementwise add operator example, xrt argument order is: `output, input_1, input_2, super_kernel_params`
    This is reflected in the `arg_map` shown above.
        ``` cpp

            run = kernel_(2, instr_bo, instr_bo_words,
                    output_bo_.address() + DDR_AIE_ADDR_OFFSET,
                    input_1_bo_.address() + DDR_AIE_ADDR_OFFSET,
                    input_2_bo_.address() + DDR_AIE_ADDR_OFFSET,
                    param_bo.address() + DDR_AIE_ADDR_OFFSET, 0);
        ```
    * The argument order for elementwise add ONNX operator is `input_1, input_2, output`. ONNX operator documentation is available [here](https://github.com/onnx/onnx/blob/main/docs/Operators.md). This is provided in the `arg_map` above.
    * To summarize, dynamic dispatch uses the xrt argument order and onnx operator argument order to generate a single transaction binary for a given subgraph.
3. Transaction binary
    * Return transcation binary for a tensor shapes of the onnx operator in subgraph.
4. super kernel or runtime parameter
    * Return super kernel or runtime parameters needed for the operator. This can be an empty vector.
5. initialize_const_params
    * If an operator has constant inputs, for ex: weights, this method is expected to convert row major formatting of weights to a different format if the operator requires reformatting of weights. This can be as simple as a memcpy if the operator does not require any data reformatting.
    * This is a powerful mechanism to send LUTs or other constant inputs from DDR to AIE cores if needed.

## Test requirement for each operator

Each operator must come with its own unit test case.
1. C++ unit test
    1. Integrated into GTEST framework.
    2. Add tests to cover functionality of each operator shape.
    3. Please see unit tests for [matmuls](../tests/cpp/unit_tests/test_matmul.cpp) for reference.
2. ONNX node unit test
    1. Single node unit test for the operator
    2. Please see unit test for [mamtuls](../tests/cpp/single_matmul/) for reference.

The goal of these unit tests is to ensure basic functionality and compatibility with ONNX operators.
Please do not add input/golden binary files to the repository for accuracy validation in this repo. Tranasaction binaries and super kernel/runtime parameters are the only two binary files allowed to be pushed to the repository.
