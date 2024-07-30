# Background
llama.cpp is a C/C++ framework for running llama like LLMs.
Here we have added a RyzenAI backend to the llama.cpp framework to offload matrix multiplications.

# Backend
ggml-ryzenai.h and ggml-ryzenai.cpp provide the implementation.
Effectively, when llama.cpp goes to run its matrix multiplication ops, the RyzenAI backend will intercept the operation if it can be deployed to the NPU.
This is invisible to the user of the framework.
The RyzenAI backend has some criteria for choosing matrix multiplications to offload.
The matrix multiplication must have a certain size, and it must have Q4_0 quantized weights.

# Activation Preprocessing
llama.cpp does not support Google Brain Float16 "bfloat16".
However, RyzenAI's matrix multiplication operator currently requires it for input activations.
Thus, the input tensor is copied into a new buffer as float32->bfloat16 conversion takes place.

# Weights Preprocessing
The first time a matrix multiplication operator is called the weights, scales, and zero points must be statically loaded to pinned memory buffers.
Additionally, the weights and scales are transposed (See Mathematics).

## Mathematics
Traditional matrix multiplication (multiply row by column):

$$
A \times B = C
$$

In `llama.cpp` the actual tensor layout is like this:

$$
A \times B^T = C^T
$$

B is expected to already be transposed, and traspose of C is computed.
This is for cache efficiency.

Additionally, the weights are typically supplied as tensor A.

RyzenAI typically expects the weights to be the B tensor in traditional matrix multiplication.

Lets use the following equation to represent what RyzenAI's matrix multiplication operator will compute:

$$
X \times W = Z
$$

We can make use of the transpose property of matrix multiplication, which says:

$$
B^T \times A^T = C^T
$$

Thus if the weights are supplied by `llama.cpp` as A, we can forward them to RyzenAI as `A^T`:

$$
X = B^T
$$

$$
W = A^T
$$

$$
Z = C^T
$$

Transposition of the weights is a one time penalty.


# How many ops are offloaded?
The answer to this question depends on the model, but for LLama2:
224 layers in the Q4_0 model are offloaded to the NPU, resulting in the following output tensors being computed:
```
Qcur-0 shape: (4096,2,1,1)
Kcur-0 shape: (4096,2,1,1)
Vcur-0 shape: (4096,2,1,1)
kqv_out-0 shape: (4096,2,1,1)
ffn_gate-0 shape: (11008,2,1,1)
ffn_up-0 shape: (11008,2,1,1)
ffn_out-0 shape: (4096,2,1,1)
Qcur-1 shape: (4096,2,1,1)
Kcur-1 shape: (4096,2,1,1)
Vcur-1 shape: (4096,2,1,1)
kqv_out-1 shape: (4096,2,1,1)
ffn_gate-1 shape: (11008,2,1,1)
ffn_up-1 shape: (11008,2,1,1)
ffn_out-1 shape: (4096,2,1,1)
Qcur-2 shape: (4096,2,1,1)
Kcur-2 shape: (4096,2,1,1)
Vcur-2 shape: (4096,2,1,1)
kqv_out-2 shape: (4096,2,1,1)
ffn_gate-2 shape: (11008,2,1,1)
ffn_up-2 shape: (11008,2,1,1)
ffn_out-2 shape: (4096,2,1,1)
Qcur-3 shape: (4096,2,1,1)
Kcur-3 shape: (4096,2,1,1)
Vcur-3 shape: (4096,2,1,1)
kqv_out-3 shape: (4096,2,1,1)
ffn_gate-3 shape: (11008,2,1,1)
ffn_up-3 shape: (11008,2,1,1)
ffn_out-3 shape: (4096,2,1,1)
Qcur-4 shape: (4096,2,1,1)
Kcur-4 shape: (4096,2,1,1)
Vcur-4 shape: (4096,2,1,1)
kqv_out-4 shape: (4096,2,1,1)
ffn_gate-4 shape: (11008,2,1,1)
ffn_up-4 shape: (11008,2,1,1)
ffn_out-4 shape: (4096,2,1,1)
Qcur-5 shape: (4096,2,1,1)
Kcur-5 shape: (4096,2,1,1)
Vcur-5 shape: (4096,2,1,1)
kqv_out-5 shape: (4096,2,1,1)
ffn_gate-5 shape: (11008,2,1,1)
ffn_up-5 shape: (11008,2,1,1)
ffn_out-5 shape: (4096,2,1,1)
Qcur-6 shape: (4096,2,1,1)
Kcur-6 shape: (4096,2,1,1)
Vcur-6 shape: (4096,2,1,1)
kqv_out-6 shape: (4096,2,1,1)
ffn_gate-6 shape: (11008,2,1,1)
ffn_up-6 shape: (11008,2,1,1)
ffn_out-6 shape: (4096,2,1,1)
Qcur-7 shape: (4096,2,1,1)
Kcur-7 shape: (4096,2,1,1)
Vcur-7 shape: (4096,2,1,1)
kqv_out-7 shape: (4096,2,1,1)
ffn_gate-7 shape: (11008,2,1,1)
ffn_up-7 shape: (11008,2,1,1)
ffn_out-7 shape: (4096,2,1,1)
Qcur-8 shape: (4096,2,1,1)
Kcur-8 shape: (4096,2,1,1)
Vcur-8 shape: (4096,2,1,1)
kqv_out-8 shape: (4096,2,1,1)
ffn_gate-8 shape: (11008,2,1,1)
ffn_up-8 shape: (11008,2,1,1)
ffn_out-8 shape: (4096,2,1,1)
Qcur-9 shape: (4096,2,1,1)
Kcur-9 shape: (4096,2,1,1)
Vcur-9 shape: (4096,2,1,1)
kqv_out-9 shape: (4096,2,1,1)
ffn_gate-9 shape: (11008,2,1,1)
ffn_up-9 shape: (11008,2,1,1)
ffn_out-9 shape: (4096,2,1,1)
Qcur-10 shape: (4096,2,1,1)
Kcur-10 shape: (4096,2,1,1)
Vcur-10 shape: (4096,2,1,1)
kqv_out-10 shape: (4096,2,1,1)
ffn_gate-10 shape: (11008,2,1,1)
ffn_up-10 shape: (11008,2,1,1)
ffn_out-10 shape: (4096,2,1,1)
Qcur-11 shape: (4096,2,1,1)
Kcur-11 shape: (4096,2,1,1)
Vcur-11 shape: (4096,2,1,1)
kqv_out-11 shape: (4096,2,1,1)
ffn_gate-11 shape: (11008,2,1,1)
ffn_up-11 shape: (11008,2,1,1)
ffn_out-11 shape: (4096,2,1,1)
Qcur-12 shape: (4096,2,1,1)
Kcur-12 shape: (4096,2,1,1)
Vcur-12 shape: (4096,2,1,1)
kqv_out-12 shape: (4096,2,1,1)
ffn_gate-12 shape: (11008,2,1,1)
ffn_up-12 shape: (11008,2,1,1)
ffn_out-12 shape: (4096,2,1,1)
Qcur-13 shape: (4096,2,1,1)
Kcur-13 shape: (4096,2,1,1)
Vcur-13 shape: (4096,2,1,1)
kqv_out-13 shape: (4096,2,1,1)
ffn_gate-13 shape: (11008,2,1,1)
ffn_up-13 shape: (11008,2,1,1)
ffn_out-13 shape: (4096,2,1,1)
Qcur-14 shape: (4096,2,1,1)
Kcur-14 shape: (4096,2,1,1)
Vcur-14 shape: (4096,2,1,1)
kqv_out-14 shape: (4096,2,1,1)
ffn_gate-14 shape: (11008,2,1,1)
ffn_up-14 shape: (11008,2,1,1)
ffn_out-14 shape: (4096,2,1,1)
Qcur-15 shape: (4096,2,1,1)
Kcur-15 shape: (4096,2,1,1)
Vcur-15 shape: (4096,2,1,1)
kqv_out-15 shape: (4096,2,1,1)
ffn_gate-15 shape: (11008,2,1,1)
ffn_up-15 shape: (11008,2,1,1)
ffn_out-15 shape: (4096,2,1,1)
Qcur-16 shape: (4096,2,1,1)
Kcur-16 shape: (4096,2,1,1)
Vcur-16 shape: (4096,2,1,1)
kqv_out-16 shape: (4096,2,1,1)
ffn_gate-16 shape: (11008,2,1,1)
ffn_up-16 shape: (11008,2,1,1)
ffn_out-16 shape: (4096,2,1,1)
Qcur-17 shape: (4096,2,1,1)
Kcur-17 shape: (4096,2,1,1)
Vcur-17 shape: (4096,2,1,1)
kqv_out-17 shape: (4096,2,1,1)
ffn_gate-17 shape: (11008,2,1,1)
ffn_up-17 shape: (11008,2,1,1)
ffn_out-17 shape: (4096,2,1,1)
Qcur-18 shape: (4096,2,1,1)
Kcur-18 shape: (4096,2,1,1)
Vcur-18 shape: (4096,2,1,1)
kqv_out-18 shape: (4096,2,1,1)
ffn_gate-18 shape: (11008,2,1,1)
ffn_up-18 shape: (11008,2,1,1)
ffn_out-18 shape: (4096,2,1,1)
Qcur-19 shape: (4096,2,1,1)
Kcur-19 shape: (4096,2,1,1)
Vcur-19 shape: (4096,2,1,1)
kqv_out-19 shape: (4096,2,1,1)
ffn_gate-19 shape: (11008,2,1,1)
ffn_up-19 shape: (11008,2,1,1)
ffn_out-19 shape: (4096,2,1,1)
Qcur-20 shape: (4096,2,1,1)
Kcur-20 shape: (4096,2,1,1)
Vcur-20 shape: (4096,2,1,1)
kqv_out-20 shape: (4096,2,1,1)
ffn_gate-20 shape: (11008,2,1,1)
ffn_up-20 shape: (11008,2,1,1)
ffn_out-20 shape: (4096,2,1,1)
Qcur-21 shape: (4096,2,1,1)
Kcur-21 shape: (4096,2,1,1)
Vcur-21 shape: (4096,2,1,1)
kqv_out-21 shape: (4096,2,1,1)
ffn_gate-21 shape: (11008,2,1,1)
ffn_up-21 shape: (11008,2,1,1)
ffn_out-21 shape: (4096,2,1,1)
Qcur-22 shape: (4096,2,1,1)
Kcur-22 shape: (4096,2,1,1)
Vcur-22 shape: (4096,2,1,1)
kqv_out-22 shape: (4096,2,1,1)
ffn_gate-22 shape: (11008,2,1,1)
ffn_up-22 shape: (11008,2,1,1)
ffn_out-22 shape: (4096,2,1,1)
Qcur-23 shape: (4096,2,1,1)
Kcur-23 shape: (4096,2,1,1)
Vcur-23 shape: (4096,2,1,1)
kqv_out-23 shape: (4096,2,1,1)
ffn_gate-23 shape: (11008,2,1,1)
ffn_up-23 shape: (11008,2,1,1)
ffn_out-23 shape: (4096,2,1,1)
Qcur-24 shape: (4096,2,1,1)
Kcur-24 shape: (4096,2,1,1)
Vcur-24 shape: (4096,2,1,1)
kqv_out-24 shape: (4096,2,1,1)
ffn_gate-24 shape: (11008,2,1,1)
ffn_up-24 shape: (11008,2,1,1)
ffn_out-24 shape: (4096,2,1,1)
Qcur-25 shape: (4096,2,1,1)
Kcur-25 shape: (4096,2,1,1)
Vcur-25 shape: (4096,2,1,1)
kqv_out-25 shape: (4096,2,1,1)
ffn_gate-25 shape: (11008,2,1,1)
ffn_up-25 shape: (11008,2,1,1)
ffn_out-25 shape: (4096,2,1,1)
Qcur-26 shape: (4096,2,1,1)
Kcur-26 shape: (4096,2,1,1)
Vcur-26 shape: (4096,2,1,1)
kqv_out-26 shape: (4096,2,1,1)
ffn_gate-26 shape: (11008,2,1,1)
ffn_up-26 shape: (11008,2,1,1)
ffn_out-26 shape: (4096,2,1,1)
Qcur-27 shape: (4096,2,1,1)
Kcur-27 shape: (4096,2,1,1)
Vcur-27 shape: (4096,2,1,1)
kqv_out-27 shape: (4096,2,1,1)
ffn_gate-27 shape: (11008,2,1,1)
ffn_up-27 shape: (11008,2,1,1)
ffn_out-27 shape: (4096,2,1,1)
Qcur-28 shape: (4096,2,1,1)
Kcur-28 shape: (4096,2,1,1)
Vcur-28 shape: (4096,2,1,1)
kqv_out-28 shape: (4096,2,1,1)
ffn_gate-28 shape: (11008,2,1,1)
ffn_up-28 shape: (11008,2,1,1)
ffn_out-28 shape: (4096,2,1,1)
Qcur-29 shape: (4096,2,1,1)
Kcur-29 shape: (4096,2,1,1)
Vcur-29 shape: (4096,2,1,1)
kqv_out-29 shape: (4096,2,1,1)
ffn_gate-29 shape: (11008,2,1,1)
ffn_up-29 shape: (11008,2,1,1)
ffn_out-29 shape: (4096,2,1,1)
Qcur-30 shape: (4096,2,1,1)
Kcur-30 shape: (4096,2,1,1)
Vcur-30 shape: (4096,2,1,1)
kqv_out-30 shape: (4096,2,1,1)
ffn_gate-30 shape: (11008,2,1,1)
ffn_up-30 shape: (11008,2,1,1)
ffn_out-30 shape: (4096,2,1,1)
Qcur-31 shape: (4096,2,1,1)
Kcur-31 shape: (4096,2,1,1)
Vcur-31 shape: (4096,2,1,1)
kqv_out-31 shape: (4096,2,1,1)
ffn_gate-31 shape: (11008,2,1,1)
ffn_up-31 shape: (11008,2,1,1)
ffn_out-31 shape: (4096,2,1,1)
```
