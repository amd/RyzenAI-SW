# Quantization
llama.cpp has several quantization schemes however only one is supported by RyzenAI

# Q4_0
A weights tensor is divided into groups of 32 elements.
This requires that the number of elements in a row is some multiple of 32.
Every group is assigned a scaling factor that maps the floating point value to an int4 quantized value along with a zero point of 8.

To dequantize:
y = s(q-z)
y = float32 weight
s = float16 scalar
q = int4 weight
z = int4 zero point, and its always 8
