# Unit test guidelines

The following 2 sections describe existing unit-test cases for the configs supported by the custom ops in the main repository.
If a new feature is added (such as num_workers or hw_ctx), the developer is expected to write a pytest compatible unit test case, add it to the existing infrastructure and add the information to the section below.

## FAQ
- Pass ```--capture=tee-sys --verbose``` to the pytest command to display more detailed error messages if there aren't any.
- If you run into this error ```OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized```, please ```set KMP_DUPLICATE_LIB_OK=TRUE```
- To run a single test, "-k" option can be used.
- To add new argument, use conftest.py based on existing examples in the file.
- Example: ``` pytest --num_workers 2 -k test_QLinear_quantmode1 test_qlinear.py --capture=tee-sys --verbose ```

## Python tests
```
pytest test_qlinear.py
pytest --w_bit 3 test_qlinear_pergrp.py
pytest --w_bit 4 test_qlinear_pergrp.py

# Flash Attention
pytest --quant_mode w4abf16 test_opt_flash_attention.py
pytest --quant_mode w8a8    test_opt_flash_attention.py

pytest --quant_mode w4abf16 test_llama_flash_attention.py
pytest --quant_mode w8a8    test_llama_flash_attention.py

pytest --quant_mode w4abf16 test_qwen2_flash_attention.py

pytest --quant_mode w4abf16 test_chatglm3_flash_attention.py

pytest --quant_mode w4abf16 test_phi_flash_attention.py

pytest --quant_mode w4abf16 test_mistral_flash_attention.py

# Fast MLP
pytest test_llama_fast_mlp.py --w_bit 3
pytest test_llama_fast_mlp.py --w_bit 4
```

### Additional STX specific tests
```
pytest --quant_mode w8a16 test_qlinear.py
pytest --quant_mode w8a16 test_opt_flash_attention.py
pytest --quant_mode w8a16 test_llama_flash_attention.py
```

## Tests for CPU and AIE+CPU Experimental ops
Tests for Python tiling, CPU work are here. These do not need to be tested for AIE development but are used for experimental work.
```
pytest test_tiling.py
pytest test_softmax.py
pytest test_qlinear_cpu.py
pytest test_scalar_mult.py
pytest test_linear_bfloat16.py

# v0 kernel specific
pytest test_qlinear_pytiling.py

```
