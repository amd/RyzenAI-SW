# Unit test guidelines

The following 2 sections describe existing unit-test cases for the configs supported by the custom ops in the main repository. 
If a new feature is added (such as num_workers or hw_ctx), the developer is expected to write a pytest compatible unit test case, add it to the existing infrastructure and add the information to the section below. 

## FAQ
- Pass ```--capture=tee-sys --verbose``` to the pytest command to display more detailed error messages if there aren't any.
- If you run into this error ```OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized```, please ```set KMP_DUPLICATE_LIB_OK=TRUE```

## Unit tests

The following sections describe unit-test cases for verifying the custom linear operator. 

Run these tests individually, if num_dlls and num_workers are parameterized in pytest, windows throws access failure for multiple processes opening the dll at one time.

### Test cases for Phoenix/Hawk Point
```
pytest --num_dlls 1 --num_workers 1 --impl v0 test_qlinear.py
pytest --num_dlls 1 --num_workers 2 --impl v0 test_qlinear.py
pytest --num_dlls 2 --num_workers 1 --impl v0 test_qlinear.py
pytest --num_dlls 2 --num_workers 2 --impl v0 test_qlinear.py
pytest --impl v1 test_qlinear.py
pytest --w_bit 3 test_qlinear_pergrp.py 
pytest --w_bit 4 test_qlinear_pergrp.py 

# TESTS FAIL - NEEDS UPDATE BUT FA WORKS AT MODEL LEVEL
pytest --impl v0 test_opt_flash_attention.py
pytest --impl v1 test_opt_flash_attention.py
pytest --impl v0 test_llama_flash_attention.py
pytest --impl v1 test_llama_flash_attention.py
```

### Test cases for Strix
```
pytest --num_dlls 1 --num_workers 1 --impl v0 test_qlinear.py
pytest --num_dlls 1 --num_workers 2 --impl v0 test_qlinear.py
pytest --num_dlls 2 --num_workers 1 --impl v0 test_qlinear.py
pytest --num_dlls 2 --num_workers 2 --impl v0 test_qlinear.py
pytest --impl v1 --quant_mode w8a8 test_qlinear.py
pytest --impl v1 --quant_mode w8a16 test_qlinear.py
pytest --w_bit 3 test_qlinear_pergrp.py 
pytest --w_bit 4 test_qlinear_pergrp.py 
```

* To run a single test, "-k" option can be used. 
* To see stdout, --capture=tee-sys can be used (see example below)

```
pytest --num_workers 2 -k test_QLinear_quantmode1 test_qlinear.py --capture=tee-sys --verbose
```