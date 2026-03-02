# Supported Hardware

> import CIStatus from '@site/src/components/CIStatus';

import CIStatus from '@site/src/components/CIStatus';

# Supported Hardware

<CIStatus validated={false} />

## NPU-Enabled Processors

Ryzen AI 1.7 Software supports AMD processors codenamed Phoenix, Hawk Point, Strix, Strix Halo, and Krackan Point. These processors can be found in the following Ryzen series:

- Ryzen AI 300 Series, Ryzen AI PRO Series, Ryzen AI Max 300 Series (Strix, Strix Halo, Krackan Point — XDNA 2)
- Ryzen 8000 Series, Ryzen PRO 8000 Series (Hawk Point — XDNA)
- Ryzen 7000 Series, Ryzen PRO 7000 Series (Phoenix — XDNA)
- Ryzen 200 Series

For a complete list, refer to the [AMD processor specifications](https://www.amd.com/en/products/specifications/processors.html) page (look for the "AMD Ryzen AI" column and select "Available").

## GPU Support

Models can be run on the integrated AMD GPU using [DirectML](/develop/rocm-client-gpu). This uses the ONNX Runtime DirectML Execution Provider.

## Operating Systems

| OS | NPU | GPU (DirectML) |
|----|-----|----------------|
| Windows 11 23H2+ | Yes | Yes |

:::info
Linux NPU support for LLMs is available. See [Running LLM on Linux](/models-tutorials/llms/linux-setup).
:::
