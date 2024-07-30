# Operator performance scorecard

## Execution latency breakdown
| dtype | shape  | Padding | input_copy | input_sync | xrt_run | output_copy| output_sync | accumulation | total |
| ------| ------ | ------- | ---------- | ---------- | ------- | ---------- | ----------- | ------------ | ----- |
| w8a8 | 1x2kx2k |         |            |            |         |            |             |              |       |

## Memory footprint
| dtype | shape  | XRT BO | userspace memory |
| ------| ------ | ------ | ---------------- |
| w8a8 | 1x2kx2k |        |                  |
