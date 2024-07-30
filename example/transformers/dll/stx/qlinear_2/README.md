# STX Kernel performance scorecard

## Execution latency breakdown
| dtype | shape  | input_copy | input_sync | xrt_run | output_copy| output_sync | tota|
| ------| ------ | ---------- | ---------- | ------- | ---------- | ----------- | --- |
| w8a8 | 1x2kx2k |            |            |         |            |             |     |

## XRT Run breakdown
| dtype | shape  | XRT->LX* | BD Config | Kernel execution |
| ------| ------ | -------- | --------- | ---------------- |
| w8a8 | 1x2kx2k |          |           |                  |

## Bandwidth utilization
| dtype | shape  | Bandwidth |
| ------| ------ | --------  |
| w8a8 | 1x2kx2k |           |

## Memory footprint
| dtype | shape  | XRT BO | userspace memory |
| ------| ------ | ------ | ---------------- |
| w8a8 | 1x2kx2k |        |                  |
