#!/bin/bash
# Quantize esrgan_fp32.onnx (UInt8/Int8 MinMSE PowerOf2, CLE + AdaRound) -> output/ryzen_model/model.onnx
# output/ryzen_model/ holds the pre-built model. Pass --force (or set FORCE=1) to rebuild; the old output/ is moved to output_prev/.
set -e
cd "$(dirname "$0")"
[ -f esrgan_fp32.onnx ] || { echo "esrgan_fp32.onnx not found. Run 'python export_fp32.py' first."; exit 1; }
[ "$1" = "--force" ] && FORCE=1
if [ -f output/ryzen_model/model.onnx ] && [ "$FORCE" != "1" ]; then
    echo "output/ryzen_model/model.onnx already exists: this is the pre-built quantized model."
    echo "Re-quantizing needs about 175 GB of temp space and many hours on the CPU, see README.md."
    echo "To rebuild anyway, run 'bash run_quant.sh --force'. The existing output/ is moved to output_prev/."
    exit 1
fi
calib_count=$(find calibration_dataset/images -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)
if [ "$calib_count" -ne 10 ] || [ ! -f calibration_dataset/images/manifest.txt ]; then
    echo "ESRGAN calibration dataset is missing or incomplete. Generating 10 images..."
    python calibration_dataset/prepare_calibration_dataset.py --download
fi
if [ -d output ]; then rm -rf output_prev; mv output output_prev; fi
rm -rf cache
olive run --config olive_ryzen_config.json
