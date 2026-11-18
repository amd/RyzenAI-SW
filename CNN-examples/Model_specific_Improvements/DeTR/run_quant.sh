#!/bin/bash
# Quantize detr_resnet50_fp32.onnx (XINT8 trunk + A16W8 tail, merged) -> output/ryzen_model/model.onnx
# output/ryzen_model/ holds the pre-built model. Pass --force (or set FORCE=1) to rebuild; the old output/ is moved to output_prev/.
set -e
cd "$(dirname "$0")"
[ -f detr_resnet50_fp32.onnx ] || { echo "detr_resnet50_fp32.onnx not found. Run 'python export_fp32.py' first."; exit 1; }
[ "$1" = "--force" ] && FORCE=1
if [ -f output/ryzen_model/model.onnx ] && [ "$FORCE" != "1" ]; then
    echo "output/ryzen_model/model.onnx already exists: this is the pre-built quantized model."
    echo "Re-quantizing needs about 260 GB of temp space, see README.md."
    echo "To rebuild anyway, run 'bash run_quant.sh --force'. The existing output/ is moved to output_prev/."
    exit 1
fi
calib_count=$(find calibration_dataset/images -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)
if [ "$calib_count" -ne 200 ]; then
    echo "DETR calibration dataset is missing or incomplete. Generating 200 images..."
    python calibration_dataset/prepare_calibration_dataset.py
fi
if [ -d output ]; then rm -rf output_prev; mv output output_prev; fi
rm -rf cache tmp_olive_custom
export PYTHONPATH=${PYTHONPATH:+$PYTHONPATH:}$PWD/olive_ryzen
olive run --config olive_ryzen_config.json --package-config olive_ryzen/olive_ryzen_config.json
