#!/usr/bin/env bash

echo "setting VITIS Tool Path ..."

VITIS_DIR="/proj/xbuilds/SWIP/2024.1_integration_verified/installs/lin64/Vitis/HEAD"
OS_NAME=$(grep -oP '^ID="*\K\w+' /etc/os-release)
source ${VITIS_DIR}/settings64.sh

xclbin_name1="4x2_psi_model_a16w8_qdq.xclbin"
xclbin_name2="ConvDwc.xclbin"
out_xclbin="4x2_psi_integrated_model_a16w8_qdq.xclbin"

# Usage function
usage() {
    echo "Usage: $(basename "$0") [-x xclbin1] [-y xclbin2] [-o out_xclbin]" >&2
    exit 1
}

# Parse options
while getopts ":x:y:o:" opt; do
    case ${opt} in
        x )
            xclbin_name1=$OPTARG
            ;;
        y )
            xclbin_name2=$OPTARG
            ;;
        o )
            out_xclbin=$OPTARG
            ;;
        \? )
            echo "Invalid option: $OPTARG" >&2
            usage
            ;;
        : )
            echo "Option -$OPTARG requires an argument" >&2
            usage
            ;;
    esac
done

shift $((OPTIND -1))

if [ ! -e "$xclbin_name1" ]; then
    echo "File not exist: $xclbin_name1"
fi

if [ ! -e "$xclbin_name2" ]; then
    echo "File not exist: $xclbin_name2"
fi

echo "Generating xclbin: $out_xclbin";

xclbinutil --dump-section PDI:RAW:psi.pdi --input $xclbin_name1
xclbinutil --dump-section PDI:RAW:conv.pdi --input $xclbin_name2

/proj/xbuilds/IPU-TA/aie-pdi-transform/transform_static psi.pdi psi.pdi
/proj/xbuilds/IPU-TA/aie-pdi-transform/transform_static conv.pdi conv.pdi

xclbinutil --input $xclbin_name1 --remove-section PDI --output 1x4_nopdi.xclbin --force

xclbinutil --input 1x4_nopdi.xclbin --remove-section AIE_PARTITION  --output 1x4_dev.xclbin --force
xclbinutil --add-replace-section PDI:raw:psi.pdi -i 1x4_dev.xclbin -o 1x4_dev1.xclbin --force
xclbinutil --add-replace-section PDI:raw:conv.pdi -i 1x4_dev1.xclbin -o 1x4_dev2.xclbin --force
xclbinutil --input 1x4_dev2.xclbin --add-kernel add_kernel.json --output 1x4_dev3.xclbin --force
xclbinutil --input 1x4_dev3.xclbin  --add-section AIE_PARTITION[]:JSON:aie_partition.json --output $out_xclbin --force
