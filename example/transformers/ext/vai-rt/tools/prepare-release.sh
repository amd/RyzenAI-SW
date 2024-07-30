#end
cat <<EOF
boost: v1.79.0
glog: v0.6.0
protobuf: v3.20.1
eigen: d10b27fe37736d2944630ecd7557cefa95cf87c9
pybind11: v2.10.0
opencv: 4.6.0
unilog: $(git ls-remote git@gitenterprise.xilinx.com:VitisAI/unilog.git HEAD | awk '{print $1}')
xir: $(git ls-remote git@gitenterprise.xilinx.com:VitisAI/xir.git HEAD | awk '{print $1}')
target_factory: $(git ls-remote git@gitenterprise.xilinx.com:VitisAI/target_factory.git HEAD | awk '{print $1}')
vart: $(git ls-remote git@gitenterprise.xilinx.com:VitisAI/vart HEAD | awk '{print $1}')
xcompiler: $(git ls-remote git@gitenterprise.xilinx.com:VitisAI/xcompiler HEAD | awk '{print $1}')
graph-engine: $(git ls-remote git@gitenterprise.xilinx.com:VitisAI/graph-engine.git HEAD | awk '{print $1}')
vaip: $(git ls-remote git@gitenterprise.xilinx.com:VitisAI/vaip.git HEAD | awk '{print $1}')
onnxruntime: $(git ls-remote git@gitenterprise.xilinx.com:VitisAI/onnxruntime.git br-vitis-ai-2.5-v4 | awk '{print $1}')
test_onnx_runner: $(git ls-remote git@gitenterprise.xilinx.com:VitisAI/test_onnx_runner.git HEAD | awk '{print $1}')

EOF
