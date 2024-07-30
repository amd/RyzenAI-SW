
# These are the default source repository locations
# You can change them if you need to use different servers or forks
set(boost_URL "https://boostorg.jfrog.io/artifactory/main/release/1.79.0/source/boost_1_79_0.zip" CACHE STRING "Where to get boost")
set(glog_URL "https://github.com/google/glog.git" CACHE STRING "Where to get glog")
set(protobuf_URL "https://github.com/protocolbuffers/protobuf.git" CACHE STRING "Where to get protobuf")
set(pybind11_URL "https://github.com/pybind/pybind11.git" CACHE STRING "Where to get pybind11")
set(googletest_URL "https://github.com/google/googletest.git" CACHE STRING "Where to get gtest")
set(unilog_URL "https://gitenterprise.xilinx.com/VitisAI/unilog.git" CACHE STRING "Where to get unilog")
set(target_factory_URL "https://gitenterprise.xilinx.com/VitisAI/target_factory.git" CACHE STRING "Where to get target_factory")
set(xir_URL "https://gitenterprise.xilinx.com/VitisAI/xir.git" CACHE STRING "Where to get xir")
set(vart_URL "https://gitenterprise.xilinx.com/VitisAI/vart.git" CACHE STRING "Where to get vart")
set(rt_engine_URL "https://gitenterprise.xilinx.com/VitisAI/rt-engine.git" CACHE STRING "Where to get rt-engine")
set(graph_engine_URL "https://gitenterprise.xilinx.com/VitisAI/graph-engine.git" CACHE STRING "Where to get graph-engine")
set(tvm_engine_URL "https://gitenterprise.xilinx.com/VitisAI/tvm-engine.git" CACHE STRING "Where to get tvm-engine")
set(vairt_URL "https://gitenterprise.xilinx.com/VitisAI/vairt.git" CACHE STRING "Where to get vairt")
set(testcases_URL "https://gitenterprise.xilinx.com/VitisAI/testcases.git" CACHE STRING "Where to get testcases")
set(trace_logging_URL "https://gitenterprise.xilinx.com/VitisAI/trace-logging.git" CACHE STRING "Where to get trace-logging")

# These are the default branches / git commits to use
# Do not use "dev" here as that can cause instability
# Only use a release tag, or a hash in this file
# If you want to use dev branch, use cmake CLI:
#  cmake -Dvart_TAG=dev
set(boost_TAG "boost-1.79.0" CACHE STRING "Git Tag for boost") # Tag Not Really Used
set(glog_TAG "v0.6.0" CACHE STRING "Git Tag for glog")
set(protobuf_TAG "v3.18.1" CACHE STRING "Git Tag for protobuf")
set(pybind11_TAG "v2.10.0" CACHE STRING "Git Tag for pybind11")
set(googletest_TAG "release-1.12.1" CACHE STRING "Git Tag for gtest")
set(unilog_TAG "ac306fafdebf8364497a7136c9a7d22c137c018b" CACHE STRING "Git Tag for unilog")
set(target_factory_TAG "5230370914c1e8390a4858ab81f40a888ff04506" CACHE STRING "Git Tag for target_factory")
set(xir_TAG "6007e7679ce9c37669eb1a464455d2063be9bd09" CACHE STRING "Git Tag for xir")
set(vart_TAG "653d375c5b87ba8d2b71d2473be881f241bd5177" CACHE STRING "Git Tag for vart")
set(rt_engine_TAG "9feb8f3d9353335df7dd7d7de3c7b95a54dfe9fe" CACHE STRING "Git Tag for rt-engine")
set(graph_engine_TAG "8b631b22b73ce275fc5fbd1d73542d25cf6cdf5c" CACHE STRING "Git Tag for graph-engine")
set(tvm_engine_TAG "e67505209f94ac3860c688c066b1503c066f608f" CACHE STRING "Git Tag for tvm-engine")
set(vairt_TAG "82ffe5a67749cc5e2e93556cd07f9ed031cc9ffe" CACHE STRING "Git Tag for vairt")
set(testcases_TAG "ba2b9e62f95d1832a0b4a6e947eb7d61f5a62337" CACHE STRING "Git Tag for testcases")
set(trace_logging_TAG "b4d5060dd3635409f1e79fac8ec9e6c0e8a09e7f" CACHE STRING "Git Tag for trace-logging")
