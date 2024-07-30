setenv XILINXD_LICENSE_FILE 2100@aiengine-eng
setenv LM_LICENSE_FILE 1757@xsjlicsrvip
setenv XILINX_VITIS /proj/xbuilds/HEAD_INT_daily_latest/installs/lin64/Vitis/9999.0
setenv XILINX_VITIS_HLS /proj/xbuilds/HEAD_INT_daily_latest/installs/lin64/Vitis_HLS/9999.0
setenv XILINX_VIVADO /proj/xbuilds/HEAD_INT_daily_latest/installs/lin64/Vivado/9999.0
setenv XILINX_Model_Composer /proj/xbuilds/HEAD_INT_daily_latest/installs/lin64/Model_Composer/9999.0
setenv XILINX_VITIS_AIETOOLS ${XILINX_VITIS}/aietools
#source ${XILINX_VITIS_AIETOOLS}/scripts/aietools_env.csh
source ${XILINX_VITIS}/settings64.csh
source ${XILINX_VITIS_HLS}/settings64.csh
source ${XILINX_VIVADO}/settings64.csh
source ${XILINX_Model_Composer}/settings64.csh
set Version = 'IPU-TA/9999.0_integration_verified'
setenv XILINX_XRT /proj/xbuilds/${Version}/XRT-IPU/x86_64/centos-default/opt/xilinx/xrt
source /tools/batonroot/rodin/engkits/lnx64/python-3.6.5_svt/bin/activate.csh
setenv PATH /tools/batonroot/rodin/devkits/lnx64/cmake-3.2.1/bin:$PATH
