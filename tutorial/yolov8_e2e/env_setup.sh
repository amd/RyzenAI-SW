#echo "Install all the python dependencies using pip"
#pip install --trusted-host xcdpython.xilinx.com -r requirements_ptq.txt

apt-get install software-properties-common
add-apt-repository ppa:ubuntu-toolchain-r/test
apt-get update
apt-get install gcc-4.9
apt-get upgrade libstdc++6
pip install pycocotools
pip install urllib3==1.26.11

cd code/
python3 setup.py develop
