call conda create -y --name ms-build-demo python=3.9
call conda install -y -n ms-build-demo -c conda-forge nodejs zlib re2
call conda activate ms-build-demo
copy %CONDA_PREFIX%\Library\bin\zlib.dll %CONDA_PREFIX%\Library\bin\zlib1.dll
pip install -r .\requirements.txt
pip install -r .\pyapp\requirements.txt
pip install wheels\onnxruntime_vitisai-1.15.1-cp39-cp39-win_amd64.whl
pip install wheels\voe-0.1.0-cp39-cp39-win_amd64.whl
python wheels\installer.py
