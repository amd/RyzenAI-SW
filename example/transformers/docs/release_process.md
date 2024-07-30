# Release process
As we are rolling out the release version, we need to copy all dependency libs and header files into this repo in the release branch, instead of relying on submodules.

Follow these steps to get all the required libs and header files into the release branch:

### Generate dlls in the main branch
```
# check out the main branch of transformer repo
# follow the steps for setting up the main branch i.e. build the tvm and creating the env and build the transformer
cd transformers\tools
python generate_dll.py
# copy all the dll files in dll/aie (main branch), check out to release branch
# and copy those files into dll/aie (release branch)
```

### Acquire all the libs and header files needed in transformers\third_party
In current setup, the path to tvm aie compiler is ```C:\Users\Transformers\tvm_aie_compiler```. It can be another location in your setup, please change accordingly in the following steps.

```
## checkout to the release branch
# tvm
cp C:\Users\Transformers\tvm_aie_compiler\build\Release\* C:\Users\Transformers\transformers\third_party\lib\
cp -r C:\Users\Transformers\tvm_aie_compiler\include\tvm C:\Users\Transformers\transformers\third_party\include
# maize
cp C:\Users\Transformers\tvm_aie_compiler\build\maize\src\maize\Release\maize.exp C:\Users\Transformers\transformers\third_party\lib
cp C:\Users\Transformers\tvm_aie_compiler\build\maize\src\maize\Release\maize.lib C:\Users\Transformers\transformers\third_party\lib
cp C:\Users\Transformers\tvm_aie_compiler\build\maize\src\maize\Release\maize.dll C:\Users\Transformers\transformers\third_party\bin
cp -r C:\Users\Transformers\tvm_aie_compiler\3rdparty\aie_controller\maize\include\maize C:\Users\Transformers\transformers\third_party\include
# xaiengine
cp C:\Users\Transformers\tvm_aie_compiler\build\aie-rt\Release\xaiengine.lib C:\Users\Transformers\transformers\third_party\lib
cp -r C:\Users\Transformers\tvm_aie_compiler\build\aie-rt\include\* C:\Users\Transformers\transformers\third_party\include
# dmlc
cp -r C:\Users\Transformers\tvm_aie_compiler\3rdparty\dmlc-core\include\dmlc C:\Users\Transformers\transformers\third_party\include
# dlpack
cp -r C:\Users\Transformers\tvm_aie_compiler\3rdparty\dlpack\include\dlpack C:\Users\Transformers\transformers\third_party\include

```
