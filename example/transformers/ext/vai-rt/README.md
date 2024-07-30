# vai-rt
This repo is used to acquire and build Vitis-AI Runtime sources.  
  
At the root directory there is support for a Python driven flow.  
  
In the [cmake](cmake) directory there is support for a CMake driven flow.

## Compile
To start with the program, execeute "python main.py".<br>
If it is successful, when the program finishes, it will ask you to check the compliation log at a certain directory.
## Possible error
You may need to execute "export https_proxy=http://localhost:9181" inside Xilinx's internal environments for downloading external repos.
## Common option
If you need to edit the code and recompile in the future, append --dev-mode.<br>
If you need to compile only one project, append --project {project_name}.<br>
If you need to build the release version, add --type Release.<br>
## Related directory
All source files are in /workspace if --dev-mode if is enabled.<br>
The compliation intermediates are in ~/build/.<br>
The compiled executables are in ~/.local/build.

##
