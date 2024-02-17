# Installing XRT and device drivers

In case the installation script does not install drivers, following the steps below:

### Copy source files
Copy from shared network drive \\atg-w-u-2\share
- Copy ```\\atg-w-u-2\share\aaronn\install\ipu_stack\ipu_stack_rel_silicon.zip``` and unzip to ```C:\Users\Transformers\```  
- Copy ```\\atg-w-u-2\share\aaronn\install\ipu_stack\test_package.zip``` and unzip into the same folder ```C:\Users\Transformers\ipu_stack_rel_silicon```

Copy from XSJ machines using SCP
- Copy the zip package from xsj:/wrk/xsjhdnobkup1/aaronn/windows_ipu_install/6-27-23-ipu_stack.zip to your device
- Unzip the 6-27-23-ipu_stack.zip
- Copy ```6-27-23-ipu_stack\ipu_stack\ipu_stack_rel_silicon.zip``` and unzip to ```C:\Users\Transformers\``` 
- Copy ```6-27-23-ipu_stack\ipu_stack\test_package.zip``` and unzip to ```C:\Users\Transformers\ipu_stack_rel_silicon``` 

### Update XRT Headers - Bug Fix
- Update Line 20 in ```C:\Users\Transformers\ipu_stack_rel_silicon\xrt-ipu\xrt\include\windows\types.h``` with following:
  ```
  #ifdef MS_WIN64
  typedef __int64 ssize_t;
  #else
  typedef size_t ssize_t;
  #endif
  typedef int pid_t;
  ```

### Update XRT and drivers
Uninstall default/old drivers:
- Check Device Manager → System devices → "AMD IPU Device". If device is there, uninstall it
- Right click "AMD IPU Device" and choose Uninstall device from the menu
  - If available, check the "delete the driver software for the device" box and click Uninstall

Install our custom AMD IPU Device:
- Go to Desktop\ipu_stack* and double click amd_install_kipudrv.bat.
- Check Device Manager → System devices → AMD IPU Device → properties → Driver.
