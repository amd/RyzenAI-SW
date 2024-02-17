# Windows 10/11 Device Setup

Disable device encryption
```
Type and search Device encryption settings in Windows search bar
Set the option to OFF for Device encryption
```

Disable Secure Boot from BIOS, OEM specific
```
Restart your computer, go to the BIOS settings
Under security, look for Secure Boot
Set the option to Disable
Restart machine
```

Open cmd.exe as Administrator, enter the following command and restart you device
```
bcdedit -set testsigning on
```

Once the device has restarted, look for a string printed on Windows desktop like show below: ![img](bcdedit_sample_notification.png)

Now your device is ready for installing the firmware.
