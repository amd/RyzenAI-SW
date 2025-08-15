..
.. Heading guidelines
..
..     # with overline, for parts
..     * with overline, for chapters
..     =, for sections
..     -, for subsections
..     ^, for subsubsections
..     “, for paragraphs
..

.. include:: /icons.txt

########################
NPU Management Interface
########################

*******************************
Introduction
*******************************

The ``xrt-smi`` utility is a command-line interface to monitor and manage the NPU integrated AMD CPUs. 

It is installed in ``C:\Windows\System32\AMD`` and it can be directly invoked from within the conda environment created by the Ryzen AI Software installer.

The ``xrt-smi`` utility currently supports three primary commands:

- ``examine`` - generates reports related to the state of the AI PC and the NPU.
- ``validate`` - executes sanity tests on the NPU.
- ``configure`` - manages the performance level of the NPU.

By default, the output of the ``xrt-smi examine`` and ``xrt-smi validate`` commands goes to the terminal. It can also be written to file in JSON format as shown below:  

.. code-block:: shell

    xrt-smi examine -f JSON -o <path/to/output.json>

The utility also support the following options which can be used with any command:

- ``--help`` - help to use xrt-smi or one of its sub commands
- ``--version`` - report the version of XRT, driver and firmware
- ``--verbose`` - turn on verbosity
- ``--batch`` - enable batch mode (disables escape characters)
- ``--force`` - when possible, force an operation. Eg - overwrite a file in examine or validate

The ``xrt-smi`` utility requires `Microsoft Visual C++ Redistributable <https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170>`_ (version 2015 to 2022) to be installed.


*******************************
Overview of Key Commands
*******************************

.. list-table:: 
   :widths: 35 65 
   :header-rows: 1

   * - Command
     - Description    
   * - examine 
     - system config, device name
   * - examine --report platform   
     - performance mode, power
   * - examine --report aie-partitions 
     - hw contexts
   * - validate --run latency  
     - latency test
   * - validate --run throughput   
     - throughput test
   * - validate --run gemm
     - INT8 GEMM test TOPS. This is a full array test and it should not be run while another workload is running. **NOTE**: This command is not supported on PHX and HPT NPUs.
   * - configure --pmode <mode>    
     - set performance mode


|memo| **NOTE**: The ``examine --report aie-partition`` report runtime information. These commands should be used when a model is running on the NPU. You can run these commands in a loop to see live updates of the reported data.


*******************************
xrt-smi examine
*******************************

System Information
==================

Reports OS/system information of the AI PC and confirm the presence of the AMD NPU.

.. code-block:: shell

    xrt-smi examine

Sample Command Line Output::


    System Configuration
      OS Name              : Windows NT
      Release              : 26100
      Machine              : x86_64
      CPU Cores            : 20
      Memory               : 32063 MB
      Distribution         : Microsoft Windows 11 Enterprise
      Model                : HP OmniBook Ultra Laptop 14-fd0xxx
      BIOS Vendor          : HP
      BIOS Version         : W81 Ver. 01.01.14

    XRT
      Version              : 2.19.0
      Branch               : HEAD
      Hash                 : f62307ddadf65b54acbed420a9f0edc415fefafc
      Hash Date            : 2025-03-12 16:34:48
      NPU Driver Version   : 32.0.203.257
      NPU Firmware Version : 1.0.7.97

    Device(s) Present
    |BDF             |Name       |
    |----------------|-----------|
    |[00c4:00:01.1]  |NPU Strix  |
    

Sample JSON Output::


    {
    "schema_version": {
        "schema": "JSON",
        "creation_date": "Tue Mar 18 22:43:38 2025 GMT"
    },
    "system": {
        "host": {
            "os": {
                "sysname": "Windows NT",
                "release": "26100",
                "machine": "x86_64",
                "distribution": "Microsoft Windows 11 Enterprise",
                "model": "HP OmniBook Ultra Laptop 14-fd0xxx",
                "hostname": "XCOUDAYD02",
                "memory_bytes": "0x7d3f62000",
                "cores": "20",
                "bios_vendor": "HP",
                "bios_version": "W81 Ver. 01.01.14"
            },
            "xrt": {
                "version": "2.19.0",
                "branch": "HEAD",
                "hash": "f62307ddadf65b54acbed420a9f0edc415fefafc",
                "build_date": "2025-03-12 16:34:48",
                "drivers": [
                    {
                        "name": "NPU Driver",
                        "version": "32.0.203.257"
                    }
                ]
            },
            "devices": [
                {
                    "bdf": "00c4:00:01.1",
                    "device_class": "Ryzen",
                    "name": "NPU Strix",
                    "id": "0x0",
                    "firmware_version": "1.0.7.97",
                    "instance": "mgmt(inst=1)",
                    "is_ready": "true"
                }
            ]
        }
      }
    }


    

Platform Information
====================

Reports more detailed information about the NPU, such as the performance mode and power consumption.

.. code-block:: shell

    xrt-smi examine --report platform

Sample Command Line Output::

    --------------------------
    [00c5:00:01.1] : NPU Strix
    --------------------------
    Platform
      Name                   : NPU Strix
      Performance Mode       : Default
     
    Power                  : 1.277 Watts

|memo| **NOTE**: Power reporting is not supported on PHX and HPT NPUs. Power reporting is only available on STX devices and onwards.

NPU Partitions
==============

Reports details about the NPU partition and column occupancy on the NPU.

.. code-block:: shell

    xrt-smi examine --report aie-partitions

Sample Command Line Output::

    --------------------------
    [00c5:00:01.1] : NPU Strix
    --------------------------
    AIE Partitions
      Partition Index: 0
        Columns: [0, 1, 2, 3]
        HW Contexts:
          |PID    |Ctx ID  |Status  |Instr BO  |Sub  |Compl  |Migr  |Err  |Prio    |GOPS  |EGOPS  |FPS  |Latency  |
          |-------|--------|--------|----------|-----|-------|------|-----|--------|------|-------|-----|---------|
          |20696  |0       |Active  |64 KB     |57   |56     |0     |0    |Normal  |0     |0      |0    |0        |


NPU Context Bindings
====================

Reports details about the columns to NPU HW context binding.

.. code-block:: shell

    xrt-smi examine --report aie-partitions --verbose

Sample Command Line Output::

    Verbose: Enabling Verbosity
    Verbose: SubCommand: examine
     
    --------------------------
    [00c5:00:01.1] : NPU Strix
    --------------------------
    AIE Partitions
      Partition Index: 0
        Columns: [0, 1, 2, 3]
        HW Contexts:
          |PID    |Ctx ID  |Status  |Instr BO  |Sub  |Compl  |Migr  |Err  |Prio    |GOPS  |EGOPS  |FPS  |Latency  |
          |-------|--------|--------|----------|-----|-------|------|-----|--------|------|-------|-----|---------|
          |20696  |0       |Active  |64 KB     |57   |56     |0     |0    |Normal  |0     |0      |0    |0        |
     
    AIE Columns
      |Column  ||HW Context Slot  |
      |--------||-----------------|
      |0       ||[1]              |
      |1       ||[1]              |
      |2       ||[1]              |
      |3       ||[1]              |





*******************************
xrt-smi validate
*******************************

Executing a Sanity Check on the NPU
===================================

Runs a set of built-in NPU sanity tests which includes latency, throughput, and gemm.

Note: All tests are run in performance mode.

- ``latency`` - this test executes a no-op control code and measures the end-to-end latency on all columns
- ``throughput`` - this test loops back the input data from DDR through a MM2S Shim DMA channel back to DDR through a S2MM Shim DMA channel. The data movement within the AIE array follows the lowest latency path i.e. movement is restricted to just the Shim tile.
- ``gemm`` - An INT8 GeMM kernel is deployed on all 32 cores by the application. Each core is storing cycle count in the core data memory. The cycle count is read by the firmware. The TOPS application uses the "XBUTIL" tool to capture the IPUHCLK while the workload runs. Once all cores are executed, the cycle count from all cores will be synced back to the host. Finally, the application uses IPUHCLK, core cycle count, and GeMM kernel size to calculate the TOPS. This is a full array test and it should not be run while another workload is running. **NOTE**: This command is not supported on PHX and HPT NPUs.
- ``all`` - All applicable validate tests will be executed (default)


.. code-block:: shell

    xrt-smi validate --run all

|memo| **NOTE**: Some sanity checks may fail if other applications (for example MEP, Microsoft Experience Package) are also using the NPU. 

Sample Command Line Output::


    Validate Device           : [00c4:00:01.1]
       Platform              : NPU Strix
       Power Mode            : Performance
    -------------------------------------------------------------------------------
    Test 1 [00c4:00:01.1]     : gemm
       Details               : TOPS: 51.3
       Test Status           : [PASSED]
    -------------------------------------------------------------------------------
    Test 2 [00c4:00:01.1]     : latency
       Details               : Average latency: 84.2 us
       Test Status           : [PASSED]
    -------------------------------------------------------------------------------
    Test 3 [00c4:00:01.1]     : throughput
       Details               : Average throughput: 59891.0 ops
       Test Status           : [PASSED]
    -------------------------------------------------------------------------------
    Validation completed. Please run the command '--verbose' option for more details
    
*******************************
xrt-smi configure
*******************************

Managing the Performance Level of the NPU
=========================================

To set the performance level of the NPU, you can choose from the following modes: powersaver, balanced, performance, or default. Use the command below:

.. code-block:: shell

   xrt-smi configure --pmode <default | powersaver | balanced | performance | turbo>

- ``default`` - adapts to the Windows Power Mode setting, which can be adjusted under System -> Power & battery -> Power mode. For finer control of the NPU settings, it is recommended to use the xrt-smi mode setting, which overrides the Windows Power mode and ensures optimal results.
- ``powersaver`` - configures the NPU to prioritize power saving, preserving laptop battery life.
- ``balanced`` - configures the NPU to provide a compromise between power saving and performance.
- ``performance`` - configures the NPU to prioritize performance, consuming more power.
- ``turbo`` - configures the NPU for maximum performance performance, requires AC power to be plugged in otherwise uses ``performance`` mode.

Example: Setting the NPU to high-performance mode

.. code-block:: shell

   xrt-smi configure --pmode performance

To check the current performance level, use the following command:

.. code-block:: shell

   xrt-smi examine --report platform

