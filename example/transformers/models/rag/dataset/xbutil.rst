NPU Management Interface
========================

**NOTE**: This feature is currently in the early access stage. 

The ``xbutil`` utility is a command-line interface to monitor and manage the NPU. It is installed in ``C:\Windows\System32\AMD`` and it can be directly invoked from within the conda environment created by tge Ryzen AI Software installer.

The ``xbutil`` utility currently supports three primary commands:

- **examine:** Examines the state of the AI PC and the NPU.
- **validate:** Executes sanity tests on the NPU.
- **configure:** Manages the performance level of the NPU.

You can use ``--help`` with any command, such as ``xbutil examine --help``, to view all supported subcommands and their details.

Examining the AI PC and the NPU
-------------------------------

- To provide OS/system information of the AI PC and informs about the presence of the NPU:

  .. code-block:: shell

     xbutil examine

- To provide more detailed information about the NPU, such as its architecture and performance mode:

  .. code-block:: shell

     xbutil examine -d --report platform

- To provide information about the NPU Binary loaded during model inference:

  .. code-block:: shell

     xbutil examine -d --report dynamic-regions

- To provide information about the column occupancy on the NPU, allowing you to determine if more models can run in parallel:

  .. code-block:: shell

     xbutil examine -d --report aie-partitions

Executing a Sanity Check on the NPU
-----------------------------------

- To run a built-in test on the NPU to ensure it is in a deployable state:

  .. code-block:: shell

     xbutil validate -d --run verify

Managing the Performance Level of the NPU
-----------------------------------------

- To set the performance level of the NPU. You can choose powersaver mode, balanced mode, performance mode, or use the default:

  .. code-block:: shell

     xbutil configure -d --performance <powersaver | balanced | performance | default>

