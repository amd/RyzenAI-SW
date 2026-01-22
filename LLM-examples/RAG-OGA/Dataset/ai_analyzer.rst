###########
AI Analyzer
###########

AMD AI Analyzer is a tool that supports analysis and visualization of model compilation and inference on Ryzen AI. The primary goal of the tool is to help users better understand how the models are processed by the hardware, and to identify performance bottlenecks that may be present during model inference. Using AI Analyzer, users can visualize graph and operator partitions between the NPU and CPU. 

Installation
~~~~~~~~~~~~

If you installed the Ryzen AI software using automatic installer, AI Analyzer is already installed in the conda environment. 

If you manually installed the software, you will need to install the AI Analyzer wheel file in your environment. 


.. code-block::

   python -m pip install path\to\RyzenAI\installation\files\aianalyzer-<version>.whl


Enabling Profiling and Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Profiling and Visualization can be enabled by passing additional provider options to the ONNXRuntime Inference Session. An example is shown below: 

.. code-block::

   provider_options = [{
                'config_file': 'vaip_config.json',
                'cacheDir': str(cache_dir),
                'cacheKey': 'modelcachekey', 
                'ai_analyzer_visualization': True,
                'ai_analyzer_profiling': True,
            }]
   session = ort.InferenceSession(model.SerializeToString(), providers=providers,
                               provider_options=provider_options)


The ``ai_analyzer_profiling`` flag enables generation of artifacts related to the inference profile. The ``ai_analyzer_visualization`` flag enables generation of artifacts related to graph partitions and operator fusion. These artifacts are generated as .json files in the current run directory.

AI Analyzer also supports native ONNX Runtime profiling, which can be used to analyze the parts of the session running on the CPU. Users can enable ONNX Runtime profiling through session options and pass it alongside the provider options as shown below:

.. code-block::

   # Configure session options for profiling
   sess_options = rt.SessionOptions()
   sess_options.enable_profiling = True
 
   provider_options = [{
                'config_file': 'vaip_config.json',
                'cacheDir': str(cache_dir),
                'cacheKey': 'modelcachekey', 
                'ai_analyzer_visualization': True,
                'ai_analyzer_profiling': True,
            }]
 
  session = ort.InferenceSession(model.SerializeToString(), sess_options, providers=providers,
                               provider_options=provider_options)


Launching AI Analyzer
~~~~~~~~~~~~~~~~~~~~~

Once the artifacts are generated, `aianalyzer` can be invoked through the command line as follows: 


.. code-block::

    aianalyzer <logdir> <additional options>


**Positional Arguments**

``logdir``: Path to the folder containing generated artifacts 

Additional Options

``-v``, ``--version``: Show the version info and exit.

``-b ADDR``, ``--bind ADDR``: Hostname or IP address on which to listen, default is 'localhost'.

``-p PORT``, ``--port PORT``: TCP port on which to listen, default is '8000'.

``-n``, ``--no-browser``: Prevent the opening of the default url in the browser.

``-t TOKEN``, ``--token TOKEN``: Token used for authenticating first-time connections to the server. The default is to generate a new, random token. Setting to an empty string disables authentication altogether, which is NOT RECOMMENDED.



Features
~~~~~~~~

AI Analyzer provides visibility into how your AI model is compiled and executed on Ryzen AI hardware. Its two main use cases are:

1. Analyzing how the model was partitioned and mapped onto Ryzen AI's CPU and NPU accelerator
2. Profiling model performance as it executes inferencing workloads

When launched, the AI Analyzer server will scan the folder specified with the logdir argument and detect and load all files relevant to compilation and/or inferencing  per the ai_analyzer_visualization and ai_anlayzer_profiling flags.

You can instruct the AI Analyzer server to either start a browser on the same host or else return to you a URL that you can then load into a browser on any host.


User Interface
~~~~~~~~~~~~~~

AI Analyzer has the following three sections as seen in the left-panel navigator

1. PARTITIONING - A breakdown of your model was assigned to execute inference across CPU and NPU
2. NPU INSIGHTS - A detailed look at the how your model was optimized for inference execution on NPU
3. PERFORMANCE - A breakdown of inference execution through the model


These sections are described in more detail below



PARTITIONING
@@@@@@@@@@@@

This section is comprised of two pages: Summary and Graph

**Summary**

The Summary page gives an overview of how the  models operators have been assigned to Ryzen's CPU and NPU along with charts capturing GigaOp (GOP) offloading by operator type .

There is also table titled "CPU Because" that shows the reasons why certain operators were not offloaded to the NPU.

**Graph**

The graph page shows an interactive diagram of the partitioned ONNX model, showing graphically how the layers are assigned to the Ryzen hardware.



Toolbar

- You can choose to show/hide individual NPU partitions, if any, with the "Filter by Partition" button
- A panel that displays properties for selected objects can be shown or hidden via the "Show Properties" toggle button
- The model table can be shown and hidden via the "Show Table" toggle button.
- Settings
 
  - Show Processor will separate operators that run on CPU and NPU respectively
  - Show Partition will separate operators running on the NPU by their respective NPU partition, if any
  - Show Instance Name will display the full hierarchical name for the operators in the ONNX model

All objects in the graph have properties which can be viewed to the right of the graph.



*Model Table*

This table below the graph lists all objects in the partitioned ONNX model:

- Processor (NPU or CPU)
- Function (Layer)
- Operator
- Ports
- NPU Partitions


NPU INSIGHTS
@@@@@@@@@@@@

This section is comprised of three pages: Summary, Original Graph, and Optimized Graph.



**Summary**

The Summary page gives an overview of how your model was mapped to the AMD Ryzen NPU. Charts are displayed showing statistics on the number of operators and total GMACs that have been mapped to the NPU (and if necessary, back to CPU via the "Failsafe CPU" mechanism). The statistics are shown per operator type and per NPU partition. 



**Original Graph**

This is an interactive graph representing your model lowered to supported NPU primitive operators, and broken up into partitions if necessary. As with the PARTITIONING graph, there is a companion table containing all of the model elements that will cross-probe to the graph view. The objects in the graph and table will also cross-probe to the PARTITIONING graph.

Toolbar 

You can choose to show/hide individual NPU partitions, if any, with the "Filter by Partition" button
A panel that displays properties for selected objects can be shown or hidden via the "Show Properties" toggle button
A code viewer showing the MLIR source code with cross-probing can be shown/hidden via the "Show Code View" button
The table below can be shown and hidden via the "Show Table" toggle button.
Display options for the graph can be accessed with the "Settings" button




**Optimized Graph**

This page shows the final model that will be mapped to the NPU after all transformations and optimizations such as fusion and chaining. It will also report the operators that had to be moved back to the CPU via the "Failsafe CPU" mechanism. As usual, there is a companion table below that contains all of the graph's elements, and cross-selection is supported to and from the PARTITIONING graph and the Original Graph.

Toolbar 

You can choose to show/hide individual NPU partitions, if any, with the "Filter by Partition" button
A panel that displays properties for selected objects can be shown or hidden via the "Show Properties" toggle button
The table below can be shown and hidden via the "Show Table" toggle button.
Display options for the graph can be accessed with the "Settings" button


PERFORMANCE
@@@@@@@@@@@

This section is used to view the performance of your model on RyzenAI when running one or more inferences. It is comprised of two pages: Summary and Timeline.



**Summary**

The performance summary page shows several overall statistics on the inference(s) as well as charts breaking down operator runtime by operator. If you run with ONNX runtime profiler enabled, you will see overall inference time including layers that run on the CPU. If you have NPU profiling enabled via the ai_analyzer_profiling flag, you will see numerous NPU-based statistics, including GOP and MAC efficiency and a chart of runtime per NPU operator type.

The clock frequency field shows the assumed NPU clock frequency, but it can be edited. If you change the frequency, all timestamp data that is collected as clock cycles but displayed in time units will be adjusted accordingly.


**Timeline**

The Performance timeline shows a layer-by-layer breakdown of your model's execution.  The upper section is a graphical depiction of layer execution across a timeline, while the lower section shows the same information in tabular format. It is important to note that the Timeline page shows one inference at a time, so if you have captured profiling data for two or more inferences, you can choose which one to display with the "Inferences" chooser.



Within each inference, you can examine the overall model execution or the detailed NPU execution data by using the "Partition" chooser. 



Toolbar 

A panel that displays properties for selected objects can be shown or hidden via the "Show Properties" toggle button
The table below can be shown and hidden via the "Show Table" toggle button.
The graphical timeline can be downloaded to SVG via the "Export to SVG" button


..
  ------------

  #####################################
  License
  #####################################

 Ryzen AI is licensed under `MIT License <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ . Refer to the `LICENSE File <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ for the full license text and copyright notice.

