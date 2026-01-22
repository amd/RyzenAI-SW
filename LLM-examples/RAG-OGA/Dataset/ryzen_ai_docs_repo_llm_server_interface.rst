.. Heading guidelines
..     # with overline, for parts
..     * with overline, for chapters
..     =, for sections
..     -, for subsections
..     ^, for subsubsections
..     “, for paragraphs

###########################
Server Interface (REST API)
###########################

The Lemonade SDK offers a server interface that allows your application to load an LLM on Ryzen AI hardware in a process, and then communicate with this process using standard ``REST`` APIs. This allows applications written in any language (C#, JavaScript, Python, C++, etc.) to easily integrate with Ryzen AI LLMs.

Server interfaces are used across the LLM ecosystem because they allow for no-code plug-and-play between the higher level of the application stack (GUIs, agents, RAG, etc.) with the LLM and hardware that have been abstracted by the server. 

For example, open source projects such as `Open WebUI <#open-webui-demo>`_ have out-of-box support for connecting to a variety of server interfaces, which in turn allows users to quickly start working with LLMs in a GUI.

************
Server Setup
************

Lemonade Server can be installed via the Lemonade Server Installer executable by following these steps:

1. Make sure your system has the recommended Ryzen AI driver installed as described in :ref:`install-driver`.
2. Download and install ``Lemonade_Server_Installer.exe`` from the `latest Lemonade release <https://github.com/lemonade-sdk/lemonade/releases>`_.
3. Launch the server by double-clicking the ``lemonade_server`` shortcut added to your desktop.

See the `Lemonade Server README <https://github.com/lemonade-sdk/lemonade/blob/main/docs/server/README.md>`_ for more details.

************
Server Usage
************

The Lemonade Server provides the following OpenAI-compatible endpoints:

- POST ``/api/v0/chat/completions`` - Chat Completions (messages to completions)
- POST ``/api/v0/completions`` - Text Completions (prompt to completion)
- GET ``/api/v0/models`` - List available models

Please refer to the `server specification <https://github.com/lemonade-sdk/lemonade/blob/main/docs/server/server_spec.md>`_ document in the Lemonade repository for details about the request and response formats for each endpoint. 

The `OpenAI API documentation <https://platform.openai.com/docs/guides/streaming-responses?api-mode=chat>`_ also has code examples for integrating streaming completions into an application. 

Open WebUI Demo
===============

To experience the Lemonade Server, try using it with an OpenAI-compatible application, such as Open WebUI.

Instructions:
-------------

1. **Launch Lemonade Server:** Double-click the lemon icon on your desktop. See `server setup <#server-setup>`_ for installation instructions.

2. **Install and Run Open WebUI:** In a terminal, install Open WebUI using the following commands:

.. code-block:: bash

    conda create -n webui python=3.11
    conda activate webui
    pip install open-webui
    open-webui serve

3. **Launch Open WebUI**: In a browser, navigate to `<http://localhost:8080/>`_.

4. **Connect Open WebUI to Lemonade Server:** In the top-right corner of the UI, click the profile icon and then:

   - Go to ``Settings`` → ``Connections``.
   - Click the ``+`` button to add our OpenAI-compatible connection.
   - In the URL field, enter ``http://localhost:8000/api/v0``, and in the key field put ``-``, then press save.

**Done!** You are now able to run Open WebUI with Hybrid models. Feel free to choose any of the available “-Hybrid” models in the model selection menu.

**********
Next Steps
**********

- See `Lemonade Server Examples <https://github.com/lemonade-sdk/lemonade/tree/main/docs/server/apps>`_ to find applications that have been tested with Lemonade Server.
- Check out the `Lemonade Server specification <https://github.com/lemonade-sdk/lemonade/blob/main/docs/server/server_spec.md>`_ to learn more about supported features.
- Try out your Lemonade Server install with any application that uses the OpenAI chat completions API.


..
  ------------
  #####################################
  License
  #####################################
  
  Ryzen AI is licensed under `MIT License <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ . Refer to the `LICENSE File <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ for the full license text and copyright notice.
