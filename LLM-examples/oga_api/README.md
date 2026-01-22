# Ryzen AI LLM - Onnxruntime GenAI

Ryzen AI Software includes support for deploying LLMs on Ryzen AI PCs using the ONNX Runtime generate() API (OGA). 

## Pre-optimized Models

AMD provides a set of pre-optimized LLMs ready to be deployed with Ryzen AI Software and the supporting runtime for hybrid and NPU execution. These models can be found on Hugging Face: 

### Published models: 
- [Ryzen AI Hybrid models.](https://huggingface.co/collections/amd/ryzenai-14-llm-hybrid-models-67da31231bba0f733750a99c)
- [Ryzen AI NPU models.](https://huggingface.co/collections/amd/ryzenai-13-llm-npu-models-6759f510b8132db53e044aaf)

## Ryzen AI Installation

- The steps for installing Ryzen AI along with it's requirement can be found in the Official Ryzen AI Software documantion page here - https://ryzenai.docs.amd.com/en/latest/inst.html

## Steps to compile and run LLM example.
- Activate Ryzen AI environment:
```
conda activate ryzen-ai-1.5.0
```
- Download the model: This example uses the Llama-2-7b-chat model.
```
#hyrbid model:
git clone https://huggingface.co/amd/Llama-2-7b-chat-hf-awq-g128-int4-asym-fp16-onnx-hybrid

#npu model:
git clone https://huggingface.co/amd/Llama2-7b-chat-awq-g128-int4-asym-bf16-onnx-ryzen-strix
```

- Clone the RyzenAI-SW repository:
```
git clone https://github.com/amd/RyzenAI-SW
```
- Navigate to OGA_API folder:
```
cd path\to\RyzenAI-SW\example\llm\oga_api
```
- Copy necessary DLLs and header files:
```
xcopy /I "%RYZEN_AI_INSTALLATION_PATH%\deployment\*" libs
xcopy /I "%RYZEN_AI_INSTALLATION_PATH%\LLM\lib\onnxruntime-genai.lib" libs
xcopy /I "%RYZEN_AI_INSTALLATION_PATH%\LLM\include\*" include
```
- Compile and build the code:
```
   mkdir build
   cd build
   cmake .. -A x64
   cmake --build . --config Release
   cd bin\Release
```
- Execute code:
```
.\example.exe -m "<path_to_model>"
```
- Sample command
```
.\example.exe -m "path\to\Llama-2-7b-chat-hf-awq-g128-int4-asym-fp16-onnx-hybrid"
```

- Sample output:
```
Initializing ORT GenAI...
Loading Model from: C:\Users\satreysa\Downloads\RyzenAI-SW\example\llm\oga_api\models\Llama-2-7b-chat-hf-awq-g128-int4-asym-fp16-onnx-hybrid
Model loaded.
Creating Tokenizer...
Tokenizer created.
Creating Generator...
Generator created.
--------------------------------
Enter prompt: Explain the basics of object oriented programming
Generating response:
  Of course! Object-oriented programming (OOP) is a programming paradigm that organizes software design around objects, which are instances of classes, rather than functions and data. Here are the basics of OOP:

1. Classes and Objects: In OOP, a class is a blueprint or template for creating objects. A class defines the properties and behaviors of an object, and it can contain other classes or objects as members. An object is an instance of a class, and it has its own set of attributes (data) and methods (functions).
2. Inheritance: Inheritance is the process of creating a new class based on an existing class. The new class (the subclass) inherits the properties and behaviors of the existing class (the superclass), and it can also add new properties and behaviors.
3. Polymorphism: Polymorphism is the ability of an object to take on many forms. In OOP, polymorphism can occur in two ways: method overriding and method overloading. Method overriding occurs when a subclass provides a different implementation of a method that is already defined in its superclass. Method overloading occurs when a class provides multiple definitions for a method with the same name but different parameters.
4. Encapsulation: Encapsulation is the practice of hiding the implementation details of an object from the outside world. In OOP, encapsulation is used to protect the data and methods of an object from external interference or misuse.
5. Abstraction: Abstraction is the process of representing complex real-world objects or systems in a simplified way. In OOP, abstraction is used to focus on the essential features of an object and to hide the irrelevant details.
6. Composition: Composition is the process of combining objects or classes to create a new object or system. In OOP, composition is used to create complex objects by combining simpler objects or classes.
7. Inheritance Hierarchy: Inheritance hierarchy is a tree-like structure that represents the relationship between classes. A class can inherit properties and behaviors from its parent class, and it can also have its own subclasses that inherit properties and behaviors from it.
8. Interfaces: Interfaces are used to define a set of methods that a class must implement. Interfaces are used to define a contract between a class and its clients, and they are used to ensure that a class implements a set of methods that are common to all classes in a particular category.
9. Abstract Classes: Abstract classes are classes that cannot be instantiated. They are used to define a blueprint for a class, and they can contain methods that are intended to be overridden by subclasses.
10. Final Classes: Final classes are classes that cannot be subclassed. They are used to define a class that cannot be modified or extended.

These are the basic concepts of object-oriented programming. Of course, there are many more advanced concepts and techniques that can be used in OOP, but these are the fundamental building blocks upon which all other concepts are based.
```

**Note:** This example script demonstrates how to run the LLaMA-2-7b-Chat model. The chat template used in `main.cpp` is specifically tailored for the LLaMA-2-7b-Chat model. If you are using a different model, you may need to modify the chat template accordingly to ensure compatibility with that model’s expected input format.

```
std::string apply_llama2_chat_template(const std::string& user_input, const std::string& system_prompt = "You are a helpful assistant.") {
    return "<s>[INST] <<SYS>>\n" + system_prompt + "\n<</SYS>>\n\n" + user_input + " [/INST]";
}
```

# Copyright

Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
