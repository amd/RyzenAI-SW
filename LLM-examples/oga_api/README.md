# Ryzen AI LLM - Onnxruntime GenAI

Ryzen AI Software includes support for deploying LLMs on Ryzen AI PCs using the ONNX Runtime generate() API (OGA). 

## Pre-optimized Models

AMD provides a set of pre-optimized LLMs ready to be deployed with Ryzen AI Software and the supporting runtime for hybrid and NPU execution. These models can be found on Hugging Face: 

### Published models: 
- [Ryzen AI Hybrid models.](https://huggingface.co/collections/amd/ryzen-ai-180-hybrid)
- [Ryzen AI NPU models.](https://huggingface.co/collections/amd/ryzen-ai-180-npu-16k)

## Ryzen AI Installation

- The steps for installing Ryzen AI along with it's requirement can be found in the Official Ryzen AI Software documantion page here - https://ryzenai.docs.amd.com/en/latest/inst.html

## Steps to compile and run LLM example.
- Activate Ryzen AI environment:
```
  conda activate ryzen-ai-<version>
```
- Download the model: This example uses the Llama-3.1-8B-Instruct model.
```
#hyrbid model:
git clone https://huggingface.co/amd/Meta-Llama-3.1-8B-Instruct_rai_1.8.0_hybrid

#npu model:
git clone https://huggingface.co/amd/Meta-Llama-3.1-8B-Instruct_rai_1.8.0_npu_16K
```

- Clone the RyzenAI-SW repository:
```
git clone https://github.com/amd/RyzenAI-SW
```
- Navigate to OGA_API folder:
```
cd path\to\RyzenAI-SW\LLM-examples\oga_api
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
.\example.exe -m "path\to\Llama-3.1-8B-Instruct_hybrid"
```

- Sample output:
```
Initializing ORT GenAI...
Loading Model from: path\to\Meta-Llama-3.1-8B-Instruct_hybrid
Model loaded.
Creating Tokenizer...
Tokenizer created.
Creating Generator...
Generator created.
--------------------------------
Enter prompt: Explain the basics of object oriented programming
Generating response:
Object-Oriented Programming (OOP) is a programming paradigm that revolves around the concept of objects and their interactions. Here are the basics:

**Key Concepts:**

1. **Objects**: An object is a self-contained entity that has its own properties and behavior. It's a "thing" in your program that can interact with other objects.
2. **Classes**: A class is a blueprint or a template that defines the structure and behavior of an object. It's a set of instructions that defines the characteristics and actions of an object.
3. **Inheritance**: Inheritance is the mechanism by which one class can inherit the properties and behavior of another class. This allows for code reuse and facilitates the creation of a hierarchy of classes.
4. **Polymorphism**: Polymorphism is the ability of an object to take on multiple forms. This can be achieved through method overriding or method overloading.
5. **Encapsulation**: Encapsulation is the concept of bundling data and its associated methods that operate on that data within a single unit (i.e., an object).
6. **Abstraction**: Abstraction is the process of hiding implementation details and showing only the necessary information to the outside world.

**OOP Principles:**

1. **Single Responsibility Principle (SRP)**: Each object should have a single responsibility and should not be responsible for multiple, unrelated tasks.
2. **Open/Closed Principle (OCP)**: Objects should be open for extension but closed for modification.
3. **Liskov Substitution Principle (LSP)**: Derived classes should be substitutable for their base classes.
4. **Interface Segregation Principle (ISP)**: A client should not be forced to depend on interfaces it does not use.
5. **Dependency Inversion Principle (DIP)**: High-level modules should not depend on low-level modules, but both should depend on abstractions.

**OOP Concepts in Action:**

1. **Objects**: Create an object that represents a car. The car object has properties (color, make, model) and behavior (startEngine, accelerate).
2. **Classes**: Define a Car class that serves as a blueprint for creating car objects.
3. **Inheritance**: Create a ElectricCar class that inherits from the Car class and adds electric-specific properties and behavior.
4. **Polymorphism**: Create a method that takes an object as an argument and performs different actions based on the object's type.
5. **Encapsulation**: Create a BankAccount object that encapsulates the account balance and associated methods (deposit, withdraw).

**Benefits of OOP:**

1. **Modularity**: OOP promotes modularity, making it easier to maintain and extend code.
2. **Code Reuse**: OOP enables code reuse through inheritance and polymorphism.
3. **Easier Maintenance**: OOP makes it easier to modify and extend code without affecting existing functionality.
4. **Improved Readability**: OOP promotes clear and concise code that's easier to understand.

I hope this helps! Do you have any specific questions or topics you'd like me to expand on??
```

**Note:** This example script demonstrates how to run the Llama-3.1-8B-Instruct model. The chat template used in `main.cpp` is specifically tailored for Llama-3 Instruct models. If you are using a different model, you may need to modify the chat template accordingly to ensure compatibility with that model’s expected input format. Note that this template only works with instruction-tuned (Instruct) models; base (pretrained) models do not follow chat templates.

```
std::string apply_llama3_chat_template(const std::string& user_input, const std::string& system_prompt = "You are a helpful assistant.") {
    return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + system_prompt +
           "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + user_input +
           "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
}
```

# Copyright

Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
