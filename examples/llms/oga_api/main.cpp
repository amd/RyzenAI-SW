#include <iostream>
#include <string>
#include "ort_genai.h"

std::string apply_llama2_chat_template(const std::string& user_input, const std::string& system_prompt = "You are a helpful assistant.") {
    return "<s>[INST] <<SYS>>\n" + system_prompt + "\n<</SYS>>\n\n" + user_input + " [/INST]";
}

int main(int argc, char* argv[]) {
    std::string model_path;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            model_path = argv[++i];
        }
    }

    if (model_path.empty()) {
        std::cerr << "Usage: " << argv[0] << " -m <model_path>" << std::endl;
        return 1;
    }

    std::cout << "Initializing ORT GenAI..." << std::endl;
    OgaHandle handle;

    std::cout << "Loading Model from: " << model_path << std::endl;
    auto model = OgaModel::Create(model_path.c_str());
    if (!model) {
        std::cerr << "Failed to load model from: " << model_path << std::endl;
        return -1;
    }
    std::cout << "Model loaded." << std::endl;

    std::cout << "Creating Tokenizer..." << std::endl;
    auto tokenizer = OgaTokenizer::Create(*model);
    std::cout << "Tokenizer created." << std::endl;

    auto tokenizer_stream = OgaTokenizerStream::Create(*tokenizer);
    auto params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("max_length", 1024);

    std::cout << "Creating Generator..." << std::endl;
    auto generator = OgaGenerator::Create(*model, *params);
    std::cout << "Generator created." << std::endl;

    std::cout << "--------------------------------" << std::endl;

    std::string input;
    std::cout << "Enter prompt: ";
    std::getline(std::cin, input);

    std::string prompt = apply_llama2_chat_template(input);

    auto sequences = OgaSequences::Create();
    tokenizer->Encode(prompt.c_str(), *sequences);
    generator->AppendTokenSequences(*sequences);

    std::cout << "Generating response:\n";
    while (!generator->IsDone()) {
        generator->GenerateNextToken();
        auto token_id = generator->GetSequenceData(0)[generator->GetSequenceCount(0) - 1];
        std::cout << tokenizer_stream->Decode(token_id) << std::flush;
    }

    std::cout << std::endl;
    return 0;
}
