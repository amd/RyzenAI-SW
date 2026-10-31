import transformers
from transformers import AutoModel, AutoTokenizer
import torch
import argparse
import os

def main(args):
    input_texts = [
        "what is the capital of China?",
        "how to implement quick sort in python?",
        "Beijing",
        "sorting algorithms"]


    # Get Model
    model_path = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')


    # onnx model path
    os.makedirs(args.output_dir,exist_ok=True)

    # convert to onnx model
    onnx_model_path = os.path.join(args.output_dir,'gte-large-en-v1.5.onnx')
    dummy_input = dict(batch_dict)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export from Huggingface to ONNX model')
    parser.add_argument('--model_name', type=str, required=True,default='Alibaba-NLP/gte-large-en-v1.5', help='Name or path of the Hugging Face model')
    parser.add_argument('--output_dir', type=str, required=True,default ='models',help='Output directory for the ONNX model')

    args = parser.parse_args()

    main(args)