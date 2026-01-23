import torch
import argparse
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main(args):

   model_id = "distilbert-base-uncased-finetuned-sst-2-english"
   model = AutoModelForSequenceClassification.from_pretrained(model_id)
   tokenizer = AutoTokenizer.from_pretrained(model_id)

   seq_length = 128
   dummy_input_ids = torch.randint(0, 30522, (1, seq_length), dtype=torch.int64)  
   dummy_attention_mask = torch.ones((1, seq_length), dtype=torch.int64)  

   os.makedirs(args.output_dir,exist_ok=True)
   onnx_model_path = os.path.join(args.output_dir,'distilbert-base-uncased-finetuned-sst-2-english.onnx')


   torch.onnx.export(
      model,
      (dummy_input_ids, dummy_attention_mask),
      onnx_model_path,
      input_names=["input_ids", "attention_mask"],
      output_names=["logits"],
      dynamic_axes=None,  
      opset_version=17,
   )

   print("Exported ONNX model with fixed sequence length 128 (int64 inputs).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export from Huggingface to ONNX model')
    parser.add_argument('--output_dir', type=str, required=True,default ='models',help='Output directory for the ONNX model')

    args = parser.parse_args()

    main(args)
