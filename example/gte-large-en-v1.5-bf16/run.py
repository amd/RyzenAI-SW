import argparse
import onnxruntime as ort
from transformers import AutoModel,AutoTokenizer
import numpy
import numpy as np
from numpy.linalg import norm
from pathlib import Path

def main(args):

   model_path = args.model_path
   cache_dir = Path(__file__).parent.resolve()
   npu_session = ort.InferenceSession(
        model_path,
        providers=["VitisAIExecutionProvider"],
        provider_options=[{"config_file": "vitisai_config.json",
                           "cache_dir": "./",
                           "cacheKey": "modelcachekey"}],
    )

   print('Model compiled Successfully')

   # Required if compile on Linux
   if args.vaiml_compile:
      return

   # Inference pipeline
   input_texts = [
    "what is the capital of China?",
    "how to implement quick sort in python?",
    "Beijing",
    "sorting algorithms"]


   # Generate model data
   model_path_remote = 'Alibaba-NLP/gte-large-en-v1.5'
   tokenizer = AutoTokenizer.from_pretrained(model_path_remote)

   batch_dict = tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')

   ort_input={
       "input.1":batch_dict['input_ids'].numpy(),
       "attention_mask":batch_dict['attention_mask'].numpy(),
       "input.3":batch_dict['token_type_ids'].numpy()
   }

   # NPU Inference on ort_input data
   print("Creating NPU session")
   for i in range(args.num_inferences):
      npu_output = npu_session.run(None,ort_input)

   npu_output = np.array(npu_output)
   npu_embeddings = npu_output[0,:,0]

   print("NPU Embeddings")
   print(npu_embeddings)

   # CPU Inference on ort_input data
   print("Creating CPU session")
   cpu_session = ort.InferenceSession(
       model_path,
       provider = ["CPUExecutionProvider"]
   )

   for i in range(args.num_inferences):
      cpu_output = cpu_session.run(None,ort_input)

   cpu_output = np.array(cpu_output)
   cpu_embeddings = cpu_output[0,:,0]

   print("CPU Embeddings")
   print(cpu_embeddings)

   # Mean Absolute Error
   def mae_error(cpu,npu):
      return(np.mean(np.absolute(cpu - npu)))

   print("Mean Absolute Error between CPU and NPU Embeddings: ",mae_error(cpu_embeddings,npu_embeddings))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ONNX Runtime Inference')
    parser.add_argument('--model_path', type=str, default="models/model_quantized_bf16.onnx", help='Number of inferences to run')
    parser.add_argument('--num_inferences', type=int, default=1, help='Number of inferences to run')
    parser.add_argument('--vaiml_compile', action='store_true', help='Compile Model on Linux')
    args = parser.parse_args()
    main(args)
