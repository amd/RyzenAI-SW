import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
from pathlib import Path



cache_dir = Path(__file__).parent.resolve()
# Create session options
session_options = ort.SessionOptions()
session_options.log_severity_level = 1  # 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
ort_session = ort.InferenceSession("model/distilbert-base-uncased-finetuned-sst-2-english.onnx",
                                    sess_options = session_options,
                                    providers=["VitisAIExecutionProvider"],
                                    provider_options=[{'config_file': 'vitisai_config.json',
                                                       'cache_dir': str(cache_dir),
                                                       'cache_key': 'compiled_distilbert-base-uncased-finetuned-sst-2-english_bf16'}])

id2label = {0: "NEGATIVE", 1: "POSITIVE"}

def preprocess(text, tokenizer, max_length=128):
    encoded = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=max_length)
    return encoded["input_ids"].astype(np.int64), encoded["attention_mask"].astype(np.int64)

prompts = [
    "Hello, my dog is cute",
    "Stop talking to me",
    "That painting is ugly",
    "Life is beautiful"
]

for text in prompts:
    input_ids, attention_mask = preprocess(text, tokenizer)

    outputs = ort_session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})

    logits = outputs[0]
    predicted_class_id = np.argmax(logits, axis=1)[0]  
    predicted_label = id2label[predicted_class_id]  

    # Print result
    print("*" * 10)
    print("Prompt:", text)
    print("Text classification:", predicted_label)
