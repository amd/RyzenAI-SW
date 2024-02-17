#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved. 
#

import torch
import logging 
import time 
import random
import numpy as np 

prompts = [ "What is the meaning of life?",             
            "Tell me something you don't know.",        
            "What does Xilinx do?",                     
            "What is the mass of earth?",                
            "What is a poem?",                          
            "What is recursion?",                        
            "Tell me a one line joke.",                  
            "Who is Gilgamesh?",                         
            "Tell me something about cryptocurrency.",  
            "How did it all begin?"                     
            ]

def warmup(model, tokenizer, max_new_tokens=30):
    print(f"Warming up ... ")
    for prompt in prompts[0:1]:
        inputs = tokenizer(prompt, return_tensors="pt") 
        generate_ids = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=max_new_tokens)
        _ = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f"Warm up DONE!! ")


def decode_prompt(model, tokenizer, prompt, input_ids=None, max_new_tokens=30):
    if input_ids is None:
        print(f"prompt: {prompt}")
        start = time.time()
        inputs = tokenizer(prompt, return_tensors="pt") 
        end = time.time()
        logging.critical(f"[PROFILE][WARMUP] tokenizer: {end-start}")
    else:
        logging.critical(f"[PROFILE][WARMUP] tokenizer: na") # for logging consistency

    start, end = 0, 0
    prompt_tokens = 0
    input_ids_ = input_ids if prompt is None else inputs.input_ids
    attention_mask = torch.ones((1, input_ids.numel())) if prompt is None else inputs.attention_mask
    start = time.time()
    generate_ids = model.generate(input_ids_, attention_mask=attention_mask, max_new_tokens=max_new_tokens)
    end = time.time()
    prompt_tokens = input_ids_.shape[1]
    num_tokens_out = generate_ids.shape[1]
    new_tokens_generated = num_tokens_out - prompt_tokens
    generate_time = (end - start)
    time_per_token = (generate_time/new_tokens_generated)*1e3
    logging.critical(f"[PROFILE][AIE] generate: {generate_time} for {num_tokens_out} tokens; prompt-tokens: {prompt_tokens}; time per generated token: {time_per_token}")

    start = time.time()
    response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    end = time.time()
    logging.critical(f"[PROFILE][WARMUP] tokenizer decode: {end-start}")
    
    print(f"response: {response}")
    logging.critical(f"response: {response}")


def decode_prompts(model, tokenizer, max_new_tokens=30):
    for prompt in prompts:
        logging.critical("*"*40)
        print("*"*40)
        decode_prompt(model, tokenizer, prompt, max_new_tokens=max_new_tokens)


def get_wikitext2(tokenizer, dataset="non-raw", nsamples=128, seqlen=2048):
    """ gptq """
    from datasets import load_dataset
    if dataset == "non-raw":
        traindata = load_dataset('wikitext', 'wikitext-2-v1', split='train')
        testdata = load_dataset('wikitext', 'wikitext-2-v1', split='test')
    elif dataset == "raw":
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    else:
        raise ValueError(
                "You are using an unsupported dataset, only support wikitext2-raw-v1 and wikitext2-v1."
                "Using wikitext2-raw-v1 with --dataset=raw and wikitext2-v1 with --dataset=non-raw."
            )

    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    dataloader = []
    for _ in range(nsamples):
        i = random.randint(0, testenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = testenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        dataloader.append((inp, tar))
    return dataloader, testenc


def perplexity(model, tokenizer, dataset, framework="pytorch"):
    random.seed(0)
    np.random.seed(0)
    torch.random.manual_seed(0)
    print(f"Calculating Perplexity on wikitext2 test set ...")
    model = model#.cuda()
    dataloader, testenc = get_wikitext2(tokenizer, dataset=dataset)
    
    model.seqlen = 2048
    test_enc = testenc.input_ids
    nsamples = 2 #test_enc.numel() // model.seqlen
    if framework == "pytorch":
        dtype = next(iter(model.parameters())).dtype

    loss = torch.nn.CrossEntropyLoss()
    nlls = []

    with torch.no_grad():
        attention_mask = torch.ones((1, test_enc.numel()))#.cuda()
        for i in range(nsamples):
            batch = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)]#.cuda()
            if framework == "pytorch":
                out = model(
                    batch,
                    attention_mask=attention_mask[:, (i * model.seqlen):((i + 1) * model.seqlen)].reshape((1, -1))
                )
            else :
                out = model(
                    batch,
                    attention_mask=batch.new_ones(batch.shape)
                )
            shift_labels = test_enc[
                :, (i * model.seqlen):((i + 1) * model.seqlen)
            ][:, 1:]#.cuda()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(out.logits[0][:-1, :], shift_labels.view(-1))
            neg_log_likelihood = loss.float() * model.seqlen
            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
        print('Perplexity:', ppl.item())

