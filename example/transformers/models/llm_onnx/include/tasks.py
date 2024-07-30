##
## Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
##

import logging
import random
import time

import numpy as np
import torch
from colorama import Back, Fore, Style

from .logger import LINE_SEPARATER


def get_wikitext2(tokenizer, dataset="non-raw", nsamples=128, seqlen=2048):
    """gptq"""
    from datasets import load_dataset

    if dataset == "non-raw":
        traindata = load_dataset("wikitext", "wikitext-2-v1", split="train")
        testdata = load_dataset("wikitext", "wikitext-2-v1", split="test")
    elif dataset == "raw":
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    else:
        raise ValueError(
            "You are using an unsupported dataset, only support wikitext2-raw-v1 and wikitext2-v1."
            "Using wikitext2-raw-v1 with --dataset=raw and wikitext2-v1 with --dataset=non-raw."
        )

    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    dataloader = []
    for _ in range(nsamples):
        i = random.randint(0, testenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = testenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        dataloader.append((inp, tar))
    return dataloader, testenc


class Tasks:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def decode(self, prompts: list, max_length=64, do_sample=False):
        # User Prompt Decode
        for prompt in prompts:
            # Encode
            logging.critical(f"[PROFILE] tokenizer:")
            # Start timer
            start = time.perf_counter()
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = inputs.input_ids

            # Generate
            generate_ids = self.model.generate(
                input_ids,
                do_sample=do_sample,
                max_length=max_length,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # Stop timer
            end = time.perf_counter()
            generate_time = end - start

            # Num of tokens
            prompt_tokens = input_ids.shape[1]
            num_tokens_out = generate_ids.shape[1]
            new_tokens_generated = num_tokens_out - prompt_tokens

            time_per_token = (generate_time / new_tokens_generated) * 1e3
            # log time
            log_str = (
                "[PROFILE] generate: {} ".format(generate_time)
                + "for {} tokens; ".format(num_tokens_out)
                + "prompt-tokens: {}; ".format(prompt_tokens)
                + "time per generated token: {}".format(time_per_token)
            )
            logging.critical(log_str)

            # Decode
            transcription = self.tokenizer.batch_decode(
                generate_ids, skip_special_tokens=True
            )
            logging.critical(f"response: {transcription[0]}")

            print(
                "\n- Prompt: "
                + Fore.BLUE
                + "{}".format(prompt)
                + Fore.RESET
                + "\n- Response: "
                + Fore.CYAN
                + "{}\n".format(transcription[0])
                + Style.RESET_ALL
                + LINE_SEPARATER
            )

    def benchmark(self, in_seqlens, max_new_tokens, do_sample=False):

        # Load dataset
        trainloader, testenc = get_wikitext2(self.tokenizer, nsamples=2, seqlen=4096)

        if hasattr(self.model.config, "max_position_embeddings"):
            max_pos_embeddings = self.model.config.max_position_embeddings
        else:
            # Define a default value if the attribute is missing in the config.
            max_pos_embeddings = 1024
        # Get tokens
        input_ids = next(iter(trainloader))[0][:, :max_pos_embeddings]
        for seqlen in in_seqlens:
            # Ger "seqlen" tokens

            if (seqlen + max_new_tokens) > max_pos_embeddings:
                continue
            input_ids_test = input_ids[:, :seqlen]

            logging.critical("*" * 40)
            print(
                "\n- Benchmarking for Input Seq lenth = {}, ".format(seqlen)
                + "Max New Tokens = {}\n{}".format(max_new_tokens, LINE_SEPARATER)
            )

            logging.critical(f"[PROFILE] tokenizer:")
            # Start timer
            start = time.perf_counter()

            # Generate
            generate_ids = self.model.generate(
                input_ids_test,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
            )

            # Stop timer
            end = time.perf_counter()
            generate_time = end - start
            # Token lengths
            prompt_tokens = input_ids_test.shape[1]
            num_tokens_out = generate_ids.shape[1]
            new_tokens_generated = num_tokens_out - prompt_tokens

            time_per_token = (generate_time / new_tokens_generated) * 1e3
            # log time
            log_str = (
                "[PROFILE] generate: {} ".format(generate_time)
                + "for {} tokens; ".format(num_tokens_out)
                + "prompt-tokens: {}; ".format(prompt_tokens)
                + "time per generated token: {}".format(time_per_token)
            )
            logging.critical(log_str)

            # Decode
            transcription = self.tokenizer.batch_decode(
                generate_ids, skip_special_tokens=True
            )
            logging.critical(f"response: {transcription[0]}")

            print(
                "- Response: "
                + Fore.CYAN
                + "{}\n".format(transcription[0])
                + Style.RESET_ALL
            )

            print(
                "- Generate Time (s) = "
                + Fore.CYAN
                + "{}\n".format(generate_time)
                + Fore.RESET
            )

    def perplexity(self, framework="onnx"):
        random.seed(0)
        np.random.seed(0)
        torch.random.manual_seed(0)
        print(f"Calculating Perplexity on wikitext2 test set ...")
        model = self.model  # .cuda()
        dataloader, testenc = get_wikitext2(self.tokenizer)

        if hasattr(self.model.config, "max_position_embeddings"):
            max_pos_embeddings = self.model.config.max_position_embeddings
        else:
            # Define a default value if the attribute is missing in the config.
            max_pos_embeddings = 1024

        model.seqlen = max_pos_embeddings  # 2048
        test_enc = testenc.input_ids
        nsamples = test_enc.numel() // model.seqlen
        if framework == "pytorch":
            dtype = next(iter(model.parameters())).dtype

        cache = {"past": None}

        def clear_past(i):
            def tmp(layer, inp, out):
                if cache["past"]:
                    cache["past"][i] = None

            return tmp

        if framework == "pytorch":
            for i, layer in enumerate(model.model.decoder.layers):
                layer.register_forward_hook(clear_past(i))

        loss = torch.nn.CrossEntropyLoss()
        nlls = []

        with torch.no_grad():
            attention_mask = torch.ones((1, test_enc.numel()))  # .cuda()
            for i in range(nsamples):
                batch = test_enc[
                    :, (i * model.seqlen) : ((i + 1) * model.seqlen)
                ]  # .cuda()

                if framework == "pytorch":
                    out = model(
                        batch,
                        attention_mask=attention_mask[
                            :, (i * model.seqlen) : ((i + 1) * model.seqlen)
                        ].reshape((1, -1)),
                    )
                else:
                    # Create attention_mask and position_ids
                    attn_mask = batch.new_ones(batch.shape)
                    pos_ids = attn_mask.long().cumsum(-1) - 1
                    pos_ids.masked_fill_(attn_mask == 0, 1)
                    # call model forward
                    out = model(batch, attention_mask=attn_mask, position_ids=pos_ids)

                shift_labels = test_enc[
                    :, (i * model.seqlen) : ((i + 1) * model.seqlen)
                ][
                    :, 1:
                ]  # .cuda()

                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(out.logits[0][:-1, :], shift_labels.view(-1))
                neg_log_likelihood = loss.float() * model.seqlen
                nlls.append(neg_log_likelihood)

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
            print("Perplexity:", ppl.item())
