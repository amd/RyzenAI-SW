import time

import torch
import torch.nn.functional as F

from transformers import top_k_top_p_filtering


def sample(
    logits,
    return_probs: bool = False,
    do_sample: bool = False,
    tokenizer_vocab_size: int = 32016,
    top_k: int = 10,
    top_p: float = 0.95,
):

    if return_probs:

        all_probs = logits.softmax(-1)
        if do_sample and top_k != 1 and top_p != 0.0:
            _logits = top_k_top_p_filtering(
                logits.view(-1, logits.size(-1)), top_k=top_k, top_p=top_p
            )
            if _logits.shape[1] != tokenizer_vocab_size:
                _logits = manual_padding(_logits)
            output_ids = torch.multinomial(_logits.softmax(-1), num_samples=1).view(
                logits.shape[:-1]
            )
            probs = torch.gather(all_probs, -1, output_ids.unsqueeze(-1)).squeeze(-1)
        else:
            probs, output_ids = torch.max(all_probs, dim=-1)

        return output_ids, probs

    else:

        if do_sample and top_k != 1 and top_p != 0.0:
            _logits = top_k_top_p_filtering(
                logits.view(-1, logits.size(-1)), top_k=top_k, top_p=top_p
            )
            output_ids = torch.multinomial(_logits.softmax(-1), num_samples=1).view(
                logits.shape[:-1]
            )
        else:
            output_ids = torch.argmax(logits, dim=-1)

        return output_ids


def base_generate(
    model,
    draft_model,
    tokenizer,
    input_ids,
    max_new_tokens=512,
    do_sample=False,
    top_k=10,
    top_p=0.95,
    temperature=0.0,
    early_stop=False,
    device=None,
    draft_device=None,
):
    del draft_model
    del device
    del draft_device
    eps = 1e-5
    current_input_ids = input_ids
    to_cur_device = model.device

    generate_ids = torch.empty(
        [input_ids.size(0), max_new_tokens], dtype=torch.long, device=to_cur_device
    )
    past_key_values = None
    time_rec = {}

    with torch.no_grad():
        for step in range(max_new_tokens):
            time_rec[f"base_gen_{step}_before"] = time.time()
            output = model(
                input_ids=current_input_ids,
                past_key_values=past_key_values,
                return_dict=True,
                use_cache=True,
            )
            time_rec[f"base_gen_{step}_after"] = time.time()
            logits = output["logits"][:, -1:] / (temperature + eps)
            output_ids = sample(
                logits,
                do_sample=do_sample,
                tokenizer_vocab_size=tokenizer.vocab_size,
                top_k=top_k,
                top_p=top_p,
            )
            generate_ids[:, step] = output_ids
            current_input_ids = output_ids
            past_key_values = output["past_key_values"]

            if early_stop and current_input_ids.item() == tokenizer.eos_token_id:
                break

    step = min(step + 1, max_new_tokens)
    generate_ids = generate_ids[:, :step]

    return {"generate_ids": generate_ids, "time_records": time_rec}


def max_fn(x, eps=1e-6):
    x_max = torch.where(x > 0, x, 0)
    return x_max / (torch.sum(x_max) + eps)


def manual_padding(logits):
    right_padding = 16  ##hyper-setting
    logits_padded = F.pad(logits.unsqueeze(1), (0, right_padding), value=float("-inf"))
    return logits_padded


def speculative_sample(
    target_model,
    draft_model,
    tokenizer,
    input_ids,
    max_new_tokens=512,
    early_stop=False,
    max_step_draft=8,
    th_stop_draft=0.5,
    do_sample=False,
    do_sample_draft=False,
    top_k=10,
    top_p=0.95,
    temperature=0.0,
    device=None,
    draft_device=None,
    max_seql=2048,
):
    del max_seql
    eps = 1e-5
    step = 0
    step_draft = 0
    step_verify = 0

    current_input_ids = input_ids
    prompt_length = current_input_ids.shape[1]
    to_cur_device = target_model.device

    generate_ids = torch.empty(
        [input_ids.size(0), max_new_tokens + max_step_draft],
        dtype=torch.long,
        device=to_cur_device,
    )
    draft_generate_ids = torch.empty(
        [input_ids.size(0), prompt_length + max_new_tokens + max_step_draft + 2],
        dtype=torch.long,
        device=to_cur_device,
    )
    draft_generate_probs = torch.empty(
        [input_ids.size(0), max_step_draft, target_model.config.vocab_size],
        dtype=torch.float,
        device=draft_model.lm_head.weight.device,
    )
    past_key_values = None
    draft_past_key_values = None

    n_matched = 0
    n_drafted = 0
    KK = []
    time_rec = {}
    with torch.no_grad():

        ##time: before-while
        time_rec["before-while"] = time.time()

        while step < max_new_tokens:

            ##time-while: step
            # if step <= 1:
            time_rec[f"time-while-{step}"] = time.time()

            current_input_length = current_input_ids.shape[1]
            draft_current_input_ids = current_input_ids
            draft_generate_ids[:, 0:current_input_length] = current_input_ids
            random_list = torch.rand(max_step_draft)

            ##time-inner-while: step
            time_rec[f"time-inner-{step}"] = time.time()

            for step_draft in range(max_step_draft):

                ##time00 -- start/init
                time_rec[f"time01-{step}-{step_draft}"] = time.time()

                draft_output = draft_model(
                    input_ids=draft_current_input_ids.to(
                        draft_device
                    ),  ####NOTE: Manually move.
                    past_key_values=draft_past_key_values,
                    return_dict=True,
                    use_cache=True,
                )

                ##time01
                time_rec[f"time02-{step}-{step_draft}"] = time.time()

                draft_output["logits"] = draft_output["logits"][:, -1:].to(
                    torch.float32
                ) / (temperature + eps)

                if (
                    do_sample_draft
                    and top_k != 1
                    and top_p != 0.0
                    and temperature != 0.0
                ):
                    logits = draft_output["logits"]
                    _logits = top_k_top_p_filtering(
                        logits.view(-1, logits.size(-1)), top_k=top_k, top_p=top_p
                    )
                    if (
                        _logits.shape[1] != tokenizer.vocab_size
                    ):  ##NOTE: drafted tokens do not match with target's tokenizer; needed for CodeLlama-7b/13b.
                        draft_probs = manual_padding(_logits).softmax(-1)
                    else:
                        draft_probs = _logits.softmax(-1)
                    draft_output_ids = torch.multinomial(
                        _logits.softmax(-1), num_samples=1
                    ).view(logits.shape[:-1])
                else:
                    if (
                        draft_output["logits"].shape[1] != tokenizer.vocab_size
                    ):  ##NOTE: drafted tokens do not match with target's tokenizer; needed for CodeLlama-7b/13b.
                        draft_probs = manual_padding(draft_output["logits"]).softmax(-1)
                    else:
                        draft_probs = draft_output["logits"].softmax(-1)
                    draft_output_ids, _ = sample(
                        draft_output["logits"],
                        return_probs=True,
                        do_sample=do_sample_draft,
                        tokenizer_vocab_size=tokenizer.vocab_size,
                    )
                draft_generate_ids[:, step_draft + current_input_length] = (
                    draft_output_ids
                )
                draft_generate_probs[:, step_draft] = draft_probs
                draft_current_input_ids = draft_output_ids
                draft_past_key_values = draft_output["past_key_values"]
                # origin_output_probs = torch.gather(draft_output['logits'].softmax(-1), -1, draft_output_ids.unsqueeze(-1)).squeeze(-1)
                # if (origin_output_probs.item() < th_stop_draft and (1-random_list[step_draft]) <= th_random_draft) or step + step_draft + 2 >= max_new_tokens:
                #     break
            drafted_n_tokens = step_draft + 1
            drafted_input_ids = draft_generate_ids[
                :, : drafted_n_tokens + current_input_length
            ]  # raft input + raft completion

            ##time02
            time_rec[f"time03-{step}"] = time.time()

            output = target_model(
                input_ids=drafted_input_ids.to(device),  ##NOTE: Manually move.
                past_key_values=past_key_values,
                return_dict=True,
                use_cache=True,
            )

            ##time03
            time_rec[f"time04-{step}"] = time.time()

            output["logits"] = output["logits"][:, -drafted_n_tokens - 1 :].to(
                torch.float32
            ).to(draft_device) / (temperature + eps)
            if do_sample and top_k != 1 and top_p != 0.0 and temperature != 0.0:
                logits = output["logits"]
                _logits = top_k_top_p_filtering(
                    logits.view(-1, logits.size(-1)), top_k=top_k, top_p=top_p
                )
                probs = _logits.unsqueeze(0).softmax(-1)
            else:
                probs = output["logits"].softmax(-1)

            output_ids = draft_generate_ids[
                :, current_input_length : current_input_length + drafted_n_tokens + 1
            ]
            observed_r_list = (
                probs[0, :drafted_n_tokens] / draft_generate_probs[0, :drafted_n_tokens]
            ).cpu()
            all_accept = True
            for i in range(drafted_n_tokens):
                j = output_ids[0, i]
                r = observed_r_list[i, j]
                if random_list[i] < min(1, r):
                    pass
                else:
                    all_accept = False
                    output_ids[0, i] = torch.multinomial(
                        max_fn((probs[0, i] - draft_generate_probs[0, i])),
                        num_samples=1,
                    )
                    i += 1
                    break
            if all_accept:
                i += 1
                output_ids[0, i] = sample(
                    output["logits"][0, i],
                    do_sample=do_sample,
                    tokenizer_vocab_size=tokenizer.vocab_size,
                    top_k=top_k,
                    top_p=top_p,
                )

            past_key_values = output["past_key_values"]

            # including the one generated by the base model
            max_matched = i
            max_of_max_matched = drafted_input_ids.size(1) - current_input_length + 1

            if max_of_max_matched != max_matched:
                output_ids = output_ids[:, :max_matched]
                past_key_values = [
                    (
                        k[:, :, : prompt_length + step + max_matched - 1],
                        v[:, :, : prompt_length + step + max_matched - 1],
                    )
                    for k, v in past_key_values
                ]
                draft_past_key_values = [
                    (
                        k[:, :, : prompt_length + step + max_matched - 1],
                        v[:, :, : prompt_length + step + max_matched - 1],
                    )
                    for k, v in draft_past_key_values
                ]

            generate_ids[:, step : step + output_ids.size(1)] = output_ids
            current_input_ids = output_ids[:, -1:]

            step += output_ids.size(1)

            # remove one generated by the base model
            n_matched += max_matched - 1
            KK.append(max_matched)
            n_drafted += drafted_n_tokens
            step_verify += 1

            if early_stop and tokenizer.eos_token_id in output_ids[0].tolist():
                break

    step = min(step, max_new_tokens)
    generate_ids = generate_ids[:, :step]

    return {
        "generate_ids": generate_ids,
        "matchness": n_matched / n_drafted,
        "num_drafted_tokens": n_drafted,
        "th_stop_draft": th_stop_draft,
        "matchness_list": KK,
        "time_records": time_rec,
    }


generate_fn_mapping = {
    "base": base_generate,
    "speculative_sample": speculative_sample,
}


def infer(
    target_model,
    draft_model,
    tokenizer,
    prompt,
    generate_fn="base",
    decode_timing=True,
    seed=42,
    *args,
    **kargs,
):

    if isinstance(generate_fn, str):
        generate_fn = generate_fn_mapping[generate_fn]

    if seed is not None:
        torch.manual_seed(seed)

    to_cur_device = target_model.device

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(to_cur_device)
    if decode_timing:
        tic = time.time()
    generate_dict = generate_fn(
        target_model, draft_model, tokenizer, input_ids, *args, **kargs
    )
    generate_ids = generate_dict["generate_ids"]
    if decode_timing:
        toc = time.time()
        decode_time = toc - tic
    else:
        decode_time = None
    completion = tokenizer.decode(generate_ids[0])
    generate_dict["completion"] = completion
    generate_dict["time"] = decode_time
    return generate_dict


def clip_input(
    tokenizer, prompt, task_name, max_new_tokens=512, prompt_shots="", max_seql=2048
):
    if task_name == "xsum":
        input_ids = tokenizer(
            prompt_shots + "Article: " + prompt["document"] + "\nSummary:",
            return_tensors="pt",
        ).input_ids
    elif task_name == "cnndm":
        input_ids = tokenizer(
            prompt_shots + "Article: " + prompt["article"] + "\nSummary:",
            return_tensors="pt",
        ).input_ids
    elif task_name == "samsum":
        input_ids = tokenizer(
            prompt_shots + "Dialogue: " + prompt["dialogue"] + "\nSummary:",
            return_tensors="pt",
        ).input_ids
    elif task_name == "humaneval":
        format_tabs = False  ##Originally: True
        if format_tabs:
            prompt = prompt["prompt"].replace("    ", "\t")
        else:
            prompt = prompt["prompt"]
        # print("prompt:\n", prompt)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    if len(input_ids[0]) + max_new_tokens >= max_seql:
        print("(input ids+max token)> {}".format(max_seql))
        sample_num = len(input_ids[0]) + max_new_tokens - max_seql
        input_ids = torch.cat(
            (input_ids[0][:2], input_ids[0][2:-3][:-sample_num], input_ids[0][-3:]),
            dim=0,
        ).unsqueeze(0)
    return input_ids


def infer_input_ids(
    target_model,
    draft_model,
    tokenizer,
    input_ids,
    generate_fn="base",
    decode_timing=True,
    seed=42,
    *args,
    **kargs,
):

    ##time of initiating inference
    init_infer_time = time.time()

    if isinstance(generate_fn, str):
        generate_fn = generate_fn_mapping[generate_fn]

    if seed is not None:
        torch.manual_seed(seed)

    to_cur_device = target_model.device
    input_ids = input_ids.to(to_cur_device)

    # input_ids = input_ids.to(model.device)
    if decode_timing:
        tic = time.time()
    generate_dict = generate_fn(
        target_model, draft_model, tokenizer, input_ids, *args, **kargs
    )
    generate_ids = generate_dict["generate_ids"]
    if decode_timing:
        toc = time.time()
        decode_time = toc - tic
    else:
        decode_time = None
    # new_generated_tokens = len(generate_ids[0, input_ids.size(0):])
    completion = tokenizer.decode(
        generate_ids[0, input_ids.size(0) :], skip_special_tokens=True
    )
    generate_dict["new_generated_tokens"] = generate_ids.shape[1]
    generate_dict["completion"] = completion
    generate_dict["time"] = decode_time
    generate_dict["init_infer_time"] = init_infer_time

    return generate_dict
