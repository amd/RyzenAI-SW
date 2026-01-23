# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

import onnxruntime_genai as og
import argparse
import os
import json
import time

# Default settings
DEFAULT_MODEL_MAX_CONTEXT = 2048
DEFAULT_OUTPUT_RESERVE = 256

def get_tools_list(input_tools):
    tools_list = []
    try:
        tools_list = json.loads(input_tools)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format for tools list, expected format: '[{\"name\": \"fn1\"},{\"name\": \"fn2\"}]'")
    if len(tools_list) == 0:
        raise ValueError("Tools list cannot be empty")
    return tools_list

def create_prompt_tool_input(tools_list):
    tool_input = str(tools_list[0])
    for tool in tools_list[1:]:
        tool_input += ',' + str(tool)
    return tool_input

def get_json_grammar(input_tools):
    tools_list = get_tools_list(input_tools)
    prompt_tool_input = create_prompt_tool_input(tools_list)
    if len(tools_list) == 1:
        return prompt_tool_input, json.dumps(tools_list[0])
    else:
        output = '{ "anyOf": [' + json.dumps(tools_list[0])
        for tool in tools_list[1:]:
            output += ',' + json.dumps(tool)
        output += '] }'
        return prompt_tool_input, output

def get_lark_grammar(input_tools):
    tools_list = get_tools_list(input_tools)
    prompt_tool_input = create_prompt_tool_input(tools_list)
    if len(tools_list) == 1:
        output = ("start: TEXT | fun_call\n" "TEXT: /[^{](.|\\n)*/\n" " fun_call: <|tool_call|> %json " + json.dumps(convert_tool_to_grammar_input(tools_list[0])))
        return prompt_tool_input, output
    else:
        return prompt_tool_input, "start: TEXT | fun_call \n TEXT: /[^{](.|\n)*/ \n fun_call: <|tool_call|> %json {\"anyOf\": [" + ','.join([json.dumps(tool) for tool in tools_list]) + "]}"

def convert_tool_to_grammar_input(tool):
    param_props = {}
    required_params = []
    for param_name, param_info in tool.get("parameters", {}).items():
        param_props[param_name] = {
            "type": param_info.get("type", "string"),
            "description": param_info.get("description", "")
        }
        required_params.append(param_name)
    output_schema = {
        "description": tool.get('description', ''),
        "type": "object",
        "required": ["name", "parameters"],
        "additionalProperties": False,
        "properties": {
            "name": { "const": tool["name"] },
            "parameters": {
                "type": "object",
                "properties": param_props,
                "required": required_params,
                "additionalProperties": False
            }
        }
    }
    if len(param_props) == 0:
        output_schema["required"] = ["name"]
    return output_schema

def load_prompt_from_file(file_path):
    """Load entire file content as a single prompt."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    
    if not content:
        raise ValueError("Prompt file is empty")
    
    return content

def calculate_chat_template_overhead(tokenizer, jinja_path, template_str):
    """Calculate the number of tokens added by the chat template."""
    minimal_message = '[{"role": "user", "content": "X"}]'
    
    if os.path.exists(jinja_path) and template_str:
        templated = tokenizer.apply_chat_template(
            messages=minimal_message, 
            add_generation_prompt=True, 
            template_str=template_str
        )
    else:
        templated = tokenizer.apply_chat_template(
            messages=minimal_message, 
            add_generation_prompt=True
        )
    
    templated_tokens = tokenizer.encode(templated)
    minimal_content_tokens = tokenizer.encode("X")
    overhead = len(templated_tokens) - len(minimal_content_tokens)
    
    return overhead

def truncate_text_to_token_length(text, tokenizer, max_tokens, verbose=False):
    """Truncates text to fit within a specified token length."""
    if max_tokens is None or max_tokens <= 0:
        return text
    
    tokens = tokenizer.encode(text)
    original_length = len(tokens)
    
    if len(tokens) <= max_tokens:
        if verbose:
            print(f"Text has {original_length} tokens (no truncation needed)")
        return text
    
    truncated_tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(truncated_tokens)
    
    if verbose:
        print(f"Truncated text from {original_length} to {len(truncated_tokens)} tokens")
    
    return truncated_text

def print_input_prompt(prompt_string, tokens, description="Input"):
    """Print the full input prompt clearly formatted."""
    print()
    print("=" * 80)
    print(f"{description} PROMPT (Full chat-templated input)")
    print("=" * 80)
    print(prompt_string)
    print("=" * 80)
    print(f"Total tokens: {len(tokens)}")
    print("=" * 80)
    print()

def print_timing_stats(input_tokens_count, new_tokens_count, prompt_time, run_time):
    """Print timing statistics, handling edge cases like zero time."""
    stats = [f"Prompt length: {input_tokens_count}", f"New tokens: {new_tokens_count}"]
    
    if prompt_time > 0:
        stats.append(f"Time to first: {prompt_time:.2f}s")
        stats.append(f"Prompt tokens per second: {input_tokens_count/prompt_time:.2f} tps")
    else:
        stats.append("Time to first: <0.01s")
    
    if run_time > 0:
        stats.append(f"New tokens per second: {new_tokens_count/run_time:.2f} tps")
    elif new_tokens_count > 0:
        stats.append("New tokens per second: N/A (too fast to measure)")
    
    print(", ".join(stats))

def main(args):
    if args.verbose: print("Loading model...")
    if args.timings:
        started_timestamp = 0
        first_token_timestamp = 0

    config = og.Config(args.model_path)
    if args.execution_provider != "follow_config":
        config.clear_providers()
        if args.execution_provider != "cpu":
            if args.verbose: print(f"Setting model to {args.execution_provider}")
            config.append_provider(args.execution_provider)
    model = og.Model(config)

    if args.verbose: print("Model loaded")
    
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()
    if args.verbose: print("Tokenizer created")
    if args.verbose: print()

    search_options = {name:getattr(args, name) for name in ['do_sample', 'max_length', 'min_length', 'top_p', 'top_k', 'temperature', 'repetition_penalty'] if name in args}
    search_options['batch_size'] = 1

    if args.verbose: print(search_options)

    system_prompt = args.system_prompt
    guidance_type = ""
    prompt_tool_input = ""
    guidance_input = ""
    if args.guidance_type != "none":
        guidance_type = args.guidance_type
        if not args.guidance_info:
            raise ValueError("Guidance information is required if guidance type is provided")
        if guidance_type == "json_schema" or guidance_type == "lark_grammar":
            tools_list = args.guidance_info
            if guidance_type == "json_schema":
                prompt_tool_input, guidance_input = get_json_grammar(tools_list)
            elif guidance_type == "lark_grammar":
                prompt_tool_input, guidance_input = get_lark_grammar(tools_list)
        elif guidance_type == "regex":
            guidance_input = args.guidance_info
        else:
            raise ValueError("Guidance Type can only be [json_schema, regex, or lark_grammar]")

    params = og.GeneratorParams(model)
    params.set_search_options(**search_options)
    if guidance_type:
        params.set_guidance(guidance_type, guidance_input)
        if args.verbose:
            print("Guidance type is set to:", guidance_type)
            print("Guidance input is:", guidance_input)

    generator = og.Generator(model, params)
    if args.verbose: print("Generator created")
    if guidance_type == "json_schema" or guidance_type == "lark_grammar":
        messages = f"""[{{"role": "system", "content": "{system_prompt}", "tools": "{prompt_tool_input}"}}]"""
    else:
        messages = f"""[{{"role": "system", "content": "{system_prompt}"}}]"""

    # Apply Chat Template - load template string
    template_str = ""
    jinja_path = os.path.join(args.model_path, "chat_template.jinja")
    if os.path.exists(jinja_path):
        with open(jinja_path, "r", encoding="utf-8") as f:
            template_str = f.read()
            tokenizer_input_system_prompt = tokenizer.apply_chat_template(messages=messages, add_generation_prompt=False, template_str=template_str)
    else:
        tokenizer_input_system_prompt = tokenizer.apply_chat_template(messages=messages, add_generation_prompt=False)

    input_tokens = tokenizer.encode(tokenizer_input_system_prompt)
    if guidance_type:
        input_tokens = input_tokens[:-1]
    system_prompt_length = len(input_tokens)
    
    if args.verbose:
        print_input_prompt(tokenizer_input_system_prompt, input_tokens, "SYSTEM")
        print(f"System prompt tokens: {system_prompt_length}")
    
    generator.append_tokens(input_tokens)

    # Calculate chat template overhead once
    chat_template_overhead = calculate_chat_template_overhead(tokenizer, jinja_path, template_str)
    if args.verbose:
        print(f"Chat template overhead: {chat_template_overhead} tokens")

    # Get settings
    model_max_context = getattr(args, 'max_context', DEFAULT_MODEL_MAX_CONTEXT)
    output_reserve = getattr(args, 'output_reserve', DEFAULT_OUTPUT_RESERVE)
    
    # Calculate the maximum possible user content that fits
    max_possible_content = model_max_context - system_prompt_length - chat_template_overhead - output_reserve
    
    if max_possible_content <= 0:
        raise ValueError(f"Model max context ({model_max_context}) is too small. "
                        f"System prompt ({system_prompt_length}) + template ({chat_template_overhead}) + "
                        f"output reserve ({output_reserve}) = {system_prompt_length + chat_template_overhead + output_reserve} tokens, "
                        f"leaving no room for user content.")

    # Check if prompt file is provided
    if hasattr(args, 'prompt_file') and args.prompt_file:
        text = load_prompt_from_file(args.prompt_file)
        
        if args.verbose: 
            print(f"Loaded prompt from file: {args.prompt_file}")
            print(f"Original prompt length: {len(text)} characters")
        
        if args.timings: started_timestamp = time.time()

        # Get requested user content limit
        requested_limit = getattr(args, 'input_prompt_length', None)
        
        # Determine actual limit (cap to max possible)
        if requested_limit is not None:
            if requested_limit > max_possible_content:
                actual_limit = max_possible_content
                print(f"\n⚠️  Requested {requested_limit} tokens, but capped to {actual_limit} tokens")
                print(f"   (Model max: {model_max_context} - system: {system_prompt_length} - "
                      f"template: {chat_template_overhead} - output reserve: {output_reserve} = {max_possible_content})\n")
            else:
                actual_limit = requested_limit
        else:
            # No limit specified, use max possible
            actual_limit = max_possible_content
        
        # Truncate user content
        original_text = text
        text = truncate_text_to_token_length(text, tokenizer, actual_limit, args.verbose)
        
        actual_content_tokens = len(tokenizer.encode(text))
        
        # Print token budget summary
        print(f"\n{'='*55}")
        print(f"TOKEN BUDGET")
        print(f"{'='*55}")
        print(f"Model max context:       {model_max_context:>6} tokens")
        print(f"Output reserve:          {output_reserve:>6} tokens")
        print(f"System prompt:           {system_prompt_length:>6} tokens")
        print(f"Chat template overhead:  {chat_template_overhead:>6} tokens")
        print(f"Max user content:        {max_possible_content:>6} tokens")
        print(f"{'='*55}")
        if requested_limit is not None:
            print(f"Requested (-ipl):        {requested_limit:>6} tokens")
        print(f"Actual user content:     {actual_content_tokens:>6} tokens")
        print(f"{'='*55}\n")

        # Properly escape the text for JSON
        escaped_text = json.dumps(text)
        messages = f"""[{{"role": "user", "content": {escaped_text}}}]"""

        # Apply Chat Template AFTER truncation
        user_prompt = ""
        if os.path.exists(jinja_path):
            user_prompt = tokenizer.apply_chat_template(messages=messages, add_generation_prompt=True, template_str=template_str)
        else:
            user_prompt = tokenizer.apply_chat_template(messages=messages, add_generation_prompt=True)
        
        input_tokens = tokenizer.encode(user_prompt)
        
        # Print the full chat-templated input prompt
        print_input_prompt(user_prompt, input_tokens, "USER")
        
        # Print summary
        total_input = system_prompt_length + len(input_tokens)
        remaining_for_output = model_max_context - total_input
        print(f"TOTAL INPUT: {system_prompt_length} (system) + {len(input_tokens)} (user msg) = {total_input} tokens")
        print(f"AVAILABLE FOR OUTPUT: {remaining_for_output} tokens")
        print()
        
        generator.append_tokens(input_tokens)

        if args.verbose: print("Running generation loop ...")
        if args.timings:
            first = True
            new_tokens = []

        print("Output: ", end='', flush=True)

        try:
            while not generator.is_done():
                generator.generate_next_token()
                if args.timings:
                    if first:
                        first_token_timestamp = time.time()
                        first = False

                new_token = generator.get_next_tokens()[0]
                print(tokenizer_stream.decode(new_token), end='', flush=True)
                if args.timings: new_tokens.append(new_token)
        except KeyboardInterrupt:
            print("  --control+c pressed, aborting generation--")
        print()
        print()

        if args.timings:
            prompt_time = first_token_timestamp - started_timestamp
            run_time = time.time() - first_token_timestamp
            print_timing_stats(len(input_tokens), len(new_tokens), prompt_time, run_time)
        
        return

    # Interactive mode
    while True:
        text = input("Prompt (Use quit() to exit): ")
        if not text:
            print("Error, input cannot be empty")
            continue

        if text == "quit()":
            break

        if args.timings: started_timestamp = time.time()

        escaped_text = json.dumps(text)
        messages = f"""[{{"role": "user", "content": {escaped_text}}}]"""

        user_prompt = ""
        if os.path.exists(jinja_path):
            user_prompt = tokenizer.apply_chat_template(messages=messages, add_generation_prompt=True, template_str=template_str)
        else:
            user_prompt = tokenizer.apply_chat_template(messages=messages, add_generation_prompt=True)
        input_tokens = tokenizer.encode(user_prompt)
        
        print_input_prompt(user_prompt, input_tokens, "USER")
        
        generator.append_tokens(input_tokens)

        if args.verbose: print("Running generation loop ...")
        if args.timings:
            first = True
            new_tokens = []

        print("Output: ", end='', flush=True)

        try:
            while not generator.is_done():
                generator.generate_next_token()
                if args.timings:
                    if first:
                        first_token_timestamp = time.time()
                        first = False

                new_token = generator.get_next_tokens()[0]
                print(tokenizer_stream.decode(new_token), end='', flush=True)
                if args.timings: new_tokens.append(new_token)
        except KeyboardInterrupt:
            print("  --control+c pressed, aborting generation--")
        print()
        print()

        if args.timings:
            prompt_time = first_token_timestamp - started_timestamp
            run_time = time.time() - first_token_timestamp
            print_timing_stats(len(input_tokens), len(new_tokens), prompt_time, run_time)

        if args.rewind:
            generator.rewind_to(system_prompt_length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description="End-to-end AI Question/Answer example for gen-ai")
    parser.add_argument('-m', '--model_path', type=str, required=True, help='Onnx model folder path (must contain genai_config.json and model.onnx)')
    parser.add_argument('-pr', '--prompt_file', type=str, help='Path to .txt file containing the prompt')
    
    # Token limit settings
    parser.add_argument('-ipl', '--input_prompt_length', type=int, default=None, 
                        help='Desired user content tokens (1-2048). Automatically capped if it exceeds '
                             'what fits in model context. If not specified, uses maximum available.')
    parser.add_argument('-mc', '--max_context', type=int, default=DEFAULT_MODEL_MAX_CONTEXT,
                        help=f'Model max context length (default: {DEFAULT_MODEL_MAX_CONTEXT})')
    parser.add_argument('-or', '--output_reserve', type=int, default=DEFAULT_OUTPUT_RESERVE,
                        help=f'Tokens to reserve for output (default: {DEFAULT_OUTPUT_RESERVE})')
    
    parser.add_argument('-e', '--execution_provider', type=str, required=False, default='follow_config', choices=["cpu", "cuda", "dml", "follow_config"], help="Execution provider")
    parser.add_argument('-i', '--min_length', type=int, help='Min number of tokens to generate including the prompt')
    parser.add_argument('-l', '--max_length', type=int, help='Max number of tokens to generate including the prompt')
    parser.add_argument('-ds', '--do_sample', action='store_true', help='Do random sampling')
    parser.add_argument('-p', '--top_p', type=float, help='Top p probability to sample with')
    parser.add_argument('-k', '--top_k', type=int, help='Top k tokens to sample from')
    parser.add_argument('-t', '--temperature', type=float, help='Temperature to sample with')
    parser.add_argument('-re', '--repetition_penalty', type=float, help='Repetition penalty')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Print verbose output')
    parser.add_argument('-tm', '--timings', action='store_true', default=False, help='Print timing information')
    parser.add_argument('-gtype', '--guidance_type', type=str, default="none", choices=["none", "json_schema", "regex", "lark_grammar"], help='Guidance type')
    parser.add_argument('-ginfo', '--guidance_info', type=str, default='', help='Guidance information')
    parser.add_argument('-s', '--system_prompt', type=str, default='You are a helpful AI assistant.', help='System prompt')
    parser.add_argument('-r', '--rewind', action='store_true', default=False, help='Rewind to system prompt after each generation')
    args = parser.parse_args()
    main(args)