import io
import os
import sys
import json
# sys.path.append(os.getcwd())
import argparse
import time
import warnings
import multiprocessing
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.filterwarnings(action="ignore", message="None of PyTorch, TensorFlow >= 2.0, or Flax have been found.*")

import numpy as np
import tqdm
import pyaudio
import soundfile as sf
import speech_recognition as sr

from whisper.model import load_model, available_models
from whisper.audio import SAMPLE_RATE, N_FRAMES, HOP_LENGTH, pad_or_trim, log_mel_spectrogram
from whisper.decoding import DecodingOptions, DecodingResult
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer
from whisper.utils import exact_div, format_timestamp, optional_int, optional_float, str2bool, DisplayCPU
from whisper.metrics import postprocess, word_error_rate

if TYPE_CHECKING:
    from whisper.model import Whisper

class Color:
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERCE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'

def transcribe(
    *,
    model: "Whisper",
    audio: Union[str, np.ndarray],
    verbose: Optional[bool] = None,
    temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold: Optional[float] = 2.4,
    logprob_threshold: Optional[float] = -1.0,
    no_speech_threshold: Optional[float] = 0.6,
    condition_on_previous_text: bool = True,
    **decode_options,
):
    """
    Transcribe an audio file using Whisper

    Parameters
    ----------
    model: Whisper
        The Whisper model instance

    audio: Union[str, np.ndarray]
        The path to the audio file to open, or the audio waveform

    verbose: bool
        Whether to display the text being decoded to the console. If True, displays all the details,
        If False, displays minimal details. If None, does not display anything

    temperature: Union[float, Tuple[float, ...]]
        Temperature for sampling. It can be a tuple of temperatures, which will be successfully used
        upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.

    compression_ratio_threshold: float
        If the gzip compression ratio is above this value, treat as failed

    logprob_threshold: float
        If the average log probability over sampled tokens is below this value, treat as failed

    no_speech_threshold: float
        If the no_speech probability is higher than this value AND the average log probability
        over sampled tokens is below `logprob_threshold`, consider the segment as silent

    condition_on_previous_text: bool
        if True, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

    decode_options: dict
        Keyword arguments to construct `DecodingOptions` instances

    Returns
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """
    mel: np.ndarray = log_mel_spectrogram(audio)

    if decode_options.get("language", None) is None:
        if verbose:
            print("Detecting language using up to the first 30 seconds. Use `--language` to specify the language")
        segment = pad_or_trim(mel, N_FRAMES)
        _, probs = model.detect_language(segment)
        decode_options["language"] = max(probs, key=probs.get)
        if verbose is not None:
            print(f"Detected language: {LANGUAGES[decode_options['language']].title()}")

    mel = mel[np.newaxis, ...]
    language = decode_options["language"]
    task = decode_options.get("task", "transcribe")
    tokenizer = get_tokenizer(model.is_multilingual, language=language, task=task)

    def decode_with_fallback(segment: np.ndarray) -> List[DecodingResult]:
        temperatures = [temperature] if isinstance(temperature, (int, float)) else temperature
        kwargs = {**decode_options}
        total_encoder_time, total_decoder_time = 0, 0
        t = temperatures[0]
        if t == 0:
            best_of = kwargs.pop("best_of", None)
        else:
            best_of = kwargs.get("best_of", None)

        options = DecodingOptions(**kwargs, temperature=t)
        results = model.decode(segment, options)
        # every result of results are same for the encoder_time and decoder_time
        count = 0
        total_decoder_time += results[0].decoder_time
        total_encoder_time += results[0].encoder_time

        kwargs.pop("beam_size", None)  # no beam search for t > 0
        kwargs.pop("patience", None)  # no patience for t > 0
        kwargs["best_of"] = best_of  # enable best_of for t > 0
        for t in temperatures[1:]:
            needs_fallback = [
                compression_ratio_threshold is not None
                and result.compression_ratio > compression_ratio_threshold
                or logprob_threshold is not None
                and result.avg_logprob < logprob_threshold
                for result in results
            ]
            if any(needs_fallback):
                print("call fallback")
                options = DecodingOptions(**kwargs, temperature=t)
                retries = model.decode(segment[needs_fallback], options)
                # every result of results are same for the encoder_time and decoder_time
                total_decoder_time += retries[0].decoder_time
                total_encoder_time += retries[0].encoder_time
                for retry_index, original_index in enumerate(np.nonzero(needs_fallback)[0]):
                    results[original_index] = retries[retry_index]

        return results, total_encoder_time, total_decoder_time

    seek = 0
    input_stride = exact_div(
        N_FRAMES, model.dims.n_audio_ctx
    )  # mel frames per output token: 2
    time_precision = (
        input_stride * HOP_LENGTH / SAMPLE_RATE
    )  # time per output token: 0.02 (seconds)
    all_tokens = []
    all_segments = []
    prompt_reset_since = 0

    initial_prompt = decode_options.pop("initial_prompt", None) or []
    if initial_prompt:
        initial_prompt = tokenizer.encode(" " + initial_prompt.strip())
        all_tokens.extend(initial_prompt)

    def add_segment(
        *, start: float, end: float, text_tokens: np.ndarray, result: DecodingResult
    ):
        text = tokenizer.decode([token for token in text_tokens if token < tokenizer.eot])
        if len(text.strip()) == 0:  # skip empty text output
            return

        all_segments.append(
            {
                "id": len(all_segments),
                "seek": seek,
                "start": start,
                "end": end,
                "text": text,
                "tokens": result.tokens,
                "temperature": result.temperature,
                "avg_logprob": result.avg_logprob,
                "compression_ratio": result.compression_ratio,
                "no_speech_prob": result.no_speech_prob,
            }
        )
        if verbose:
            print(f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}", flush=True)

    # show the progress bar when verbose is False (otherwise the transcribed text will be printed)
    num_frames = mel.shape[-1]
    previous_seek_value = seek
    audio_duration = num_frames * HOP_LENGTH / SAMPLE_RATE

    # with tqdm.tqdm(total=num_frames, unit='frames', disable=verbose is not False) as pbar:
    encoder_time, decoder_time = 0, 0
    for _ in range(num_frames):
        # print(f"num_frames: {num_frames}")
        while seek < num_frames:
            timestamp_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
            segment = pad_or_trim(mel[:, :, seek:], N_FRAMES)
            segment_duration = segment.shape[-1] * HOP_LENGTH / SAMPLE_RATE
            decode_options["prompt"] = all_tokens[prompt_reset_since:]
            results, once_encoder_time, once_decoder_time = decode_with_fallback(segment)
            encoder_time += once_encoder_time
            decoder_time += once_decoder_time
            result = results[0]
            tokens = result.tokens

            if no_speech_threshold is not None:
                # no voice activity check
                should_skip = result.no_speech_prob > no_speech_threshold
                if logprob_threshold is not None and result.avg_logprob > logprob_threshold:
                    # don't skip if the logprob is high enough, despite the no_speech_prob
                    should_skip = False

                if should_skip:
                    seek += segment.shape[-1]  # fast-forward to the next segment boundary
                    continue

            timestamp_tokens: np.ndarray = np.greater_equal(tokens, tokenizer.timestamp_begin)
            consecutive = np.add(np.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0], 1)
            if len(consecutive) > 0:  # if the output contains two consecutive timestamp tokens
                last_slice = 0
                for current_slice in consecutive:
                    sliced_tokens = tokens[last_slice:current_slice]
                    start_timestamp_position = (
                        sliced_tokens[0] - tokenizer.timestamp_begin
                    )
                    end_timestamp_position = (
                        sliced_tokens[-1] - tokenizer.timestamp_begin
                    )
                    add_segment(
                        start=timestamp_offset + start_timestamp_position * time_precision,
                        end=timestamp_offset + end_timestamp_position * time_precision,
                        text_tokens=sliced_tokens[1:-1],
                        result=result,
                    )
                    last_slice = current_slice
                last_timestamp_position = (
                    tokens[last_slice - 1] - tokenizer.timestamp_begin
                )
                seek += last_timestamp_position * input_stride
                all_tokens.extend(list(tokens[: last_slice + 1]))
            else:
                duration = segment_duration
                tokens = np.asarray(tokens) if isinstance(tokens, list) else tokens
                timestamps = tokens[
                    np.ravel_multi_index(np.nonzero(timestamp_tokens), timestamp_tokens.shape)
                ]
                if len(timestamps) > 0:
                    # no consecutive timestamps but it has a timestamp; use the last one.
                    # single timestamp at the end means no speech after the last timestamp.
                    last_timestamp_position = timestamps[-1] - tokenizer.timestamp_begin
                    duration = last_timestamp_position * time_precision

                add_segment(
                    start=timestamp_offset,
                    end=timestamp_offset + duration,
                    text_tokens=tokens,
                    result=result,
                )

                seek += segment.shape[-1]
                all_tokens.extend(list(tokens))

            if not condition_on_previous_text or result.temperature > 0.5:
                # do not feed the prompt tokens if a high temperature was used
                prompt_reset_since = len(all_tokens)

            # update progress bar
            # pbar.update(min(num_frames, seek) - previous_seek_value)
            previous_seek_value = seek

    return dict(text=tokenizer.decode(all_tokens[len(initial_prompt):]), segments=all_segments, language=language, 
                audio_duration=audio_duration, encoder_time=encoder_time, decoder_time=decoder_time)

def cli():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    def check_range(value):
        ivalue = int(value)
        if ivalue < 0:
            raise argparse.ArgumentTypeError(f"{value} must be greater than or equal to 0.")
        return ivalue
    parser.add_argument("--audio", nargs="*", type=str, help="Specify the path to at least one or more audio files (wav, mp4, mp3, etc.). e.g. --audio aaa.mp4 bbb.mp3 ccc.mp4")
    parser.add_argument("--output_dir", "-o", type=str, default=".", help="directory to save the outputs")
    parser.add_argument("--target",  type=str, default="cpu-aie", choices=["aie-cpu", "cpu-aie", "aie-aie"], help="which target to run encoder and decoder models")
    parser.add_argument("--librispeech",  type=str, default=None, help="test WER of LibriSpeech dataset if you set path of LibriSpeech dataset.")
    parser.add_argument("--test_num",  type=check_range, default=0, help="dataset samples count to calculate WER, 0 means whole dataset.")

    args = parser.parse_args().__dict__
    
    # hard code for some options
    args["mode"] = "audio"
    args["model"] = "tiny"
    args["task"] = "transcribe"
    args["language"] = "en"
    args["verbose"] = False
    args["temperature"] = 0
    args["best_of"] = 5
    args["beam_size"] = 5
    args["patience"] = None
    args["length_penalty"] = 0.08
    args["suppress_tokens"] = "-1"
    args["initial_prompt"] = None
    args["condition_on_previous_text"] = None
    args["temperature_increment_on_fallback"] = None
    args["compression_ratio_threshold"] = 2.4
    args["logprob_threshold"] = -1
    args["no_speech_threshold"] = 0.6

    mode: str = args.pop("mode")
    model_name: str = args.pop("model")
    output_dir: str = args.pop("output_dir")
    target: str = args.pop("target")
    test_num: int = args.pop("test_num")

    os.makedirs(output_dir, exist_ok=True)

    target_ls: List = target.split("-")
    encoder_target: str = target_ls[0]
    decoder_target: str = target_ls[1]

    # hard code the model path
    if encoder_target == "cpu":
        onnx_encoder_path: str = os.path.join("models", "float-encoder.onnx")
    else:
        onnx_encoder_path: str = os.path.join("models", "quant-encoder.onnx")

    if decoder_target == "cpu":
        onnx_decoder_path: str = os.path.join("models", "float-decoder.onnx")
    else:
        onnx_decoder_path: str = os.path.join("models", "quant-decoder.onnx")

    if model_name.endswith(".en") and args["language"] not in {"en", "English"}:
        warnings.warn(f"{model_name} is an English-only model but receipted '{args['language']}'; using English instead.")
        args["language"] = "en"

    temperature = args.pop("temperature")
    temperature_increment_on_fallback = args.pop("temperature_increment_on_fallback")
    if temperature_increment_on_fallback is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback))
    else:
        temperature = [temperature]

    model = load_model(model_name, onnx_encoder_path, onnx_decoder_path, encoder_target, decoder_target)
    librispeech = args.pop("librispeech", None)
    pid = os.getpid()
    manager = multiprocessing.Manager()
    is_running = manager.Value('b', True)
    shared_list = manager.list()
    # display_cpu = DisplayCPU(pid, is_running, shared_list)
    if mode == 'audio':
        audios = args.pop("audio")
        if not audios and not librispeech:
            print(f"{Color.RED}ERROR:{Color.RESET} Specify the path to at least one or more audio files (mp4, mp3, etc.). e.g. --audio aaa.mp4 bbb.mp3 ccc.mp4")
            sys.exit(0)
        if audios:
            for audio_path in audios:
                start_time = time.perf_counter()
                result = \
                    transcribe(
                        model=model,
                        audio=audio_path,
                        temperature=temperature,
                        **args,
                    )
                end_time = time.perf_counter()
                audio_duration = result["audio_duration"]
                encoder_time = round(result["encoder_time"], 2)
                decoder_time = round(result["decoder_time"], 2)
                pred_duration = end_time - start_time
                other_process_time = round((pred_duration - encoder_time - decoder_time), 2)
                real_time_factor = round(pred_duration/audio_duration, 3)
                if audio_duration > 10.24:
                    print(f"{Color.RED}[Warning]:{Color.RESET} The audio duration more than 10.24 seconds, the part which after 10.24 seconds maybe decoded not well.")
                print('---------------------result----------------------')
                print("Prediction Result: ", result['text'])
                print(f"Encoder running time: {encoder_time}s, Decoder running time: {decoder_time}s, Other process time: {other_process_time}s")
                print(f"Real time factor: {real_time_factor}, Audio duration: {audio_duration}s, Decoding time: {round(pred_duration, 2)}s")
                print('-------------------------------------------------')
            
            audio_basename = os.path.basename(audio_path)

            # save text result
            with open(os.path.join(output_dir, audio_basename + ".txt"), "w", encoding="utf-8") as wf:
                wf.write(f"Audio file name: {audio_basename}\n")
                wf.write(f"Prediction: {result['text']}\n")
                wf.write(f"Encoder running time: {encoder_time}s, Decoder running time: {decoder_time}s, Other process time: {other_process_time}s\n")
                wf.write(f"Real time factor: {real_time_factor}, Audio duration: {audio_duration}s, Decoding time: {round(pred_duration, 2)}s\n")

            print(f"[Info] Save the transcribe result into {os.path.join(output_dir, audio_basename)}.txt successfully.")
    
    if librispeech:
        with open(librispeech, 'r') as f:
            metainfo = json.load(f)

        predictions = []
        rtf_ls = []
        labels = []
        count = 0 
        warmup_steps = 3
        warmup_step = 0
        condition_on_previous_text = args.pop('condition_on_previous_text')

        save_json_name = "test_librispeech_results.json"
        save_json_path = os.path.join(output_dir, save_json_name)
        final_results = []
        total_decoding_time = 0
        total_encoder_running_time = 0
        total_decoder_running_time = 0
        total_other_process_time = 0
        
        # Becasue we fix input length to 1024, we make some filter of the input wavs
        metainfo = [info for info in metainfo if info["original_duration"] <= 10]
        # metainfo = metainfo[1:]
        # warm up steps
        print(f"Warm up some steps...")
        with tqdm.tqdm(total=warmup_steps, desc="warm up", unit="audio") as pbar:
            for i in range(warmup_steps):
                info = metainfo[i]
                input = info['files'][0]['fname']
                label = info['transcript']
                result = transcribe(model=model, audio=input, temperature=temperature, condition_on_previous_text=False, **args)
                pbar.update(1)
        print("Warm up done.")

        if test_num > 0:
            count = min(len(metainfo), test_num)
        else:
            count = len(metainfo)
        # display_cpu.start()
        for info in tqdm.tqdm(metainfo[:count], desc="Dataset", unit="audio"):
            input = info['files'][0]['fname']
            label = info['transcript']
            # display_cpu = DisplayCPU()
            # display_cpu.start()
            # try:
            start_time = time.perf_counter()
            result = transcribe(model=model, audio=input, temperature=temperature, condition_on_previous_text=False, **args)
            end_time = time.perf_counter()
            # finally:
                # display_cpu.show()
            pred_duration = end_time - start_time
            total_decoding_time += pred_duration

            prediction = result['text']
            audio_duration = result["audio_duration"]
            encoder_time = round(result["encoder_time"], 2)
            decoder_time = round(result["decoder_time"], 2)
            other_process_time = round((pred_duration - encoder_time - decoder_time), 2)
            total_encoder_running_time += encoder_time
            total_decoder_running_time += decoder_time
            total_other_process_time += other_process_time
            rtf = round(pred_duration/audio_duration, 3)
            rtf_ls.append(rtf)
            
            prediction = postprocess(prediction)
            labels.append(label)
            predictions.append(prediction)

            sample_result = {}
            sample_result["fname"] = input
            sample_result["transcript"] = label
            sample_result["prediction"] = prediction
            sample_result["real time factor"] = rtf
            sample_result["audio duration"] = str(audio_duration) + "s"
            sample_result["decoding time"] = str(round(pred_duration, 2)) + "s"
            sample_result["encoder running time"] = str(encoder_time) + "s"
            sample_result["decoder running time"] = str(decoder_time) + "s"
            sample_result["other process time"] = str(other_process_time) + "s"
            final_results.append(sample_result)

            print('---------------------result----------------------')
            print('transcript: ', label)
            print('prediction: ', prediction)
            print(f"Encoder running time: {encoder_time}s, Decoder running time: {decoder_time}s, Other process time: {other_process_time}s")
            print(f"real time factor: {rtf}, Audio duration: {audio_duration}s, Decoding time: {round(pred_duration, 2)}s")
            print('-------------------------------------------------')

        # display_cpu.stop()

        rtf_50 = round(np.percentile(rtf_ls, 50), 3)
        rtf_90 = round(np.percentile(rtf_ls, 90), 3)
        rtf_99 = round(np.percentile(rtf_ls, 99), 3)
        rtf_mean = round(np.mean(rtf_ls), 3)

        total_encoder_running_time = round(total_encoder_running_time, 2)
        total_decoder_running_time = round(total_decoder_running_time, 2)
        total_other_process_time = round(total_other_process_time, 2)
        total_decoding_time = round(total_decoding_time, 2)

        wer, scores, words = word_error_rate(predictions, labels)
        wer = round(wer, 4)
        final_result = {}
        final_result["Final Results"] = {"word error rate": wer, "total test samples": count, "error scores": scores,
                                        "total test words": words, "RTF Average": rtf_mean, "RTF 50%": rtf_50,
                                        "RTF 90%": rtf_90, "RTF 99%": rtf_99, 
                                        "total decoding time": str(total_decoding_time)+"s",
                                        "total encoder running time": str(total_encoder_running_time)+"s",
                                        "total decoder running time": str(total_decoder_running_time)+"s",
                                        "total other process time": str(total_other_process_time)+"s"}
        
        final_results.insert(0, final_result)
        with open(save_json_path, "w") as wf:
            json.dump(final_results, wf)

        print(f"WER>>>>>:{wer}, error scores: {scores}, total words in test set: {words}, total samples: {count}")
        print(f"RTF Average: {rtf_mean}, RTF 50%: {rtf_50}, RTF 90%: {rtf_90}, RTF 99%: {rtf_99}")
        print(f"total encoder running time: {total_encoder_running_time}s")
        print(f"total decoder running time: {total_decoder_running_time}s")
        print(f"total other process time: {total_other_process_time}s")
        print(f'Total decoding time: {total_decoding_time}s')
        print(f"[Info] Test results are save into {save_json_path} successfully.")


if __name__ == '__main__':
    cli()
