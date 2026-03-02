import argparse
import json
import numpy as np
import onnxruntime as ort
import torchaudio
import queue
import threading
import time
import os
from transformers import WhisperFeatureExtractor, WhisperTokenizer
from pathlib import Path
from jiwer import wer, cer
from huggingface_hub import snapshot_download

SAMPLE_RATE = 16000
CHUNK_SIZE = 1600  # 0.1 sec chunks


class WhisperONNX:
    def __init__(self, encoder_path, decoder_path,
                 model_type, encoder_providers=None, decoder_providers=None, language=None):

        self.encoder = ort.InferenceSession(encoder_path, providers=encoder_providers)
        self.decoder = ort.InferenceSession(decoder_path, providers=decoder_providers)

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/{model_type}")
        self.tokenizer = WhisperTokenizer.from_pretrained(f"openai/{model_type}")
        self.decoder_start_token = self.sot_token = self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        self.eos_token = self.tokenizer.eos_token_id
        self.max_length = min(448, self.decoder.get_inputs()[0].shape[1])
        if not isinstance(self.max_length, int):
            raise ValueError("Invalid/Dynamic input shapes")

        self.language = language
        if self.language:
            self.tokenizer.set_prefix_tokens(language=self.language, task="transcribe")
            self.initial_tokens = list(self.tokenizer.prefix_tokens)
        else:
            self.initial_tokens = [self.decoder_start_token]

    def preprocess(self, audio):
        """
        Convert raw audio to Whisper log-mel spectrogram
        """
        inputs = self.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="np")
        return inputs["input_features"]

    def encode(self, input_features):
        """
        Run encoder ONNX model
        """
        input_name = self.encoder.get_inputs()[0].name
        return self.encoder.run(None, {input_name: input_features})[0]

    def decode(self, encoder_out):
        """
        Greedy decode with fixed-length input_ids
        """
        tokens = list(self.initial_tokens)
        first_token_delay = None
        decode_start = time.time()

        # Get decoder input names
        decoder_inputs = self.decoder.get_inputs()
        input_ids_name = decoder_inputs[0].name
        encoder_out_name = decoder_inputs[1].name

        # Distinguish inputs by data type if the order is not guaranteed
        if decoder_inputs[0].type != 'tensor(int64)':
            input_ids_name, encoder_out_name = encoder_out_name, input_ids_name

        for _ in range(len(tokens), self.max_length):
            decoder_input = np.full((1, self.max_length), self.eos_token, dtype=np.int64)
            decoder_input[0, :len(tokens)] = tokens

            outputs = self.decoder.run(None, {
                input_ids_name: decoder_input,
                encoder_out_name: encoder_out
            })
            logits = outputs[0]
            next_token = int(np.argmax(logits[0, len(tokens) - 1]))

            if next_token == self.eos_token:
                break
            tokens.append(next_token)
            if first_token_delay is None:
                first_token_delay = time.time() - decode_start
        return tokens, first_token_delay

    def transcribe(self, audio, chunk_length_s=30, is_mic=False):
        """
        Full encode-decode pipeline with support for long-form transcription using chunking.
        """
        chunk_size = SAMPLE_RATE * chunk_length_s
        total_samples = len(audio)
        transcription = []
        chunk_idx = 0
        total_start_time = time.time()

        overlap = SAMPLE_RATE * 1  # Tune this
        for start in range(0, total_samples, chunk_size - overlap):
            end = min(start + chunk_size, total_samples)
            audio_chunk = audio[start:end]

            input_features = self.preprocess(audio_chunk)
            encoder_out = self.encode(input_features)
            tokens, first_token_delay = self.decode(encoder_out)
            decoded_text = self.tokenizer.decode(
                tokens[len(self.initial_tokens):],
                skip_special_tokens=True
            ).strip()
            transcription.append(decoded_text)
            chunk_idx += 1
            if not is_mic:
                if first_token_delay is not None:
                    print(f"\nPerformance Metric (Chunk {chunk_idx}):")
                    print(f" Time to First Token for this chunk: {first_token_delay:.2f} seconds")
                else:
                    print(f"\nPerformance Metric (Chunk {chunk_idx}):")
                    print(" Time to First Token for this chunk: n/a (no token before EOS)")

        total_end_time = time.time()
        input_audio_duration = total_samples / SAMPLE_RATE
        rtf = (total_end_time - total_start_time) / input_audio_duration
        if not is_mic:
            print(f" RTF: {rtf:.2f}")

        return " ".join(transcription), rtf


def evaluate(model, dataset_dir, results_dir):
    dataset_name = Path(dataset_dir).name
    wav_dir = Path(dataset_dir) / "wav"
    transcript_file = Path(dataset_dir) / "transcripts.txt"

    if not transcript_file.exists() or not wav_dir.exists():
        print(f"Missing transcripts.txt or wav folder in {dataset_dir}")
        return

    with open(transcript_file, "r", encoding="utf-8") as f:
        references = {line.split()[0]: " ".join(line.strip().split()[1:]) for line in f.readlines()}

    output_dir = Path(results_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / "results.txt"

    total_wer, total_cer, total_rtf, count = 0, 0, 0, 0

    with result_file.open("w", encoding="utf-8") as out_f:
        for wav_path in sorted(wav_dir.glob("*.wav")):
            key = wav_path.stem
            if key not in references:
                print(f"Reference for {key} not found in transcripts.txt")
                continue
            reference = references[key].lower()
            waveform, sr = torchaudio.load(str(wav_path))
            if sr != SAMPLE_RATE:
                waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)
            audio = waveform.squeeze(0).numpy()
            predicted, rtf = model.transcribe(audio)

            sample_wer = wer(reference, predicted)
            sample_cer = cer(reference, predicted)
            total_wer += sample_wer
            total_cer += sample_cer
            total_rtf += rtf
            count += 1

            out_f.write(f"{key}\n")
            out_f.write(f"Reference: {reference}\n")
            out_f.write(f"Predicted: {predicted}\n")
            out_f.write(f"WER: {sample_wer:.3f}, CER: {sample_cer:.3f}, RTF: {rtf:.3f}\n\n")

        if count:
            avg_wer = total_wer / count
            avg_cer = total_cer / count
            avg_rtf = total_rtf / count
            print(f"Evaluation completed for {count} files.")
            print(f"Average WER: {avg_wer:.3f}, Average CER: {avg_cer:.3f}, Average RTF: {avg_rtf:.3f}")
            out_f.write(f"Summary:\nAverage WER: {avg_wer:.3f}\nAverage CER: {avg_cer:.3f}\nAverage RTF: {avg_rtf:.3f}\n")
        else:
            print("No valid audio-transcript pairs found.")


def load_provider_options(config, model_name, device):
    model_key = model_name.replace("whisper-", "")
    if model_key not in config["whisper"]:
        raise ValueError(f"Model type '{model_key}' not found in config")

    if device not in config["whisper"][model_key]:
        raise ValueError(f"Device '{device}' not found in config for model type '{model_key}'")

    model_config = config["whisper"][model_key][device]
    encoder_opts = model_config["encoder"]
    decoder_opts = model_config["decoder"]

    def build_provider_opts(opts):
        if opts.get("config_file"):
            return [
                (
                    "VitisAIExecutionProvider",
                    {
                        "config_file": opts["config_file"],
                        "cache_dir": opts.get("cache_dir", ""),
                        "cache_key": opts.get("cache_key", "")
                    }
                )
            ]
        else:
            return ["CPUExecutionProvider"]

    print("Selected Provider Options: ")
    print("Decoder: ", build_provider_opts(decoder_opts))
    print("Encoder: ", build_provider_opts(encoder_opts))
    return build_provider_opts(encoder_opts), build_provider_opts(decoder_opts)


def mic_stream(model, duration=0, silence_threshold=0.01, silence_duration=5.0):
    q_audio = queue.Queue()
    stop_flag = threading.Event()

    def audio_callback(indata, frames, time, status):
        if status:
            print(status, flush=True)
        q_audio.put(indata.copy())

    def feeder():
        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                                blocksize=CHUNK_SIZE, callback=audio_callback):
                if duration > 0:
                    sd.sleep(int(duration * 1000))
                    stop_flag.set()
                else:
                    while not stop_flag.is_set():
                        sd.sleep(100)
        except sd.PortAudioError as e:
            print(f"\n Microphone error: {e}")
            print("⚠️ Could not initialize microphone. Please check your audio device settings.")
            stop_flag.set()

    threading.Thread(target=feeder, daemon=True).start()

    buffer = np.zeros((0,), dtype=np.float32)
    silence_start = None
    print("\n🎤 Real-time Transcription. Start Speaking ..\n")
    while not stop_flag.is_set():
        try:
            chunk = q_audio.get(timeout=0.1).squeeze()
            buffer = np.concatenate((buffer, chunk))

            rms = np.sqrt(np.mean(chunk ** 2))
            if rms < silence_threshold:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= silence_duration:
                    print("\n🔕 Silence detected. Stopping transcription.")
                    stop_flag.set()
                    break
            else:
                silence_start = None

            if len(buffer) >= SAMPLE_RATE * 2:
                text, _ = model.transcribe(buffer, is_mic=True)
                print(text)
                buffer = np.zeros((0,), dtype=np.float32)
        except queue.Empty:
            continue


def download_whisper_onnx(model_type: str):
    """
    Download Whisper ONNX encoder/decoder from Hugging Face if not already present.
    Returns paths to encoder and decoder model files.
    """
    hf_model_map = {
        "whisper-small": "amd/whisper-small-onnx-npu",
        "whisper-medium": "amd/whisper-medium-onnx-npu",
        "whisper-large-v3-turbo": "amd/whisper-large-turbo-onnx-npu"
    }

    repo_id = hf_model_map.get(model_type)
    if repo_id is None:
        raise ValueError(f"Unsupported model_type '{model_type}' for ONNX auto-download.")

    local_dir = snapshot_download(
        repo_id=repo_id,
    )

    # Construct paths to encoder/decoder ONNX files
    encoder_path = os.path.join(local_dir, "encoder_model.onnx")
    decoder_path = os.path.join(local_dir, "decoder_model.onnx")

    if not (os.path.exists(encoder_path) and os.path.exists(decoder_path)):
        raise FileNotFoundError(f"Could not find encoder/decoder in {local_dir}")


    return encoder_path, decoder_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="WAV file path or 'mic'")
    parser.add_argument("--encoder", help="Path to Whisper encoder ONNX model (optional, auto-download if not provided)")
    parser.add_argument("--decoder", help="Path to Whisper decoder ONNX model (optional, auto-download if not provided)")
    parser.add_argument("--model-type", required=True, default="whisper-base",
                        choices=["whisper-tiny", "whisper-base", "whisper-small",
                                 "whisper-medium", "whisper-large-v3-turbo"],
                        help="Whisper model name")
    parser.add_argument("--eval-dir", help="Dataset directory with wavs/ and transcripts.txt")
    parser.add_argument("--results-dir", default="results", help="Directory to store evaluation results")
    parser.add_argument("--config-file", default="./config/model_config.json", help="Path to Model provider configs")
    parser.add_argument("--device", choices=['cpu', 'npu'], default='cpu')
    parser.add_argument("--duration", type=int, default=0, help="Mic duration in seconds (0 = unlimited)")
    parser.add_argument("--language", help="Force decoder language (e.g., 'zh', 'ja', 'en')")
    args = parser.parse_args()

    if Path(args.config_file).exists():
        with open(args.config_file) as f:
            model_config = json.load(f)
    else:
        raise FileNotFoundError(f"Config file {args.config_file} not found")

    encoder_providers, decoder_providers = load_provider_options(
        model_config, args.model_type, args.device
    )

    # Auto-download ONNX models if not provided
    if args.encoder is None or args.decoder is None:
        print(f"Downloading ONNX models for {args.model_type} from Hugging Face ...")
        encoder_path, decoder_path = download_whisper_onnx(args.model_type)
    else:
        encoder_path, decoder_path = args.encoder, args.decoder

    model = WhisperONNX(
        encoder_path,
        decoder_path,
        args.model_type,
        encoder_providers=encoder_providers,
        decoder_providers=decoder_providers,
        language=args.language
    )

    if args.eval_dir:
        evaluate(model, args.eval_dir, args.results_dir)
        return

    if not args.input and not args.eval_dir:
        print("Error: You must provide --input (wav or mic) or --eval-dir.")
        return

    if args.input and args.input.lower() not in ['mic'] and not Path(args.input).suffix == '.wav':
        print("Error: --input must be 'mic' or path to a .wav file.")
        return

    if args.input.lower() == 'mic':
        import sounddevice as sd
        try:
            mic_stream(model, args.duration)
        except sd.PortAudioError as e:
            print("Fix your device or try using a .wav file instead of mic. Exiting")
        return
    else:
        waveform, sr = torchaudio.load(args.input)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)
        audio = waveform.squeeze(0).numpy()
        text, _ = model.transcribe(audio, chunk_length_s=30)
        print("\nTranscription:", text)


if __name__ == "__main__":
    main()
