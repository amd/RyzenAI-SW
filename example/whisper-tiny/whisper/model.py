import io
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import requests
import onnx
import psutil
import onnxruntime as ort
from whisper.decoding import detect_language as detect_language_function, decode as decode_function
from whisper.utils import onnx_dtype_to_np_dtype_convert
from whisper.decoding import sot_l


_MODELS = {
    "tiny.en": "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/tiny.en.pt",
    "tiny": "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/tiny.pt",
    "base.en": "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/base.en.pt",
    "base": "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/base.pt",
    "small.en": "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/small.en.pt",
    "small": "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/small.pt",
    "medium.en": "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/medium.en.pt",
    "medium": "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/pt/medium.pt",
}

def model_download(name: str, onnx_file_save_path: str='.') -> onnx.ModelProto:
    onnx_file_path = f'{onnx_file_save_path}/{name}_11.onnx'
    onnx_serialized_graph = None
    if not os.path.exists(onnx_file_path):
        url = f'https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/onnx/{name}_11.onnx'
        onnx_serialized_graph = requests.get(url).content
        with io.BytesIO(onnx_serialized_graph) as f:
            onnx_graph: onnx.ModelProto = onnx.load(f)
            onnx.save(onnx_graph, f'{onnx_file_save_path}/{name}_11.onnx')
    else:
        onnx_graph: onnx.ModelProto = onnx.load(onnx_file_path)
        onnx_serialized_graph = onnx._serialize(onnx_graph)
    return onnx_serialized_graph

def load_local_model(name: str) -> onnx.ModelProto:
    onnx_file_path = name
    onnx_serialized_graph = None
    if not os.path.exists(onnx_file_path):
        raise FileNotFoundError(f"Model {onnx_file_path} not found.")
    else:
        onnx_graph: onnx.ModelProto = onnx.load(onnx_file_path)
        onnx_serialized_graph = onnx._serialize(onnx_graph)
    return onnx_serialized_graph

def load_model(name: str, onnx_encoder_path: str, onnx_decoder_path: str, encoder_target: str, decoder_target: str):
    """
    Load a Whisper ASR model

    Parameters
    ----------
    name : str
        one of the official model names listed by `whisper.available_models()`, or
        path to a model checkpoint containing the model dimensions and the model state_dict.

    Returns
    -------
    model : Whisper
        The Whisper ASR model instance
    """

    if name == "tiny":
        dims_config = {'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 512, 'n_audio_state': 384, 'n_audio_head': 6, 'n_audio_layer': 4, 'n_text_ctx': 512, 'n_text_state': 384, 'n_text_head': 6, 'n_text_layer': 4}
    # Fix length of "n_text_ctx" and "n_audio_ctx"
    elif name == "tiny.en":
        dims_config = {'n_mels': 80, 'n_vocab': 51864, 'n_audio_ctx': 512, 'n_audio_state': 384, 'n_audio_head': 6, 'n_audio_layer': 4, 'n_text_ctx': 512, 'n_text_state': 384, 'n_text_head': 6, 'n_text_layer': 4}
    elif name == "base":
        dims_config = {'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 512, 'n_audio_head': 8, 'n_audio_layer': 6, 'n_text_ctx': 448, 'n_text_state': 512, 'n_text_head': 8, 'n_text_layer': 6}
    elif name == "base.en":
        dims_config = {'n_mels': 80, 'n_vocab': 51864, 'n_audio_ctx': 1500, 'n_audio_state': 512, 'n_audio_head': 8, 'n_audio_layer': 6, 'n_text_ctx': 448, 'n_text_state': 512, 'n_text_head': 8, 'n_text_layer': 6}
    elif name == "small":
        dims_config = {'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 768, 'n_audio_head': 12, 'n_audio_layer': 12, 'n_text_ctx': 448, 'n_text_state': 768, 'n_text_head': 12, 'n_text_layer': 12}
    elif name == "small.en":
        dims_config = {'n_mels': 80, 'n_vocab': 51864, 'n_audio_ctx': 1500, 'n_audio_state': 768, 'n_audio_head': 12, 'n_audio_layer': 12, 'n_text_ctx': 448, 'n_text_state': 768, 'n_text_head': 12, 'n_text_layer': 12}
    elif name == "medium":
        dims_config = {'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 1024, 'n_audio_head': 16, 'n_audio_layer': 24, 'n_text_ctx': 448, 'n_text_state': 1024, 'n_text_head': 16, 'n_text_layer': 24}
    elif name == "medium.en":
        dims_config = {'n_mels': 80, 'n_vocab': 51864, 'n_audio_ctx': 1500, 'n_audio_state': 1024, 'n_audio_head': 16, 'n_audio_layer': 24, 'n_text_ctx': 448, 'n_text_state': 1024, 'n_text_head': 16, 'n_text_layer': 24}
    else:
        raise ValueError(f"model type {name} not supported")

    dims = ModelDimensions(**dims_config)
    model = Whisper(dims=dims, model_name=name, onnx_encoder_path=onnx_encoder_path, onnx_decoder_path=onnx_decoder_path, encoder_target=encoder_target, decoder_target=decoder_target)
    return model

def available_models() -> List[str]:
    """Returns the names of available models"""
    return list(_MODELS.keys())

@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class OnnxAudioEncoder():
    def __init__(
        self,
        model: str,
        model_path: str,
        target: str,
    ):
        sess_options = ort.SessionOptions()
        if target == "cpu":
            self.provider = "CPUExecutionProvider"
            self.provider_options = {}
        else:
            self.provider = "VitisAIExecutionProvider"
            self.provider_options = {
                'config_file': '.\\other_libs_qdq\\vaip_config_gemm_asr_encoder.json'
            }
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            sess_options.add_session_config_entry("session.disable_quant_qdq", "1")
        # sess_options.intra_op_num_threads = psutil.cpu_count(logical=False)
        # sess_options.intra_op_num_threads = psutil.cpu_count(1)
        self.sess = \
            ort.InferenceSession(
                # path_or_bytes=load_local_model(name=model_path),
                model_path,
                providers=[
                    self.provider,
                    # 'VitisAIExecutionProvider',
                    # 'CPUExecutionProvider',
                ],
                sess_options=sess_options,
                provider_options=[self.provider_options],
            )
        self.inputs = {
            input.name: onnx_dtype_to_np_dtype_convert(input.type) \
                for input in self.sess.get_inputs()
        }

    def __call__(
        self,
        mel: np.ndarray
    ) -> np.ndarray:
        result: np.ndarray = \
            self.sess.run(
                [],
                input_feed={
                    "mel": mel.astype(self.inputs["mel"]),
                }
            )[0]
        return result


class OnnxTextDecoder():
    def __init__(
        self,
        model: str,
        model_path: str,
        target: str,
    ):
        sess_options = ort.SessionOptions()
        if target == "cpu":
            self.provider = "CPUExecutionProvider"
            self.provider_options = {}
        else:
            self.provider = "VitisAIExecutionProvider"
            self.provider_options = {
                'config_file': '.\\other_libs_qdq\\vaip_config_gemm_asr_decoder.json'
            }
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            sess_options.add_session_config_entry("session.disable_quant_qdq", "1") 
        # print("decoder target: ", target)
        # sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)
        # sess_options.intra_op_num_threads = psutil.cpu_count(1)
        self.sess = \
            ort.InferenceSession(
                # path_or_bytes=load_local_model(name=model_path),
                model_path,
                providers=[
                    self.provider,
                    # 'VitisAIExecutionProvider',
                    # 'CPUExecutionProvider'
                ],
                sess_options=sess_options,
                provider_options=[self.provider_options],
            )
        self.inputs = {
            input.name: onnx_dtype_to_np_dtype_convert(input.type) \
                for input in self.sess.get_inputs()
        }

    def __call__(
        self,
        x: np.ndarray,
        xa: np.ndarray,
        kv_cache: np.ndarray,
        offset: int,
        mask: np.ndarray,
        pe: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        outputs = \
            self.sess.run(
                [], # Don't fill outputs name, since there are multi outputs.
                input_feed={
                    "tokens": x.astype(self.inputs["tokens"]),
                    "audio_features": xa.astype(self.inputs["audio_features"]),
                    "kv_cache": kv_cache.astype(self.inputs["kv_cache"]),
                    # "offset": np.array(offset, dtype=self.inputs["offset"]),
                    "mask": mask.astype(self.inputs["mask"]),
                    "pe": pe.astype(self.inputs["pe"]),
                }
            )
        logits: np.ndarray = outputs[0]
        output_kv_cache: np.ndarray = outputs[1]
        kv_s: List[np.ndarray] = [tmp for tmp in outputs[2:]]
        return logits, output_kv_cache, kv_s


class Whisper():
    def __init__(
        self,
        dims: ModelDimensions,
        model_name: str,
        onnx_encoder_path: str,
        onnx_decoder_path: str,
        encoder_target: str,
        decoder_target: str,
    ):
        self.model_name = model_name
        self.dims = dims

        # encoder target: `cpu` or `aie`
        self.encoder = OnnxAudioEncoder(model=model_name, model_path=onnx_encoder_path, target=encoder_target)
        # decoder target: `cpu` or `aie`
        self.decoder = OnnxTextDecoder(model=model_name, model_path=onnx_decoder_path, target=decoder_target)

    def embed_audio(
        self,
        mel: np.ndarray,
    ):
        return self.encoder(mel)

    def logits(
        self,
        tokens: np.ndarray,
        audio_features: np.ndarray,
    ):
        kv_cache = self.new_kv_cache(tokens.shape[0], tokens.shape[-1])
        output, _ = self.decoder(tokens, audio_features, kv_cache=kv_cache, offset=0)
        return output

    def __call__(
        self,
        mel: np.ndarray,
        tokens: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        kv_cache = self.new_kv_cache(tokens.shape[0], tokens.shape[-1])
        output, _ = self.decoder(tokens, self.encoder(mel), kv_cache=kv_cache, offset=0)
        return output

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def new_kv_cache(
        self,
        n_group: int,
        length: int,
    ):
        length = 451    # 3 (sot) + 1 (sot prev) + 224 + 223
        length = 512 - sot_l 
        if self.model_name == "tiny.en" or self.model_name == "tiny":
            size = [8, n_group, length, 384]
        elif self.model_name == "base.en" or self.model_name == "base":
            size = [12, n_group, length, 512]
        elif self.model_name == "small.en" or self.model_name == "small":
            size = [24, n_group, length, 768]
        elif self.model_name == "medium.en" or self.model_name == "medium":
            size = [48, n_group, length, 1024]
        elif self.model_name == "large":
            size = [64, n_group, length, 1280]
        else:
            raise ValueError(f"Unsupported model type: {self.type}")
        return np.zeros(size, dtype=np.float32)

    detect_language = detect_language_function
    # transcribe = transcribe_function
    decode = decode_function
