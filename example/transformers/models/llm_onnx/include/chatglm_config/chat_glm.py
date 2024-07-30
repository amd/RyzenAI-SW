from typing import Dict

from optimum.exporters.onnx import main_export
from optimum.exporters.onnx.config import TextDecoderOnnxConfig
from optimum.utils import DummyPastKeyValuesGenerator, NormalizedTextConfig

from transformers import AutoConfig


class ChatGLM2DummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):

    def generate(self, input_name: str, framework: str = "pt"):
        past_key_shape = (
            self.batch_size,
            self.num_attention_heads,
            self.hidden_size // self.num_attention_heads,
            self.sequence_length,
        )
        past_value_shape = (
            self.batch_size,
            self.num_attention_heads,
            self.sequence_length,
            self.hidden_size // self.num_attention_heads,
        )
        return [
            (
                self.random_float_tensor(past_key_shape, framework=framework),
                self.random_float_tensor(past_value_shape, framework=framework),
            )
            for _ in range(self.num_layers)
        ]


class CustomChatGLM2OnnxConfig(TextDecoderOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (
        ChatGLM2DummyPastKeyValuesGenerator,
    ) + TextDecoderOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES
    DUMMY_PKV_GENERATOR_CLASS = ChatGLM2DummyPastKeyValuesGenerator

    DEFAULT_ONNX_OPSET = 15  # aten::tril operator requires opset>=14
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        hidden_size="hidden_size",
        num_layers="num_layers",
        num_attention_heads="num_attention_heads",
    )

    def add_past_key_values(
        self, inputs_or_outputs: Dict[str, Dict[int, str]], direction: str
    ):

        if direction not in ["inputs", "outputs"]:
            raise ValueError(
                f'direction must either be "inputs" or "outputs", but {direction} was given'
            )

        if direction == "inputs":
            decoder_sequence_name = "past_sequence_length"
            name = "past_key_values"
        else:
            decoder_sequence_name = "past_sequence_length + 1"
            name = "present"

        for i in range(self._normalized_config.num_layers):
            inputs_or_outputs[f"{name}.{i}.key"] = {
                0: "batch_size",
                3: decoder_sequence_name,
            }
            inputs_or_outputs[f"{name}.{i}.value"] = {
                0: "batch_size",
                2: decoder_sequence_name,
            }
