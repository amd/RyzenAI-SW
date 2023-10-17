from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Optional, Sequence, Union, TYPE_CHECKING

import numpy as np
import time

from whisper.audio import CHUNK_LENGTH
from whisper.tokenizer import Tokenizer, get_tokenizer
from whisper.utils import compression_ratio, DisplayCPU

# if TYPE_CHECKING:
#     from whisper.model import Whisper

# some hard code since the static shape
static = True

WITHOUT_TIMESTAMPS=True
if WITHOUT_TIMESTAMPS:
   sot_l = 4
else:
   sot_l = 3

def softmax(x, dim=-1):
    e_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
    return e_x / (np.sum(e_x, axis=dim, keepdims=True))

def log_softmax(x, dim=-1):
    y = softmax(x, dim=dim)
    return np.log(y)

def numpy_categorical_sample(logits, temperature):
    logits /= temperature
    probs = softmax(logits, dim=-1)
    return np.array([np.random.choice(len(p), p=p) for p in probs])


def detect_language(model: "Whisper", mel: np.ndarray, tokenizer: Tokenizer = None) -> Tuple[np.ndarray, List[dict]]:
    """
    Detect the spoken language in the audio, and return them as list of strings, along with the ids
    of the most probable language tokens and the probability distribution over all language tokens.
    This is performed outside the main decode loop in order to not interfere with kv-caching.

    Returns
    -------
    language_tokens : np.ndarray, shape = (n_audio,)
        ids of the most probable language tokens, which appears after the startoftranscript token.
    language_probs : List[Dict[str, float]], length = n_audio
        list of dictionaries containing the probability distribution over all languages.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer(model.is_multilingual)
    if tokenizer.language is None or tokenizer.language_token not in tokenizer.sot_sequence:
        raise ValueError(f"This model doesn't have language tokens so it can't perform lang id")

    single = mel.ndim == 2
    if single:
        mel = mel[np.newaxis, ...]

    # skip encoder forward pass if already-encoded audio features were given
    if mel.shape[-2:] != (model.dims.n_audio_ctx, model.dims.n_audio_state):
        mel = model.encoder(mel)

    # forward pass using a single token, startoftranscript
    n_audio = mel.shape[0]
    x = np.array([[tokenizer.sot]] * n_audio)  # [n_audio, 1]
    logits = model.logits(x, mel)[:, 0]

    # collect detected languages; suppress all non-language tokens
    mask = np.ones(logits.shape[-1], dtype=np.bool_)
    mask[list(tokenizer.all_language_tokens)] = False
    logits[:, mask] = -np.inf
    language_tokens = logits.argmax(axis=-1)
    language_token_probs = softmax(logits, dim=-1)
    language_probs = [
        {
            c: language_token_probs[i, j].item()
            for j, c in zip(tokenizer.all_language_tokens, tokenizer.all_language_codes)
        }
        for i in range(n_audio)
    ]

    if single:
        language_tokens = language_tokens[0]
        language_probs = language_probs[0]

    return language_tokens, language_probs


@dataclass(frozen=True)
class DecodingOptions:
    task: str = "transcribe"  # whether to perform X->X "transcribe" or X->English "translate"
    language: Optional[str] = None  # language that the audio is in; uses detected language if None

    # sampling-related options
    temperature: float = 0.0
    sample_len: Optional[int] = None  # maximum number of tokens to sample
    best_of: Optional[int] = None     # number of independent samples to collect, when t > 0
    beam_size: Optional[int] = None   # number of beams in beam search, when t == 0
    patience: Optional[float] = None  # patience in beam search (https://arxiv.org/abs/2204.05424)

    # options for ranking generations (either beams or best-of-N samples)
    length_penalty: Optional[float] = None   # "alpha" in Google NMT, None defaults to length norm

    # prompt, prefix, and token suppression
    prompt: Optional[Union[str, List[int]]] = None   # text or tokens for the previous context
    prefix: Optional[Union[str, List[int]]] = None   # text or tokens to prefix the current context
    suppress_blank: bool = True                      # this will suppress blank outputs

    # list of tokens ids (or comma-separated token ids) to suppress
    # "-1" will suppress a set of symbols as defined in `tokenizer.non_speech_tokens()`
    suppress_tokens: Optional[Union[str, Iterable[int]]] = "-1"

    # timestamp sampling options
    without_timestamps: bool = WITHOUT_TIMESTAMPS              # use <|notimestamps|> to sample text tokens only
    max_initial_timestamp: Optional[float] = 0.0  # the initial timestamp cannot be later than this


@dataclass(frozen=True)
class DecodingResult:
    audio_features: np.ndarray
    language: str
    language_probs: Optional[Dict[str, float]] = None
    tokens: List[int] = field(default_factory=list)
    text: str = ""
    avg_logprob: float = np.nan
    no_speech_prob: float = np.nan
    temperature: float = np.nan
    compression_ratio: float = np.nan
    encoder_time: float = 0
    decoder_time: float = 0


class Inference:
    def logits(self, tokens: np.ndarray, audio_features: np.ndarray) -> np.ndarray:
        """Perform a forward pass on the decoder and return per-token logits"""
        raise NotImplementedError

    def rearrange_kv_cache(self, source_indices) -> None:
        """Update the key-value cache according to the updated beams"""
        raise NotImplementedError

    def cleanup_caching(self) -> None:
        """Clean up any resources or hooks after decoding is finished"""
        pass


class OnnxInference(Inference):
    def __init__(self, model: "Whisper", initial_token_length: int):
        self.model: "Whisper" = model
        self.initial_token_length = initial_token_length
        self.kv_cache = None
        self.n_t_ctx =  self.model.dims.n_text_ctx
        self.mask = np.full((self.n_t_ctx, self.n_t_ctx), -np.inf)
        self.mask[np.tril_indices(self.n_t_ctx, k=0)] = 0
        self.pe = np.load("./tiny_pe.npy")
        self.offset = None

    def logits(self, tokens: np.ndarray, audio_features: np.ndarray) -> np.ndarray:
        n_group = tokens.shape[0]
        # print("token shape: ", tokens.shape)
        if self.kv_cache is None:
            self.kv_cache = self.model.new_kv_cache(n_group, self.initial_token_length)
            offset = 0
            length = self.initial_token_length
        else:
            # offset = self.kv_cache.shape[2]
            offset = self.offset
            length = offset + 1
            # new_kv_cache = self.model.new_kv_cache(n_group, offset + 1)
            # new_kv_cache[:, :, :offset, :] = self.kv_cache # Since hard code of model.new_kv_cache
            # self.kv_cache = new_kv_cache
            if static:
                offset = offset - (sot_l - 1) # for length fixed to sot_l, need to shift back to sot_l-1

        if tokens.shape[-1] > self.initial_token_length:
            # only need to use the last token except in the first forward pass
            if static:
                tokens = tokens[:, -sot_l:] # To avoid dynamic shape, fix q length to 3;
            else:
                tokens = tokens[:, -1:] 

        mask = np.concatenate((np.zeros((sot_l, offset)), self.mask[:sot_l, :self.n_t_ctx-offset]), axis=1)
        mask = np.concatenate((np.full((sot_l, self.n_t_ctx-sot_l-offset), -np.inf), mask[:, :sot_l+offset]), axis=1)

        # print("audio features shape: ", audio_features.shape)
        pe = self.pe[offset : offset + tokens.shape[-1]]
        # print("offset: ", offset, "token shape: ", tokens.shape[-1], "pe shape: ", pe.shape)
        start = time.perf_counter()
        output, self.kv_cache, kv_s = self.model.decoder(tokens, audio_features, kv_cache=self.kv_cache, offset=offset, mask=mask, pe=pe)
        end = time.perf_counter()
        decoder_time = end - start
        
        id = 0
        init_s = 1
        for cache in kv_s: #[1,3,384]
            self.kv_cache[id, :, :-init_s, :] = np.copy(self.kv_cache[id, :, init_s:, :])  #kv_cache [8,1,512,384]
            self.kv_cache[id, :, -init_s:, :] = cache[:, :init_s,:]  #kv_cache [8,1,512,384]
            id += 1

        # self.kv_cache = self.kv_cache[:, :, :length, :] # Since hard code of model.new_kv_cache
        self.offset = length
        return output, decoder_time

    def cleanup_caching(self):
        self.kv_cache = None
        self.offset = None

    def rearrange_kv_cache(self, source_indices):
        self.kv_cache = self.kv_cache[:, source_indices]


class SequenceRanker:
    def rank(self, tokens: List[List[np.ndarray]], sum_logprobs: List[List[float]]) -> List[int]:
        """
        Given a list of groups of samples and their cumulative log probabilities,
        return the indices of the samples in each group to select as the final result
        """
        raise NotImplementedError


class MaximumLikelihoodRanker(SequenceRanker):
    """
    Select the sample with the highest log probabilities, penalized using either
    a simple length normalization or Google NMT paper's length penalty
    """

    def __init__(self, length_penalty: Optional[float]):
        self.length_penalty = length_penalty

    def rank(self, tokens: List[List[np.ndarray]], sum_logprobs: List[List[float]]):
        def scores(logprobs, lengths):
            result = []
            for logprob, length in zip(logprobs, lengths):
                if self.length_penalty is None:
                    penalty = length
                else:
                    # from the Google NMT paper
                    penalty = ((5 + length) / 6) ** self.length_penalty
                result.append(logprob / penalty)
            return result

        # get the sequence with the highest score
        lengths = [[len(t) for t in s] for s in tokens]
        return [np.argmax(scores(p, l)) for p, l in zip(sum_logprobs, lengths)]


class TokenDecoder:
    def reset(self):
        """Initialize any stateful variables for decoding a new sequence"""

    def update(self, tokens: np.ndarray, logits: np.ndarray, sum_logprobs: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Specify how to select the next token, based on the current trace and logits

        Parameters
        ----------
        tokens : np.ndarray, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        logits : np.ndarray, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        sum_logprobs : np.ndarray, shape = (n_batch)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : np.ndarray, shape = (n_batch, current_sequence_length + 1)
            the tokens, appended with the selected next token

        completed : bool
            True if all sequences has reached the end of text

        """
        raise NotImplementedError

    def finalize(
        self, tokens: np.ndarray, sum_logprobs: np.ndarray
    ) -> Tuple[Sequence[Sequence[np.ndarray]], List[List[float]]]:
        """Finalize search and return the final candidate sequences

        Parameters
        ----------
        tokens : np.ndarray, shape = (n_audio, n_group, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence

        sum_logprobs : np.ndarray, shape = (n_audio, n_group)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Sequence[Sequence[np.ndarray]], length = n_audio
            sequence of Tensors containing candidate token sequences, for each audio input

        sum_logprobs : List[List[float]], length = n_audio
            sequence of cumulative log probabilities corresponding to the above

        """
        raise NotImplementedError


class GreedyDecoder(TokenDecoder):
    def __init__(self, temperature: float, eot: int):
        self.temperature = temperature
        self.eot = eot

    def update(self, tokens: np.ndarray, logits: np.ndarray, sum_logprobs: np.ndarray) -> Tuple[np.ndarray, bool]:
        temperature = self.temperature
        if temperature == 0:
            next_tokens = logits.argmax(dim=-1)
        else:
            next_tokens = numpy_categorical_sample(logits, temperature)

        logprobs = log_softmax(logits, dim=-1)
        current_logprobs = logprobs[np.arange(logprobs.shape[0]), next_tokens]
        sum_logprobs += current_logprobs * (tokens[:, -1] != self.eot)

        next_tokens[tokens[:, -1] == self.eot] = self.eot
        tokens = np.concatenate([tokens, next_tokens[:, None]], axis=-1)

        completed = (tokens[:, -1] == self.eot).all()
        return tokens, completed

    def finalize(self, tokens: np.ndarray, sum_logprobs: np.ndarray):
        # make sure each sequence has at least one EOT token at the end
        tokens = np.pad(tokens, (0, 1), constant_values=self.eot)
        return tokens, sum_logprobs.tolist()


class BeamSearchDecoder(TokenDecoder):
    def __init__(self, beam_size: int, eot: int, inference: Inference, patience: Optional[float] = None):
        self.beam_size = beam_size
        self.eot = eot
        self.inference = inference
        self.patience = patience or 1.0
        self.max_candidates: int = round(beam_size * self.patience)
        self.finished_sequences = None

        assert self.max_candidates > 0, f"Invalid beam size ({beam_size}) or patience ({patience})"

    def reset(self):
        self.finished_sequences = None

    def update(self, tokens: np.ndarray, logits: np.ndarray, sum_logprobs: np.ndarray) -> Tuple[np.ndarray, bool]:
        if tokens.shape[0] % self.beam_size != 0:
            raise ValueError(f"{tokens.shape}[0] % {self.beam_size} != 0")

        n_audio = tokens.shape[0] // self.beam_size
        if self.finished_sequences is None:  # for the first update
            self.finished_sequences = [{} for _ in range(n_audio)]

        logprobs = log_softmax(logits, dim=-1)
        next_tokens, source_indices, finished_sequences = [], [], []
        for i in range(n_audio):
            scores, sources, finished = {}, {}, {}

            # STEP 1: calculate the cumulative log probabilities for possible candidates
            for j in range(self.beam_size):
                idx = i * self.beam_size + j
                prefix = list(tokens[idx])
                topk_values, topk_indices = \
                    -np.partition(-logprobs[idx], self.beam_size + 1)[:self.beam_size + 1], np.argpartition(-logprobs[idx], self.beam_size + 1)[:self.beam_size + 1]

                sort_indices = np.argsort(-topk_values)
                topk_values = topk_values[sort_indices]
                topk_indices = topk_indices[sort_indices]
                for logprob, token in zip(topk_values, topk_indices):
                    new_logprob = (sum_logprobs[idx] + logprob)
                    sequence = tuple(prefix + [token])
                    scores[sequence] = new_logprob
                    sources[sequence] = idx

            # STEP 2: rank the candidates and keep the top beam_size sequences for each audio
            saved = 0
            for sequence in sorted(scores, key=scores.get, reverse=True):
                if sequence[-1] == self.eot:
                    finished[sequence] = scores[sequence]
                else:
                    sum_logprobs[len(next_tokens)] = scores[sequence]
                    next_tokens.append(sequence)
                    source_indices.append(sources[sequence])

                    saved += 1
                    if saved == self.beam_size:
                        break

            finished_sequences.append(finished)

        tokens = np.array(next_tokens)
        self.inference.rearrange_kv_cache(source_indices)

        # add newly finished sequences to self.finished_sequences
        assert len(self.finished_sequences) == len(finished_sequences)
        for previously_finished, newly_finished in zip(self.finished_sequences, finished_sequences):
            for seq in sorted(newly_finished, key=newly_finished.get, reverse=True):
                if len(previously_finished) >= self.max_candidates:
                    break  # the candidate list is full
                previously_finished[seq] = newly_finished[seq]

        # mark as completed if all audio has enough number of samples
        completed = all(
            len(sequences) >= self.max_candidates for sequences in self.finished_sequences
        )
        return tokens, completed

    def finalize(self, preceding_tokens: np.ndarray, sum_logprobs: np.ndarray):
        # collect all finished sequences, including patience, and add unfinished ones if not enough
        sum_logprobs = sum_logprobs
        for i, sequences in enumerate(self.finished_sequences):
            if len(sequences) < self.beam_size:  # when not enough sequences are finished
                for j in list(np.argsort(sum_logprobs[i]))[::-1]:
                    sequence = list(preceding_tokens[i, j]) + [self.eot]
                    sequences[tuple(sequence)] = sum_logprobs[i][j]
                    if len(sequences) >= self.beam_size:
                        break

        tokens: List[List[np.ndarray]] = [
            [np.array(seq) for seq in sequences.keys()] for sequences in self.finished_sequences
        ]
        sum_logprobs: List[List[float]] = [
            list(sequences.values()) for sequences in self.finished_sequences
        ]
        return tokens, sum_logprobs


class LogitFilter:
    def apply(self, logits: np.ndarray, tokens: np.ndarray) -> None:
        """Apply any filtering or masking to logits in-place

        Parameters
        ----------
        logits : np.ndarray, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        tokens : np.ndarray, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        """
        raise NotImplementedError


class SuppressBlank(LogitFilter):
    def __init__(self, tokenizer: Tokenizer, sample_begin: int):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin

    def apply(self, logits: np.ndarray, tokens: np.ndarray):
        if tokens.shape[1] == self.sample_begin:
            logits[:, self.tokenizer.encode(" ") + [self.tokenizer.eot]] = -np.inf


class SuppressTokens(LogitFilter):
    def __init__(self, suppress_tokens: Sequence[int]):
        self.suppress_tokens = list(suppress_tokens)

    def apply(self, logits: np.ndarray, tokens: np.ndarray):
        logits[:, self.suppress_tokens] = -np.inf


class ApplyTimestampRules(LogitFilter):
    def __init__(
        self, tokenizer: Tokenizer, sample_begin: int, max_initial_timestamp_index: Optional[int]
    ):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin
        self.max_initial_timestamp_index = max_initial_timestamp_index

    def apply(self, logits: np.ndarray, tokens: np.ndarray):
        # suppress <|notimestamps|> which is handled by without_timestamps
        if self.tokenizer.no_timestamps is not None:
            logits[:, self.tokenizer.no_timestamps] = -np.inf

        # timestamps have to appear in pairs, except directly before EOT; mask logits accordingly
        for k in range(tokens.shape[0]):
            seq = [t for t in tokens[k, self.sample_begin :].tolist()]
            last_was_timestamp = len(seq) >= 1 and seq[-1] >= self.tokenizer.timestamp_begin
            penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= self.tokenizer.timestamp_begin

            if last_was_timestamp:
                if penultimate_was_timestamp:  # has to be non-timestamp
                    logits[k, self.tokenizer.timestamp_begin :] = -np.inf
                else:  # cannot be normal text tokens
                    logits[k, : self.tokenizer.eot] = -np.inf

        # apply the `max_initial_timestamp` option
        if tokens.shape[1] == self.sample_begin and self.max_initial_timestamp_index is not None:
            last_allowed = self.tokenizer.timestamp_begin + self.max_initial_timestamp_index
            logits[:, last_allowed + 1 :] = -np.inf

        # if sum of probability over timestamps is above any other token, sample timestamp
        logprobs = log_softmax(logits, dim=-1)
        for k in range(tokens.shape[0]):
            max_val = np.max(logprobs[k, self.tokenizer.timestamp_begin :], axis=-1, keepdims=True)
            timestamp_logprob = np.squeeze(max_val) + np.log(np.sum(np.exp(logprobs[k, self.tokenizer.timestamp_begin :] - max_val), axis=-1))
            max_text_token_logprob = np.max(logprobs[k, : self.tokenizer.timestamp_begin])
            if timestamp_logprob > max_text_token_logprob:
                logits[k, : self.tokenizer.timestamp_begin] = -np.inf


class DecodingTask:
    inference: Inference
    sequence_ranker: SequenceRanker
    decoder: TokenDecoder
    logit_filters: List[LogitFilter]

    def __init__(self, model: "Whisper", options: DecodingOptions):
        self.model = model

        language = options.language or "en"
        tokenizer = get_tokenizer(model.is_multilingual, language=language, task=options.task)
        self.tokenizer: Tokenizer = tokenizer
        self.options: DecodingOptions = self._verify_options(options)

        self.n_group: int = options.beam_size or options.best_of or 1
        self.n_ctx: int = model.dims.n_text_ctx
        self.sample_len: int = options.sample_len or model.dims.n_text_ctx // 2

        self.sot_sequence: Tuple[int] = tokenizer.sot_sequence
        if self.options.without_timestamps:
            self.sot_sequence = tokenizer.sot_sequence_including_notimestamps

        self.initial_tokens: Tuple[int] = self._get_initial_tokens()
        self.sample_begin: int = len(self.initial_tokens)
        self.sot_index: int = self.initial_tokens.index(tokenizer.sot)

        # inference: implements the forward pass through the decoder, including kv caching
        self.inference = OnnxInference(model, len(self.initial_tokens))

        # sequence ranker: implements how to rank a group of sampled sequences
        self.sequence_ranker = MaximumLikelihoodRanker(options.length_penalty)

        # decoder: implements how to select the next tokens, given the autoregressive distribution
        if options.beam_size is not None:
            self.decoder = BeamSearchDecoder(
                options.beam_size, tokenizer.eot, self.inference, options.patience
            )
        else:
            self.decoder = GreedyDecoder(options.temperature, tokenizer.eot)

        # logit filters: applies various rules to suppress or penalize certain tokens
        self.logit_filters = []
        if self.options.suppress_blank:
            self.logit_filters.append(SuppressBlank(self.tokenizer, self.sample_begin))
        if self.options.suppress_tokens:
            self.logit_filters.append(SuppressTokens(self._get_suppress_tokens()))
        if not options.without_timestamps:
            precision = CHUNK_LENGTH / model.dims.n_audio_ctx  # usually 0.02 seconds
            max_initial_timestamp_index = None
            if options.max_initial_timestamp:
                max_initial_timestamp_index = round(self.options.max_initial_timestamp / precision)
            self.logit_filters.append(
                ApplyTimestampRules(tokenizer, self.sample_begin, max_initial_timestamp_index)
            )

    def _verify_options(self, options: DecodingOptions) -> DecodingOptions:
        if options.beam_size is not None and options.best_of is not None:
            raise ValueError("beam_size and best_of can't be given together")
        if options.temperature == 0:
            if options.best_of is not None:
                raise ValueError("best_of with greedy sampling (T=0) is not compatible")
        if options.patience is not None and options.beam_size is None:
            raise ValueError("patience requires beam_size to be given")
        if options.length_penalty is not None and not (0 <= options.length_penalty <= 1):
            raise ValueError("length_penalty (alpha) should be a value between 0 and 1")

        return options

    def _get_initial_tokens(self) -> Tuple[int]:
        tokens = list(self.sot_sequence)
        prefix = self.options.prefix
        prompt = self.options.prompt

        if prefix:
            prefix_tokens = (
                self.tokenizer.encode(" " + prefix.strip()) if isinstance(prefix, str) else prefix
            )
            if self.sample_len is not None:
                max_prefix_len = self.n_ctx // 2 - self.sample_len
                prefix_tokens = prefix_tokens[-max_prefix_len:]
            tokens = tokens + prefix_tokens

        if prompt:
            prompt_tokens = (
                self.tokenizer.encode(" " + prompt.strip()) if isinstance(prompt, str) else prompt
            )
            tokens = [self.tokenizer.sot_prev] + prompt_tokens[-(self.n_ctx // 2 - 1) :] + tokens

        return tuple(tokens)

    def _get_suppress_tokens(self) -> Tuple[int]:
        suppress_tokens = self.options.suppress_tokens

        if isinstance(suppress_tokens, str):
            suppress_tokens = [int(t) for t in suppress_tokens.split(",")]

        if -1 in suppress_tokens:
            suppress_tokens = [t for t in suppress_tokens if t >= 0]
            suppress_tokens.extend(self.tokenizer.non_speech_tokens)
        elif suppress_tokens is None or len(suppress_tokens) == 0:
            suppress_tokens = []  # interpret empty string as an empty list
        else:
            assert isinstance(suppress_tokens, list), "suppress_tokens must be a list"

        suppress_tokens.extend(
            [self.tokenizer.sot, self.tokenizer.sot_prev, self.tokenizer.sot_lm]
        )
        if self.tokenizer.no_speech is not None:
            # no-speech probability is collected separately
            suppress_tokens.append(self.tokenizer.no_speech)

        return tuple(sorted(set(suppress_tokens)))

    def _get_audio_features(self, mel: np.ndarray):
        encoder_time = 0
        if mel.shape[-2:] == (self.model.dims.n_audio_ctx, self.model.dims.n_audio_state):
            # encoded audio features are given; skip audio encoding
            audio_features = mel
        else:
            start = time.perf_counter()
            audio_features = self.model.encoder(mel)
            end = time.perf_counter()
            encoder_time = end - start

        return audio_features, encoder_time

    def _detect_language(self, audio_features: np.ndarray, tokens: np.ndarray):
        languages = [self.options.language] * audio_features.shape[0]
        lang_probs = None

        if self.options.language is None or self.options.task == "lang_id":
            lang_tokens, lang_probs = self.model.detect_language(audio_features, self.tokenizer)
            languages = [max(probs, key=probs.get) for probs in lang_probs]
            if self.options.language is None:
                tokens[:, self.sot_index + 1] = lang_tokens  # write language tokens

        return languages, lang_probs

    def _main_loop(self, audio_features: np.ndarray, tokens: np.ndarray):
        assert audio_features.shape[0] == tokens.shape[0]
        n_batch = tokens.shape[0]
        sum_logprobs: np.ndarray = np.zeros(n_batch)
        no_speech_probs = [np.nan] * n_batch
        total_decoder_time: float = 0
        try:
            for i in range(self.sample_len):
                logits, decoder_time = self.inference.logits(tokens, audio_features)
                total_decoder_time += decoder_time

                if i == 0 and self.tokenizer.no_speech is not None:  # save no_speech_probs
                    probs_at_sot = softmax(logits[:, self.sot_index], dim=-1)
                    no_speech_probs = list(probs_at_sot[:, self.tokenizer.no_speech])

                # now we need to consider the logits at the last token only
                logits = logits[:, -1]

                # apply the logit filters, e.g. for suppressing or applying penalty to
                for logit_filter in self.logit_filters:
                    logit_filter.apply(logits, tokens)

                # expand the tokens tensor with the selected next tokens
                tokens, completed = self.decoder.update(tokens, logits, sum_logprobs)

                if completed or tokens.shape[-1] > self.n_ctx:
                    break
        finally:
            self.inference.cleanup_caching()

        return tokens, sum_logprobs, no_speech_probs, total_decoder_time

    def run(self, mel: np.ndarray) -> List[DecodingResult]:
        self.decoder.reset()
        tokenizer: Tokenizer = self.tokenizer
        n_audio: int = mel.shape[0]

        audio_features, encoder_time = self._get_audio_features(mel)  # encoder forward pass
        token = np.array([self.initial_tokens])
        tokens: np.ndarray = np.broadcast_to(token, (n_audio, token.shape[1]))

        # detect language if requested, overwriting the language token
        # languages, language_probs = self._detect_language(audio_features, tokens)
        languages=['en']
        if self.options.task == "lang_id":
            return [
                DecodingResult(audio_features=features, language=language, language_probs=probs, encoder_time=encoder_time)
                for features, language, probs in zip(audio_features, languages, language_probs)
            ]

        # repeat the audio & text tensors by the group size, for beam search or best-of-n sampling
        # print("self.n_group: ", self.n_group)
        audio_features = np.repeat(a=audio_features, repeats=self.n_group, axis=0)
        tokens = np.repeat(a=tokens, repeats=self.n_group, axis=0)

        # call the main sampling loop
        tokens, sum_logprobs, no_speech_probs, decoder_time = self._main_loop(audio_features, tokens)

        # reshape the tensors to have (n_audio, n_group) as the first two dimensions
        audio_features = audio_features[:: self.n_group]
        no_speech_probs = no_speech_probs[:: self.n_group]
        assert audio_features.shape[0] == len(no_speech_probs) == n_audio

        tokens = tokens.reshape(n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)

        # get the final candidates for each group, and slice between the first sampled token and EOT
        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)
        tokens: List[List[np.ndarray]] = [
            [t[self.sample_begin : (t == tokenizer.eot).nonzero()[0][0]] for t in s] for s in tokens
        ]

        # select the top-ranked sample in each group
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens: List[List[int]] = [list(t[i]) for i, t in zip(selected, tokens)]
        texts: List[str] = [tokenizer.decode(t).strip() for t in tokens]

        sum_logprobs: List[float] = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        avg_logprobs: List[float] = [lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)]

        fields = (texts, languages, tokens, audio_features, avg_logprobs, no_speech_probs)
        if len(set(map(len, fields))) != 1:
            raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")

        return [
            DecodingResult(
                audio_features=features,
                language=language,
                tokens=tokens,
                text=text,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                temperature=self.options.temperature,
                compression_ratio=compression_ratio(text),
                encoder_time=encoder_time,
                decoder_time=decoder_time,
            )
            for text, language, tokens, features, avg_logprob, no_speech_prob in zip(*fields)
        ]


def decode(model: "Whisper", mel: np.ndarray, options: DecodingOptions = DecodingOptions()) -> Union[DecodingResult, List[DecodingResult]]:
    """
    Performs decoding of 30-second audio segment(s), provided as Mel spectrogram(s).

    Parameters
    ----------
    model: Whisper
        the Whisper model instance

    mel: np.ndarray, shape = (80, 3000) or (*, 80, 3000)
        A tensor containing the Mel spectrogram(s)

    options: DecodingOptions
        A dataclass that contains all necessary options for decoding 30-second segments

    Returns
    -------
    result: Union[DecodingResult, List[DecodingResult]]
        The result(s) of decoding contained in `DecodingResult` dataclass instance(s)
    """
    single = mel.ndim == 2
    if single:
        mel = mel[np.newaxis, ...]

    result = DecodingTask(model, options).run(mel)

    if single:
        result = result[0]

    return result
