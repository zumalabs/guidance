try:
    import torch
except ImportError:
    raise ImportError(
        "Could not import `torch` package. "
        "Please install it from https://pytorch.org/get-started/locally/"
    )

import copy
from typing import List, Optional, Union

from transformers import GenerationConfig, LogitsProcessorList, StoppingCriteriaList, AutoTokenizer
from transformers.generation import SampleDecoderOnlyOutput
from transformers.generation.streamers import BaseStreamer

from model import ExLlama, ExLlamaConfig, ExLlamaCache
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
import cuda_ext


class Config:
    # sample
    top_k: int = 40
    top_p: float = 0.95
    temperature: float = 0.8
    repetition_penalty: float = 1.1
    last_n_tokens: int = 64
    seed: int = -1

    # eval
    batch_size: int = 8
    threads: int = -1

    # generate
    max_new_tokens: int = 512
    stop = None
    stream: bool = False
    reset: bool = True

    # model
    context_length: int = -1
    gpu_layers: int = 0


class Model:
    def __init__(self, llm: ExLlamaGenerator) -> None:
        self._llm = llm
        self.config = Config()
        self.config.vocab_size = llm.model.config.vocab_size
        self.config.pad_token_id = llm.model.config.pad_token_id
        self.device = None
        self._past = None

    def prepare_inputs_for_generation(self):
        raise NotImplementedError()

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        streamer: Optional[BaseStreamer] = None,
        **kwargs,
    ) -> Union[SampleDecoderOnlyOutput, torch.LongTensor]:
        llm, config = self._llm, self.config
        reset = config.reset

        assert "input_ids" not in kwargs, "TODO"
        assert inputs.shape[0] == 1, "Batch size must be 1."

        if kwargs.get("temperature") == 0.0:
            kwargs["temperature"] = 0.5
        for k in [
            "top_k",
            "top_p",
            "temperature",
            "repetition_penalty",
            "max_new_tokens",
        ]:
            kwargs[k] = kwargs.get(k, getattr(config, k))
        if generation_config is None:
            generation_config = GenerationConfig(**kwargs)
        else:
            generation_config = copy.deepcopy(generation_config)
            generation_config.update(**kwargs)

        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )
        stopping_criteria = (
            stopping_criteria
            if stopping_criteria is not None
            else StoppingCriteriaList()
        )

        output_scores = (
            output_scores
            if output_scores is not None
            else generation_config.output_scores
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else generation_config.return_dict_in_generate
        )

        scores = [] if return_dict_in_generate and output_scores else None
        # tokens = inputs.flatten().tolist()
        # n_past = len(self._past)
        # if n_past > 0 and tokens[:n_past] == self._past:
        #     tokens = tokens[n_past:]
        #     reset = False

        # if reset:
        #     llm.reset()

        # if tokens:
        #     llm.eval(tokens)

        n_past = self._past.size(dim=1) if self._past is not None else 0
        if n_past > 0 and torch.equal(self._past, inputs[:, :n_past]):
            reset = False
            llm.gen_feed_tokens(inputs[:, n_past:])

        if reset:
            llm.reset()
            llm.end_beam_search()
            llm.sequence = inputs.clone()
            llm.sequence_actual = inputs.clone()
            llm.cache.current_seq_len = 0
            llm.model.forward(llm.sequence[:, :-1], llm.cache, preprocess_only = True, lora = llm.lora)

        max_new_tokens = min(generation_config.max_new_tokens, llm.model.config.max_seq_len - inputs.shape[1])

        count = 0
        while count < max_new_tokens:
            llm.end_beam_search()
            logits = llm.model.forward(llm.sequence[:, -1:], llm.cache, lora = llm.lora)
            cuda_ext.ext_apply_rep_penalty_mask_cpu(llm.sequence,
                    llm.settings.token_repetition_penalty_max,
                    llm.settings.token_repetition_penalty_sustain,
                    llm.settings.token_repetition_penalty_decay,
                    logits
            )

            logits = logits.squeeze(0)
            logits = logits_processor(input_ids=inputs, scores=logits)
            if return_dict_in_generate and output_scores:
                scores.append(logits)
            logits = logits.unsqueeze(0)

            # logits[:, :, llm.tokenizer.bos_token_id] = -10000.0

            token, _ = llm.batched_sample(
                logits,
                generation_config.temperature,
                generation_config.top_k,
                generation_config.top_p,
                0.0,
                0,
                # repetition_penalty=generation_config.repetition_penalty,
            )
            # token = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
            llm.gen_accept_token(token)

            inputs = llm.sequence_actual
            if stopping_criteria(inputs, scores):
                break

            if token.squeeze().item() == llm.tokenizer.eos_token_id:
                break

            count += 1

        if inputs[:, -1].item() == llm.tokenizer.eos_token_id:
            llm.gen_rewind(1)
        self._past = llm.sequence_actual

        if return_dict_in_generate:
            return SampleDecoderOnlyOutput(sequences=inputs, scores=scores)
        return inputs

    # Added for Microsoft Guidance
    # https://github.com/microsoft/guidance/blob/d6b855aa625677f806fc51ec7238d2a38df594ea/guidance/llms/_transformers.py#L158
    def prepare_inputs_for_generation(self):
        raise NotImplementedError()

    # Added for Microsoft Guidance
    # https://github.com/microsoft/guidance/blob/d6b855aa625677f806fc51ec7238d2a38df594ea/guidance/llms/_transformers.py#L171
    def _update_model_kwargs_for_generation(self):
        raise NotImplementedError()


class Tokenizer:
    def __init__(self, tokenizer: ExLlamaTokenizer, config: ExLlamaConfig) -> None:
        self.tokenizer = tokenizer
        self.vocab_size = config.vocab_size
        self.eos_token_id = tokenizer.eos_token_id
        self.eos_token = self.decode([self.eos_token_id]) or "</s>"  # TODO
        self.max_sequence_length = config.max_seq_len

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)[0].tolist()

    def decode(
        self,
        token_ids: Union[int, List[int], torch.Tensor],
    ) -> str:
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids)
        return self.tokenizer.decode(token_ids.unsqueeze(0))

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]]
    ) -> Union[str, List[str]]:
        if isinstance(ids, int):
            return self.decode([ids])
        else:
            return self.decode(ids)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens)

    def convert_tokens_to_ids(
        self, tokens: Union[str, List[str]]
    ) -> Union[int, List[int]]:
        if tokens is None:
            return None
        elif isinstance(tokens, str):
            return self.encode(tokens)
        else:
            return [self.encode(token) for token in tokens]
