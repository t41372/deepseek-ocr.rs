from collections.abc import Callable, Sequence

class VisionSettingsInput:
    base_size: int
    image_size: int
    crop_mode: bool
    def __init__(self, base_size: int, image_size: int, crop_mode: bool) -> None: ...

class DecodeParametersInput:
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float | None
    top_k: int | None
    repetition_penalty: float
    no_repeat_ngram_size: int | None
    seed: int | None
    use_cache: bool
    def __init__(
        self,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float | None = ...,
        top_k: int | None = ...,
        repetition_penalty: float = ...,
        no_repeat_ngram_size: int | None = ...,
        seed: int | None = ...,
        use_cache: bool = ...,
    ) -> None: ...

class DecodeOutcomeHandle:
    text: str
    prompt_tokens: int
    response_tokens: int
    generated_tokens: list[int]

class EngineHandle:
    def decode(
        self,
        prompt: str,
        images: Sequence[bytes],
        vision: VisionSettingsInput,
        decode: DecodeParametersInput,
        stream: Callable[[int, Sequence[int]], None] | None = ...,
    ) -> DecodeOutcomeHandle: ...

MOCK_MODEL_KIND: str

def create_engine(
    model_kind: str,
    *,
    config_path: str | None = ...,
    tokenizer_path: str | None = ...,
    weights_path: str | None = ...,
    snapshot_path: str | None = ...,
    device: str | None = ...,
    dtype: str | None = ...,
) -> EngineHandle: ...
def render_prompt(template: str, system_prompt: str, raw_prompt: str) -> str: ...
def normalize_text(text: str) -> str: ...
