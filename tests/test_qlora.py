import pytest

from rosai.prepare import InstructionPair

# Rendered from Qwen/Qwen3-8B tokenizer_config.json chat_template
# with add_generation_prompt=False. The empty think block is the
# non-thinking assistant turn.
_THINKING_OFF = (
    "<|im_start|>user\n"
    "Quote the Governing Law clause or answer none.<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n\n</think>\n\n"
    "Nevada<|im_end|>\n"
)


def test_prepared_pair_formats_as_qwen3_with_thinking_off():
    pair = InstructionPair(
        document_id="contract-a",
        task="clause_span",
        messages=(("user", "Quote the Governing Law clause or answer none."),),
        completion="Nevada",
    )

    from rosai.train import format_pair

    assert format_pair(pair, thinking=False) == _THINKING_OFF


def test_thinking_on_is_refused():
    pair = InstructionPair(
        document_id="contract-a",
        task="clause_span",
        messages=(("user", "Quote the Governing Law clause or answer none."),),
        completion="Nevada",
    )

    from rosai.train import TrainingError, format_pair

    with pytest.raises(TrainingError):
        format_pair(pair, thinking=True)
