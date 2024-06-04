from bpe_tokenizer import BPETokenizer
import json
import os


def initialize_pretrained_tokenizer():
    with open(
        os.path.join("tests", "pretrained_small_tokenizer.json"), "r"
    ) as pretrained_small_tokenizer_file:
        tokenizer = BPETokenizer(
            vocabulary=json.load(pretrained_small_tokenizer_file).get("vocabulary", {})
        )

        return tokenizer


def text_to_basic_tokens(text):
    return list(map(int, text.encode(encoding="utf-8")))


def basic_tokens_to_text(tokens):
    return bytes(tokens).decode(encoding="utf-8")
