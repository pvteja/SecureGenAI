"""Tokenizer loading.

Real runs use the T5 tokenizer (``t5-small``). For unit tests and offline smoke
runs, ``tiny_model: true`` builds a small character-level tokenizer locally so
nothing has to be downloaded.
"""
from __future__ import annotations

import string

from .config import Config


def char_tokenizer():
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    specials = ["<pad>", "</s>", "<unk>"]
    chars = sorted(set(string.printable))
    vocab = {tok: i for i, tok in enumerate(specials + chars)}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tk.decoder = decoders.Fuse()
    from tokenizers.processors import TemplateProcessing

    tk.post_processor = TemplateProcessing(single="$A </s>", special_tokens=[("</s>", 1)])
    return PreTrainedTokenizerFast(
        tokenizer_object=tk, pad_token="<pad>", eos_token="</s>", unk_token="<unk>"
    )


def build_tokenizer(cfg: Config):
    if cfg.tiny_model:
        return char_tokenizer()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(cfg.model_name)
