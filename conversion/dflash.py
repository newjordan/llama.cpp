from __future__ import annotations

import json

from typing import Iterable, TYPE_CHECKING

from .base import ModelBase, TextModel, gguf, logger

if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("DFlashDraftModel")
class DFlashModel(TextModel):
    # z-lab DFlash block-diffusion speculative-decode draft head: a standalone Qwen3-style
    # transformer that cross-attends to concatenated target hidden states (target_layer_ids)
    # and borrows the target's token embeddings + lm_head. Ships no tokenizer / embed / lm_head,
    # so conversion needs --target-model-dir for the tokenizer and the target hidden size.
    model_arch = gguf.MODEL_ARCH.DFLASH

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        dfc = self.hparams.get("dflash_config") or {}
        # schema drift: top-level (Qwen3-8B head) or under dflash_config (Qwen3.5-4B head)
        self.block_size = self.hparams.get("block_size", dfc.get("block_size"))
        self.mask_token_id = self.hparams.get("mask_token_id", dfc.get("mask_token_id"))
        self.target_layer_ids = self.hparams.get("target_layer_ids", dfc.get("target_layer_ids"))
        if self.block_size is None or self.mask_token_id is None or not self.target_layer_ids:
            raise ValueError("DFlash: config missing block_size / mask_token_id / target_layer_ids")

        if self.target_model_dir is None:
            raise ValueError(
                "DFlash draft conversion requires --target-model-dir "
                "(the head ships no tokenizer and needs the target hidden size)."
            )
        with open(self.target_model_dir / "config.json", "r", encoding="utf-8") as f:
            tcfg = json.load(f)
        if "text_config" in tcfg:
            tcfg = {**tcfg, **tcfg["text_config"]}
        self.target_hidden_size = int(tcfg["hidden_size"])
        logger.info(
            "DFlash: block_size=%s mask_token_id=%s target_layers=%s target_hidden_size=%s",
            self.block_size, self.mask_token_id, self.target_layer_ids, self.target_hidden_size,
        )

    def set_vocab(self):
        # the head ships no tokenizer; borrow the target's (shared vocab)
        original_dir_model = self.dir_model
        self.dir_model = self.target_model_dir
        try:
            self._set_vocab_gpt2()
        finally:
            self.dir_model = original_dir_model

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        arch = self.gguf_writer.arch

        head_dim = hparams.get("head_dim") or hparams["hidden_size"] // hparams["num_attention_heads"]
        self.gguf_writer.add_rope_dimension_count(head_dim)
        self.gguf_writer.add_key_length(head_dim)
        self.gguf_writer.add_value_length(head_dim)

        # spec <-> target coupling (reuse the EAGLE3 KV keys the loader already understands)
        self.gguf_writer.add_array(f"{arch}.target_layers", [int(x) for x in self.target_layer_ids])
        self.gguf_writer.add_uint32(f"{arch}.target_hidden_size", self.target_hidden_size)
        # dflash-specific
        self.gguf_writer.add_uint32(f"{arch}.block_size", int(self.block_size))
        self.gguf_writer.add_uint32(gguf.Keys.Tokenizer.MASK_ID, int(self.mask_token_id))

    def index_tensors(self, remote_hf_model_id: str | None = None):
        tensors = super().index_tensors(remote_hf_model_id)
        # head stores layer/final-norm tensors without the "model." prefix; add it so the standard
        # qwen3 tensor map matches and bid (".layers.N.") parses. fc / hidden_norm stay top-level.
        renamed: dict = {}
        for name, gen in tensors.items():
            if name.startswith("layers.") or name == "norm.weight":
                renamed["model." + name] = gen
            else:
                renamed[name] = gen
        return renamed

    def modify_tensors(self, data_torch: "Tensor", name: str, bid: int | None) -> Iterable[tuple[str, "Tensor"]]:
        # dflash top-level tensors stored under raw GGUF names (match C++ tn(LLM_TENSOR_FC/...))
        if name == "fc.weight":
            yield ("fc.weight", data_torch)
            return
        if name == "hidden_norm.weight":
            yield ("hidden_norm.weight", data_torch)
            return
        yield from super().modify_tensors(data_torch, name, bid)
