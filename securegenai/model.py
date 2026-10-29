"""SecureGenAI model (Section IV of the paper, Fig. 1).

Five stages:

1. Vulnerability-aware input representation  - T5 token embeddings + position information
2. Secure semantic embedding layer           - T5 encoder (multi-head self-attention)
3. Transformer secure code generator         - T5 decoder, P(Y_s | X_v)
4. Vulnerability refinement module           - S_vul = sigmoid(W_h * H + b), trained with BCE
5. Secure code reconstruction                - beam search, with a feedback loop that sends
                                               output still scored as vulnerable back in

Training objective:  L_total = λ1·L_gen + λ2·L_sec (+ λ_cwe·L_cwe)

``L_cwe`` is an auxiliary CWE-class head that produces the per-class
confusion matrix reported in the paper. It is an implementation choice; set
``lambda_cwe: 0`` or ``use_cwe_head: false`` to train with the two-term
objective only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config, T5ForConditionalGeneration

from .config import Config
from .data import UNKNOWN_CWE


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool encoder states over real (non-padding) tokens -> H."""
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)


@dataclass
class SecureGenAIOutput:
    loss: torch.Tensor
    gen_loss: torch.Tensor
    sec_loss: torch.Tensor
    cwe_loss: torch.Tensor
    vuln_logits: Optional[torch.Tensor] = None   # [B] on vulnerable inputs
    fixed_logits: Optional[torch.Tensor] = None  # [B] on fixed inputs
    cwe_logits: Optional[torch.Tensor] = None    # [B, C]


class SecureGenAI(nn.Module):
    def __init__(self, backbone: T5ForConditionalGeneration, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        d_model = backbone.config.d_model
        self.dropout = nn.Dropout(cfg.dropout)
        self.security_head = nn.Linear(d_model, 1) if cfg.use_security_head else None
        self.cwe_head = nn.Linear(d_model, len(cfg.cwe_classes)) if cfg.use_cwe_head else None

    # ------------------------------------------------------------ helpers
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone.get_encoder()(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state

    def _pooled(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.dropout(masked_mean(hidden, mask))

    # ------------------------------------------------------------ training
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        fixed_input_ids: Optional[torch.Tensor] = None,
        fixed_attention_mask: Optional[torch.Tensor] = None,
        cwe_label: Optional[torch.Tensor] = None,
    ) -> SecureGenAIOutput:
        cfg = self.cfg
        enc_hidden = self.encode(input_ids, attention_mask)
        gen = self.backbone(
            encoder_outputs=(enc_hidden,),
            attention_mask=attention_mask,
            labels=labels,
        )
        gen_loss = gen.loss
        zero = gen_loss.new_zeros(())

        vuln_logits = fixed_logits = cwe_logits = None
        sec_loss = zero
        if self.security_head is not None and fixed_input_ids is not None:
            pooled_v = self._pooled(enc_hidden, attention_mask)
            fixed_hidden = self.encode(fixed_input_ids, fixed_attention_mask)
            pooled_f = self._pooled(fixed_hidden, fixed_attention_mask)
            vuln_logits = self.security_head(pooled_v).squeeze(-1)
            fixed_logits = self.security_head(pooled_f).squeeze(-1)
            logits = torch.cat([vuln_logits, fixed_logits])
            targets = torch.cat([torch.ones_like(vuln_logits), torch.zeros_like(fixed_logits)])
            sec_loss = F.binary_cross_entropy_with_logits(logits, targets)

        cwe_loss = zero
        if self.cwe_head is not None:
            cwe_logits = self.cwe_head(self._pooled(enc_hidden, attention_mask))
            if cwe_label is not None and (cwe_label != UNKNOWN_CWE).any():
                cwe_loss = F.cross_entropy(cwe_logits, cwe_label, ignore_index=UNKNOWN_CWE)

        loss = cfg.lambda_gen * gen_loss + cfg.lambda_sec * sec_loss + cfg.lambda_cwe * cwe_loss
        return SecureGenAIOutput(loss, gen_loss, sec_loss, cwe_loss, vuln_logits, fixed_logits, cwe_logits)

    # ----------------------------------------------------------- inference
    @torch.no_grad()
    def vulnerability_score(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """S_vul in [0, 1]: probability that the code is still vulnerable."""
        if self.security_head is None:
            raise RuntimeError("Model was built without the security head (use_security_head=false).")
        hidden = self.encode(input_ids, attention_mask)
        return torch.sigmoid(self.security_head(masked_mean(hidden, attention_mask)).squeeze(-1))

    @torch.no_grad()
    def predict_cwe(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.cwe_head is None:
            raise RuntimeError("Model was built without the CWE head (use_cwe_head=false).")
        hidden = self.encode(input_ids, attention_mask)
        return self.cwe_head(masked_mean(hidden, attention_mask))

    @torch.no_grad()
    def generate_secure(
        self,
        tokenizer,
        code: List[str],
        beam_width: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        refine_iterations: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, object]]:
        """Beam-search reconstruction with the Fig. 1 feedback loop.

        Each output is re-scored by the refinement module; anything still above
        ``threshold`` is fed back in as the new input, up to ``refine_iterations``
        extra passes.
        """
        cfg = self.cfg
        beam_width = beam_width or cfg.beam_width
        max_new_tokens = max_new_tokens or cfg.max_new_tokens
        refine_iterations = cfg.refine_iterations if refine_iterations is None else refine_iterations
        threshold = cfg.vuln_threshold if threshold is None else threshold
        device = next(self.parameters()).device

        current = list(code)
        results = [{"input": c, "output": c, "vuln_score": None, "passes": 0} for c in code]
        active = list(range(len(code)))

        for _ in range(refine_iterations + 1):
            if not active:
                break
            enc = tokenizer(
                [current[i] for i in active], truncation=True, max_length=cfg.max_length,
                padding=True, return_tensors="pt",
            ).to(device)
            out_ids = self.backbone.generate(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                num_beams=beam_width, max_new_tokens=max_new_tokens, early_stopping=True,
            )
            decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)

            scores: List[Optional[float]] = [None] * len(active)
            if self.security_head is not None:
                re_enc = tokenizer(
                    decoded, truncation=True, max_length=cfg.max_length,
                    padding=True, return_tensors="pt",
                ).to(device)
                scores = self.vulnerability_score(re_enc["input_ids"], re_enc["attention_mask"]).tolist()

            still_active = []
            for idx, text, score in zip(active, decoded, scores):
                results[idx]["output"] = text
                results[idx]["vuln_score"] = score
                results[idx]["passes"] += 1
                current[idx] = text
                if score is not None and score > threshold:
                    still_active.append(idx)
            active = still_active
        return results


# ------------------------------------------------------------- factories
def build_backbone(cfg: Config, vocab_size: Optional[int] = None) -> T5ForConditionalGeneration:
    if cfg.tiny_model:
        # Small random T5 for unit tests and offline smoke runs. Not for real training.
        t5cfg = T5Config(
            vocab_size=vocab_size or 512, d_model=64, d_kv=16, d_ff=128,
            num_layers=2, num_decoder_layers=2, num_heads=4,
            dropout_rate=cfg.dropout, pad_token_id=0, eos_token_id=1, decoder_start_token_id=0,
        )
        return T5ForConditionalGeneration(t5cfg)
    model = T5ForConditionalGeneration.from_pretrained(cfg.model_name, dropout_rate=cfg.dropout)
    if vocab_size is not None and vocab_size > model.config.vocab_size:
        model.resize_token_embeddings(vocab_size)
    return model


def build_model(cfg: Config, tokenizer) -> SecureGenAI:
    return SecureGenAI(build_backbone(cfg, vocab_size=len(tokenizer)), cfg)


def save_model(model: SecureGenAI, tokenizer, path: str) -> None:
    import os

    os.makedirs(path, exist_ok=True)
    model.backbone.save_pretrained(path)
    tokenizer.save_pretrained(path)
    heads = {}
    if model.security_head is not None:
        heads["security_head"] = model.security_head.state_dict()
    if model.cwe_head is not None:
        heads["cwe_head"] = model.cwe_head.state_dict()
    torch.save(heads, os.path.join(path, "securegenai_heads.pt"))
    model.cfg.save(os.path.join(path, "securegenai_config.yaml"))


def load_model(path: str, device: str = "cpu"):
    import os

    from transformers import AutoTokenizer

    cfg = Config.from_yaml(os.path.join(path, "securegenai_config.yaml"))
    tokenizer = AutoTokenizer.from_pretrained(path)
    backbone = T5ForConditionalGeneration.from_pretrained(path)
    model = SecureGenAI(backbone, cfg)
    heads = torch.load(os.path.join(path, "securegenai_heads.pt"), map_location="cpu")
    if model.security_head is not None:
        model.security_head.load_state_dict(heads["security_head"])
    if model.cwe_head is not None:
        model.cwe_head.load_state_dict(heads["cwe_head"])
    return model.to(device).eval(), tokenizer, cfg
