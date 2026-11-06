"""Serving and work accounting for the separately declared multi-round study.

Logical attention pairs are an operation-count proxy, not measured GPU FLOPs.
Stock question/control prompts may use one batched HF call; sparse prompts use
the existing single-token adapter. Both charge every evaluated model token.
"""
import torch
from transformers import DynamicCache


class BatchedStockSession:
    def __init__(self, model, parent):
        self.model = model
        self.pending_token = None
        self.position = parent.count
        self.past = DynamicCache()
        device = next(model.parameters()).device
        for layer in range(parent.layers):
            batch = parent.read_range(layer, 0, parent.count)
            self.past.update(batch.k.to(device), batch.v.to(device), layer)

    @property
    def controller(self):
        return self

    @torch.inference_mode()
    def forward_tokens(self, tokens, pos):
        if not tokens or pos != self.position:
            raise ValueError('explicit next position and nonempty stock input required')
        if pos + len(tokens) > self.model.config.max_position_embeddings:
            raise ValueError('native context budget exceeded')
        device = next(self.model.parameters()).device
        out = self.model(torch.as_tensor(tokens, device=device)[None],
                         past_key_values=self.past, use_cache=True, logits_to_keep=1)
        self.past = out.past_key_values
        self.position += len(tokens)
        return out.logits[0, -1]

    def step(self, token, pos):
        return self.forward_tokens([int(token)], pos)


class CountedSession:
    """Count real forwards and causal KV positions without evaluating answers."""
    def __init__(self, adapter, model, active_limit=None):
        self.adapter = adapter
        self.layers = model.config.num_hidden_layers
        self.heads = model.config.num_attention_heads
        self.active_limit = active_limit
        self.model_tokens = 0
        self.forward_calls = 0
        self.prompt_tokens = 0
        self.layer_kv_positions = 0
        self.max_kv_entries = 0
        self.trace_calls = 0
        self.stock = isinstance(adapter, BatchedStockSession)
        if not self.stock:
            if adapter.controller.trace is not None:
                raise ValueError('do not replace an existing attention audit')
            adapter.controller.trace = self._trace

    @property
    def controller(self):
        return self.adapter.controller

    @property
    def position(self):
        return self.adapter.position

    def _trace(self, layer, pos, q, view):
        count = len(view.ids)
        if self.active_limit is not None and count > self.active_limit:
            raise RuntimeError('declared working KV limit exceeded')
        self.layer_kv_positions += count
        self.max_kv_entries = max(self.max_kv_entries, count)
        self.trace_calls += 1

    def _forward(self, tokens):
        before = self.position
        traces_before = self.trace_calls
        logits = self.adapter.forward_tokens(tokens, before)
        if self.position != before + len(tokens):
            raise RuntimeError('forward position did not advance by its real token count')
        if self.stock:
            # n queries see old_length+1, ..., old_length+n valid keys. SDPA
            # may execute padded/masked work: this is not physical FLOP count.
            n = len(tokens)
            self.layer_kv_positions += self.layers * (n * before + n * (n + 1) // 2)
            self.max_kv_entries = max(self.max_kv_entries, self.position)
        elif self.trace_calls - traces_before != self.layers:
            raise RuntimeError('single-token sparse trace missed or duplicated layers')
        self.model_tokens += len(tokens)
        self.forward_calls += 1
        return logits

    def feed_prompt(self, tokens, guard=None):
        if not tokens:
            raise ValueError('empty prompt')
        if self.controller.pending_token is not None:
            raise ValueError('consume the preceding generated token before a new prompt')
        if self.stock:
            if guard:
                guard.check()
            logits = self._forward(tokens)
        else:
            for token in tokens:
                if guard:
                    guard.check()
                logits = self._forward([token])
        self.prompt_tokens += len(tokens)
        return logits

    def step(self, token, pos):
        if pos != self.position:
            raise ValueError('explicit next position required')
        return self._forward([int(token)])

    def consume_pending(self):
        token = self.controller.pending_token
        if token is None:
            raise ValueError('no pending generated token')
        self.controller.pending_token = None
        return self.step(token, self.position)

    def metrics(self):
        return {'model_token_evaluations': self.model_tokens,
                'transformer_forward_calls': self.forward_calls,
                'prompt_token_evaluations': self.prompt_tokens,
                'layer_summed_valid_KV_positions': self.layer_kv_positions,
                'logical_attention_head_pairs': self.layer_kv_positions * self.heads,
                'max_active_KV_entries': self.max_kv_entries,
                'count_scope': 'request forwards including all prompt replay and consumed generated tokens; excludes one-time archive construction; logical attention pairs are not measured GPU FLOPs'}
