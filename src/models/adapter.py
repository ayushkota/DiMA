import torch
import torch.nn as nn
from typing import List, Optional

from src.models.blocks import BertBlock


class ConditionAdapter(nn.Module):
    """
    Reads the context (ESM2 latents of the antibody with the CDR-H3 replaced by <mask> tokens) and produces
    one vector per row for each of the layers of the frozen score estimator.

    context:       [B, L, embedding_size]  normalized ESM2 latents, CDR-H3 positions encoded from <mask> tokens
    attention_mask:[B, L]                  1 for real tokens, 0 for padding
    masked_region: [B, L]                  1 where the CDR-H3 is (the part to generate), 0 elsewhere
    returns:       list of num_hidden_layers tensors [B, L, hidden_size], to be added to the score estimator layers
    """

    def __init__(self, config, num_blocks: int = 3):
        super().__init__()
        self.input_projector = nn.Linear(config.embedding_size, config.hidden_size)
        self.region_embedding = nn.Embedding(2, config.hidden_size)  # 0 = known, 1 = region to generate
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.blocks = nn.ModuleList([BertBlock(config) for _ in range(num_blocks)])
        self.layer_outputs = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.hidden_size) for _ in range(config.num_hidden_layers)]
        )
        # zero init: at the start the adapter adds nothing, so the model behaves exactly like the pretrained one
        for layer in self.layer_outputs:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, context, attention_mask, masked_region) -> List[torch.Tensor]:
        positions = torch.arange(context.size(1), device=context.device)[None]
        h = self.input_projector(context) + self.position_embeddings(positions) + self.region_embedding(masked_region.long())
        mask = attention_mask.to(h.dtype)[:, None, None, :]
        extended_mask = (1.0 - mask) * torch.finfo(h.dtype).min
        for block in self.blocks:
            h = block(hidden_states=h, attention_mask=extended_mask)
        return [layer(h) for layer in self.layer_outputs]


class ConditionedScoreEstimator(nn.Module):
    """
    Wraps the frozen score estimator and the adapter so the rest of the code (solver, calc_score) can keep
    calling it as before. Call set_context() once per batch, then use it like the normal score estimator.
    """

    def __init__(self, score_estimator: nn.Module, adapter: ConditionAdapter):
        super().__init__()
        self.score_estimator = score_estimator
        self.adapter = adapter
        self.cond: Optional[List[torch.Tensor]] = None

    def set_context(self, context, attention_mask, masked_region):
        self.cond = self.adapter(context, attention_mask, masked_region)

    def forward(self, x_t, time_t=None, attention_mask=None, x_0_self_cond=None):
        return self.score_estimator(
            x_t=x_t,
            time_t=time_t,
            attention_mask=attention_mask,
            x_0_self_cond=x_0_self_cond,
            cond=self.cond,
        )
