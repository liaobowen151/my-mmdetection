import torch
from torch import nn


class SubjectHeadClassifier(nn.Module):
    """Scores each candidate BERT token as the referent head."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hidden: torch.Tensor,
                candidate_mask: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(hidden).squeeze(-1)
        return logits.masked_fill(~candidate_mask.bool(), -1e4)
