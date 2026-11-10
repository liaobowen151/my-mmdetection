from mmdet.models.detectors.grounding_dino import GroundingDINO
from mmdet.registry import MODELS


@MODELS.register_module()
class FrozenBackboneGroundingDINO(GroundingDINO):

    def __init__(
        self,
        *args,
        freeze_visual_backbone=True,
        freeze_language_model=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.freeze_visual_backbone = freeze_visual_backbone
        self.freeze_language_model = freeze_language_model

        # Freeze before building the optimizer and DDP wrapper.
        if self.freeze_visual_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        if self.freeze_language_model:
            self.language_model.requires_grad_(False)
            self.language_model.eval()

    def train(self, mode=True):
        super().train(mode)

        # Keep Dropout/DropPath disabled in the frozen modules.
        if self.freeze_visual_backbone:
            self.backbone.eval()

        if self.freeze_language_model:
            self.language_model.eval()

        return self
