"""FLIR shared-backbone IR-LoRA32 with bidirectional deformable sum."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_flir_shared_backbone_ir_lora32_neck_add_train30.py'

custom_imports = dict(
    imports=[
        'projects.myself_ir_lora_bidirectional_deformable_sum_mmgroundingdino'
    ],
    allow_failed_imports=False)

model = dict(
    type='IRLoRABidirectionalDeformableSumGroundingDINO',
    multimodal_fusion=dict(
        type='BidirectionalDeformableSumFusion',
        embed_dims=256,
        num_heads=8,
        dropout=0.0,
        num_feature_levels=4,
        num_points=4))

