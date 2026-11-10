"""FLIR IR-LoRA32 with bidirectional deformable tri-prior fusion."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_flir_shared_backbone_ir_lora32_ir2vis_deformable_train30.py'

# Replace the one-way fusion with bidirectional deformable enhancement followed
# by illumination-local-global TPF. The Grounding DINO encoder remains intact
# and performs the subsequent vision-language interaction.
custom_imports = dict(
    imports=[
        'projects.myself_ir_lora_bidirectional_deformable_tpf_mmgroundingdino',
    ],
    allow_failed_imports=False)

model = dict(
    type='IRLoRABidirectionalDeformableTPFGroundingDINO',
    multimodal_fusion=dict(
        type='BidirectionalDeformableTPFFusion',
        embed_dims=256,
        num_heads=8,
        dropout=0.0,
        num_feature_levels=4,
        num_points=4,
        rgb_mean=(123.675, 116.28, 103.53),
        rgb_std=(58.395, 57.12, 57.375),
        illumination_pool_size=8))
