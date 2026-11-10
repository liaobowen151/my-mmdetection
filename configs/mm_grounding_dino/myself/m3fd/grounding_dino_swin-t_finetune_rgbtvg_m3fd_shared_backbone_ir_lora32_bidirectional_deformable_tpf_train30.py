_base_ = './grounding_dino_swin-t_finetune_rgbtvg_m3fd_shared_backbone_ir_lora32_bidirectional_deformable_sum_train30.py'

# Keep the bidirectional-deformable-sum experiment unchanged and replace only
# its final VIS' + IR' operation with illumination-local-global tri-prior
# fusion. Text remains integrated by the original Grounding DINO encoder.
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
