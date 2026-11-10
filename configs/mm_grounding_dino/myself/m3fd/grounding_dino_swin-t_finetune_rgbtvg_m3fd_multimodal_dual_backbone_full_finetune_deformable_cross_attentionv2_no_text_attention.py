"""M3FD deformable VIS/IR fusion without direct text attention."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_m3fd_multimodal_dual_backbone_full_finetune_deformable_cross_attentionv2.py'

custom_imports = dict(
    imports=[
        'projects.myself_dual_backbone_deformable_no_text_attention_mmgroundingdino'
    ],
    allow_failed_imports=False)

model = dict(
    type='DualBackboneDeformableNoTextAttentionGroundingDINO',
    multimodal_fusion=dict(
        _delete_=True,
        type='DeformableNoTextAttentionFusionBlock',
        embed_dims=256,
        num_heads=8,
        dropout=0.0,
        num_feature_levels=4,
        num_points=4))

work_dir = ('workdir/m3fd/multimodal/'
            'dual_backbone_deformable_cross_attentionv2_no_text_attention/'
            'groundingdino_rgbtvg_m3fd_multimodal_dual_backbone_full_'
            'finetune_deformable_cross_attentionv2_no_text_attention_train')
