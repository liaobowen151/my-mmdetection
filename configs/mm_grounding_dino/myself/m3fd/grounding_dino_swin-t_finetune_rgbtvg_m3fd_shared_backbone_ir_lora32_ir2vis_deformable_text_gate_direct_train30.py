"""M3FD IR-to-VIS deformable enhancement with direct text gating."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_m3fd_shared_backbone_ir_lora32_neck_add_train30.py'

custom_imports = dict(
    imports=[
        'projects.myself_ir_lora_ir2vis_deformable_text_gate_mmgroundingdino'
    ],
    allow_failed_imports=False)

model = dict(
    type='IRLoRAIRToVISDeformableGroundingDINO',
    multimodal_fusion=dict(
        type='IRToVISDeformableDirectTextGateFusion',
        embed_dims=256,
        num_heads=8,
        dropout=0.0,
        num_feature_levels=4,
        num_points=4,
        text_temperature=1.0,
        gate_temperature_vis=0.25,
        gate_temperature_ir=0.25,
        gate_temperature_advantage=0.25,
        vis_threshold=0.0,
        ir_threshold=0.0,
        advantage_margin=0.0,
        standardize_evidence=False,
        exclude_boundary_tokens=True,
        diagnostics_interval=100))
