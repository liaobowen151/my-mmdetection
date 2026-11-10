"""FLIR bidirectional deformable dynamic gate, lambda=0.5."""

_base_ = './fusion_only_dynamic_gate_lambda05.py'

custom_imports = dict(
    imports=[
        'projects.myself_ir_lora_bidirectional_deformable_textfusion_interest_mmgroundingdino'
    ],
    allow_failed_imports=False)

model = dict(
    type='IRLoRABidirectionalDeformableTextFusionInterestGroundingDINO',
    multimodal_fusion=dict(
        _delete_=True,
        type='BidirectionalDeformableTextFusionInterestFusion',
        embed_dims=256,
        num_heads=8,
        dropout=0.0,
        num_feature_levels=4,
        num_points=4,
        gate_hidden_dims=256,
        gate_init_bias=2.0))

work_dir = (
    'workdir/flir/multimodal/textfusion_interest50/'
    'bidirectional_deformable_dynamic_gate_lambda05/'
    'bidirectional_deformable_dynamic_gate_lambda05_train50')
