_base_ = './textfusion_interest_common.py'

model = dict(multimodal_fusion=dict(fusion_mode='dynamic_gate_identity'))
custom_hooks = [
    dict(type='TextFusionInterestScheduleHook', lambda_interest_max=1.0)
]
work_dir = 'workdir/m3fd/multimodal/textfusion_interest/dynamic_gate_identity_lambda10_train50'
