_base_ = './textfusion_interest_common.py'

model = dict(multimodal_fusion=dict(fusion_mode='direct_text_projected'))
custom_hooks = [
    dict(type='TextFusionInterestScheduleHook', lambda_interest_max=1.0)
]
work_dir = 'workdir/m3fd/multimodal/textfusion_interest/direct_text_projected_lambda10_train50'
