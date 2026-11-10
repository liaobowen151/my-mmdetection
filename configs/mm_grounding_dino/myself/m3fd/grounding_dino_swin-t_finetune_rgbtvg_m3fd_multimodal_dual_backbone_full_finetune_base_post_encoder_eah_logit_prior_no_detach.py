"""M3FD post-encoder EAH logit prior with trainable token weights."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_m3fd_multimodal_dual_backbone_full_finetune_base_post_encoder_eah.py'

custom_imports = dict(
    imports=[
        'projects.myself_dual_backbone_post_encoder_eah_logit_prior_mmgroundingdino'
    ],
    allow_failed_imports=False)

model = dict(
    type='DualBackbonePostEncoderEAHLogitPriorGroundingDINO',
    eah_prior_scale=0.2,
    eah_prior_max=2.0,
    eah_prior_eps=1e-6,
    eah_prior_detach=False)

work_dir = (
    'workdir/m3fd/multimodal/'
    'dual_backbone_base_post_encoder_eah_logit_prior_no_detach/'
    'groundingdino_rgbtvg_m3fd_multimodal_dual_backbone_full_finetune_'
    'base_post_encoder_eah_logit_prior_no_detach_train')
