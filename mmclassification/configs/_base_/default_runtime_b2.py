# checkpoint saving
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = './models/efficientnet-b2_3rdparty_8xb32_in1k_20220119-ea374a30.pth'
resume_from = None
workflow = [('train', 1)]
