_base_ = [
    '../_base_/models/convnext/convnext-small.py',
    '../_base_/datasets/imagenet_bs64_autoaug.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

# data = dict(samples_per_gpu=128)
#
# optimizer = dict(lr=4e-3)

custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]
