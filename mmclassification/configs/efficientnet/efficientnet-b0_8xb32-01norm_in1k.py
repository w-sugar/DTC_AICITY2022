_base_ = [
    '../_base_/models/efficientnet_b0.py',#模型当中修改类别数
    # '../_base_/datasets/mydataset2.py',
    '../_base_/datasets/imagenet_bs64_autoaug.py',
    '../_base_/schedules/imagenet_bs256.py',#修改lr,epoch等
    '../_base_/default_runtime_b0.py',
]

# dataset settings
#如果不注释掉下面的内容，则不会用到   '../_base_/datasets/mydataset2.py',
#因为下面新的东西会把上面的_base_覆盖掉
# dataset_type = 'ImageNet'
# img_norm_cfg = dict(
#     mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='RandomResizedCrop',
#         size=224,
#         efficientnet_style=True,
#         interpolation='bicubic'),
#     dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='ToTensor', keys=['gt_label']),
#     dict(type='Collect', keys=['img', 'gt_label'])
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='CenterCrop',
#         crop_size=224,
#         efficientnet_style=True,
#         interpolation='bicubic'),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='Collect', keys=['img'])
# ]
# data = dict(
#     train=dict(pipeline=train_pipeline),
#     val=dict(pipeline=test_pipeline),
#     test=dict(pipeline=test_pipeline))
