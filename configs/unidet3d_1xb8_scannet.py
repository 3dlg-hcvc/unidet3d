_base_ = ['mmdet3d::_base_/default_runtime.py']
custom_imports = dict(imports=['unidet3d'])

classes_scannet = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
                   'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'showercurtrain',
                   'toilet', 'sink', 'bathtub', 'otherfurniture']

# model settings
num_channels = 32
voxel_size = 0.02

model = dict(
    type='UniDet3D',
    data_preprocessor=dict(type='Det3DDataPreprocessor_'),
    in_channels=6,
    num_channels=num_channels,
    voxel_size=voxel_size,
    min_spatial_shape=128,
    query_thr=3000,
    bbox_by_mask=True,
    fast_nms=True,
    backbone=dict(
        type='SpConvUNet',
        num_planes=[num_channels * (i + 1) for i in range(5)],
        return_blocks=True
    ),
    decoder=dict(
        type='UniDet3DEncoder',
        num_layers=6,
        datasets_classes=[classes_scannet],
        in_channels=num_channels,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        datasets=['scannet'],
        ),
    criterion=dict(
        type='UniDet3DCriterion',
        bbox_loss_simple=dict(
            type='UniDet3DAxisAlignedIoULoss',
            mode='diou',
            reduction='none'),
        matcher=dict(
            type='UniMatcher',
            costs=[
                dict(type='QueryClassificationCost', weight=0.5),
                dict(type='BboxCostJointTraining',
                     weight=2.0,
                     loss_simple=dict(
                         type='UniDet3DAxisAlignedIoULoss',
                         mode='diou',
                         reduction='none')
                     )]),
        loss_weight=[0.5, 1.0],
        non_object_weight=0.1,
        topk=6,
        iter_matcher=True
    ),
    train_cfg=dict(topk=6),
    test_cfg=dict(
        low_sp_thr=0.18,
        up_sp_thr=0.81,
        topk_insts=1000,
        score_thr=0,
        iou_thr=0.5))

# scannet dataset settings

metainfo_scannet = dict(classes=classes_scannet)
data_root_scannet = 'data/scannet/'

max_class_scannet = 20
dataset_type_scannet = 'ScanNetDetDataset'
data_prefix_scannet = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask',
    sp_pts_mask='super_points'
)

train_pipeline_scannet = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadAnnotations3D_',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
        with_sp_mask_3d=True),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(type='PointSegClassMapping'),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-3.14, 3.14],
        scale_ratio_range=[0.8, 1.2],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    dict(
        type='NormalizePointsColor_',
        color_mean=[127.5, 127.5, 127.5]),
    dict(
        type='PointDetClassMappingScanNet',
        num_classes=max_class_scannet,
        stuff_classes=[0, 1]),
    dict(
        type='ElasticTransfrom',
        gran=[6, 20],
        mag=[40, 160],
        voxel_size=voxel_size,
        p=0.5),
    dict(
        type='Pack3DDetInputs_',
        keys=[
            'points', 'gt_labels_3d', 'pts_semantic_mask', 'pts_instance_mask',
            'sp_pts_mask', 'gt_sp_masks', 'elastic_coords'
        ])
]
test_pipeline_scannet = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadAnnotations3D_',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
        with_sp_mask_3d=True),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='NormalizePointsColor_',
                color_mean=[127.5, 127.5, 127.5])]),
    dict(type='Pack3DDetInputs_', keys=['points', 'sp_pts_mask'])
]

# run settings
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset_',
        datasets=[dict(
            type=dataset_type_scannet,
            ann_file='scannet_infos_train.pkl',
            data_prefix=data_prefix_scannet,
            data_root=data_root_scannet,
            metainfo=metainfo_scannet,
            pipeline=train_pipeline_scannet,
            ignore_index=max_class_scannet,
            scene_idxs=None,
            test_mode=False)]
    ))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset_',
        datasets= \
            [dict(
                type=dataset_type_scannet,
                ann_file='scannet_infos_val.pkl',
                data_prefix=data_prefix_scannet,
                data_root=data_root_scannet,
                metainfo=metainfo_scannet,
                pipeline=test_pipeline_scannet,
                ignore_index=max_class_scannet,
                test_mode=True)]
    ))

test_dataloader = val_dataloader

# load_from = 'work_dirs/tmp/oneformer3d_1xb4_scannet.pth'
load_from = None

test_evaluator = dict(type='IndoorMetric_',
                      datasets=['scannet'],
                      datasets_classes=[classes_scannet])

val_evaluator = test_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001 * 2, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2))

param_scheduler = dict(type='PolyLR', begin=0, end=1024, power=0.9)

custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=16))

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=1024,
    dynamic_intervals=[(1, 16), (1024 - 16, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
