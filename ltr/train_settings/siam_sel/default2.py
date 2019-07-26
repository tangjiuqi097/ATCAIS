import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms
from bisect import bisect_right
from ltr.dataset import Lasot, TrackingNet, MSCOCOSeq
from ltr.data import processing, sampler, LTRLoader

from ltr import actors
from ltr.actors.siam_sel2 import SiamSelActor
from ltr.trainers import LTRTrainer

from pytracking.mmdetection.mmdet.datasets.transforms import ImageTransform
from ltr.models.siam_sel.siam_sel2 import SiamSelNet

def run(settings):
    # Most common settings are assigned in the settings struct
    settings.description = 'Siam selection for detection with default settings.'
    settings.print_interval = 1                                 # How often to print loss and other info
    settings.batch_size = 1                                    # Batch size
    assert settings.batch_size==1,"only implement for batch_size 1"
    settings.num_workers = 0                                   # Number of workers for image loading
    settings.normalize_mean = [0.485, 0.456, 0.406]             # Normalize mean (default pytorch ImageNet values)
    settings.normalize_std = [0.229, 0.224, 0.225]              # Normalize std (default pytorch ImageNet values)
    settings.search_area_factor = 5.0                           # Image patch size relative to target size
    settings.feature_sz = 18                                    # Size of feature map
    settings.output_sz = settings.feature_sz * 16               # Size of input image patches

    # Settings for the image sample and proposal generation
    settings.center_jitter_factor = {'train': 0, 'test': 4.5}
    settings.scale_jitter_factor = {'train': 0, 'test': 0.5}
    settings.proposal_params = {'min_iou': 0.1, 'boxes_per_frame': 16, 'sigma_factor': [0.01, 0.05, 0.1, 0.2, 0.3]}

    # Train datasets
    lasot_train = Lasot(split='train')
    trackingnet_train = TrackingNet(set_ids=list(range(11)))
    coco_train = MSCOCOSeq()

    # Validation datasets
    trackingnet_val = TrackingNet(set_ids=list(range(11,12)))

    # # The joint augmentation transform, that is applied to the pairs jointly
    # transform_joint = dltransforms.ToGrayscale(probability=0.05)
    #
    # # The augmentation transform applied to the training set (individually to each image in the pair)
    # transform_train = torchvision.transforms.Compose([dltransforms.ToTensorAndJitter(0.2),
    #                                                   torchvision.transforms.Normalize(mean=settings.normalize_mean, std=settings.normalize_std)])
    #
    # # The augmentation transform applied to the validation set (individually to each image in the pair)
    # transform_val = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
    #                                                 torchvision.transforms.Normalize(mean=settings.normalize_mean, std=settings.normalize_std)])

    # Data processing to do on the training pairs
    # data_processing_train = processing.ATOMProcessing(search_area_factor=settings.search_area_factor,
    #                                                   output_sz=settings.output_sz,
    #                                                   center_jitter_factor=settings.center_jitter_factor,
    #                                                   scale_jitter_factor=settings.scale_jitter_factor,
    #                                                   mode='sequence',
    #                                                   proposal_params=settings.proposal_params,
    #                                                   transform=transform_train,
    #                                                   joint_transform=transform_joint)
    #
    # # Data processing to do on the validation pairs
    # data_processing_val = processing.ATOMProcessing(search_area_factor=settings.search_area_factor,
    #                                                 output_sz=settings.output_sz,
    #                                                 center_jitter_factor=settings.center_jitter_factor,
    #                                                 scale_jitter_factor=settings.scale_jitter_factor,
    #                                                 mode='sequence',
    #                                                 proposal_params=settings.proposal_params,
    #                                                 transform=transform_val,
    #                                                 joint_transform=transform_joint)

    img_transform = ImageTransform(
        size_divisor=32, mean=[123.675, 116.28, 103.53],std=[58.395, 57.12, 57.375],to_rgb=True)
    data_processing=processing.SiamSelProcessing(transform=img_transform)

    # The sampler for training
    dataset_train = sampler.ATOMSampler([lasot_train, trackingnet_train, coco_train], [1,1,1],
                                        samples_per_epoch=1000*settings.batch_size, max_gap=2000,
                                        processing=data_processing)

    # The loader for training
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=1)

    # The sampler for validation
    dataset_val = sampler.ATOMSampler([trackingnet_val], [1], samples_per_epoch=500*settings.batch_size, max_gap=2000,
                                      processing=data_processing)

    # The loader for validation
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size, num_workers=settings.num_workers,
                           shuffle=False, drop_last=True, epoch_interval=5, stack_dim=1)

    # Create network
    # net = atom_models.atom_resnet18(backbone_pretrained=True)
    net=SiamSelNet()

    # Set objective
    objective = nn.BCEWithLogitsLoss()

    # Create actor, which wraps network and objective
    actor = SiamSelActor(net=net, objective=objective)


    parameters=[]
    for key,value in actor.net.named_parameters():
        if key.find("fc_cls")>=0 or key.find("selector")>=0:
            parameters.append(value)
        else:
            value.requires_grad=False
    # Optimizer
    optimizer = optim.Adam(parameters, lr=1e-4,weight_decay=0.0001)

    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.2)
    # lr_scheduler = WarmupMultiStepLR(optimizer,[50*1000,80*1000])


    # Create trainer
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    # Run training (set fail_safe=False if you are debugging)
    trainer.train(100, load_latest=True, fail_safe=False)


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
#     def __init__(
#         self,
#         optimizer,
#         milestones,
#         gamma=0.1,
#         warmup_factor=1.0 / 3,
#         warmup_iters=500,
#         warmup_method="linear",
#         last_epoch=-1,
#     ):
#         if not list(milestones) == sorted(milestones):
#             raise ValueError(
#                 "Milestones should be a list of" " increasing integers. Got {}",
#                 milestones,
#             )
#
#         if warmup_method not in ("constant", "linear"):
#             raise ValueError(
#                 "Only 'constant' or 'linear' warmup_method accepted"
#                 "got {}".format(warmup_method)
#             )
#         self.milestones = milestones
#         self.gamma = gamma
#         self.warmup_factor = warmup_factor
#         self.warmup_iters = warmup_iters
#         self.warmup_method = warmup_method
#         super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)
#
#     def get_lr(self):
#         warmup_factor = 1
#         if self.last_epoch < self.warmup_iters:
#             if self.warmup_method == "constant":
#                 warmup_factor = self.warmup_factor
#             elif self.warmup_method == "linear":
#                 alpha = float(self.last_epoch) / self.warmup_iters
#                 warmup_factor = self.warmup_factor * (1 - alpha) + alpha
#         return [
#             base_lr
#             * warmup_factor
#             * self.gamma ** bisect_right(self.milestones, self.last_epoch)
#             for base_lr in self.base_lrs
#         ]