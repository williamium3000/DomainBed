# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .impl.original import *
from .impl.clip import CLIP, CLIP_LP, CLIP_Finetune
from .impl.w2d import W2D, W2D_v2
from .impl.clip_kd import ERM_CLIP_Logits, W2D_v2_CLIP_Logits

ALGORITHMS = [
    'ERM',
    'Fish',
    'IRM',
    'GroupDRO',
    'Mixup',
    'MLDG',
    'CORAL',
    'MMD',
    'DANN',
    'CDANN',
    'MTL',
    'SagNet',
    'ARM',
    'VREx',
    'RSC',
    'SD',
    'ANDMask',
    'SANDMask',
    'IGA',
    'SelfReg',
    "Fishr",
    'TRM',
    'IB_ERM',
    'IB_IRM',
    'CAD',
    'CondCAD',
    'Transfer',
    'CausIRL_CORAL',
    'CausIRL_MMD',
    'CLIP',
    'CLIP_LP',
    'CLIP_Finetune',
    'W2D',
    'W2D_v2',
    'ERM_CLIP_Logits',
    'W2D_v2_CLIP_Logits'
]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

