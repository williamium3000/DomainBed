# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .impl.original import *
from .impl.clip import CLIP, CLIP_LP, CLIP_Finetune, \
    CLIP_FinetuneWithTextFreeze, LanguageDrivenDG, LanguageDrivenDGV2, LanguageDrivenDGV2_EMA, \
        LanguageDrivenDGV3, LanguageDrivenDGV3_EMA, CLIP_FinetuneWithTextFreeze_EMA, CLIP_FinetuneWithTextFreeze_BetaEMA, \
            CLIPood_BetaEMA, CLIPood, CLIP_FinetuneWithTextFreezeWithDomainV2, CLIP_FinetuneWithTextFreezeWithDomain, \
                CLIP_FinetuneWithTextFreezeWithDomainV2_BetaEMA, CLIP_FinetuneWithTextFreezeWithDomainV2_EMA, \
                CLIP_FinetuneWithTextFreezeWithDomain_BetaEMA, CLIP_FinetuneWithTextFreezeWithDomain_EMA
from .impl.w2d import W2D, W2D_v2, W2D_EMA, W2D_v2_EMA
from .impl.clip_kd import ERM_CLIP_Logits, W2D_v2_CLIP_Logits, W2D_v2_CLIP_Logits_EMA, \
    ERM_CLIP_Logits_EMA, ERM_SMA_HardExampleMining, ERM_SMA_CLIPDistill
from .impl.sma import ERM_SMA

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
    'W2D_EMA',
    'W2D_v2_EMA',
    'ERM_CLIP_Logits',
    'W2D_v2_CLIP_Logits',
    'ERM_SMA',
    'W2D_v2_CLIP_Logits_EMA',
    'ERM_CLIP_Logits_EMA',
    'ERM_SMA_HardExampleMining',
    'CLIP_FinetuneWithTextFreezeWithDomain_BetaEMA',
    'CLIP_FinetuneWithTextFreezeWithDomain_EMA',
    'CLIP_FinetuneWithTextFreezeWithDomainV2_BetaEMA',
    'CLIP_FinetuneWithTextFreezeWithDomainV2_EMA',
    'ERM_SMA_CLIPDistill',
    'CLIP_FinetuneWithTextFreeze',
    'CLIP_FinetuneWithTextFreeze_EMA',
    'CLIP_FinetuneWithTextFreeze_BetaEMA',
    'CLIP_FinetuneWithTextFreezeWithDomainV2',
    'CLIP_FinetuneWithTextFreezeWithDomain'
    'LanguageDrivenDG',
    'LanguageDrivenDGV2',
    'LanguageDrivenDGV2_EMA',
    'LanguageDrivenDGV3',
    'LanguageDrivenDGV3_EMA',
    'CLIPood_BetaEMA',
    'CLIPood'
]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

