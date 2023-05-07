import torch
from torch.nn import functional as F
from torch import nn
import numpy as np
from torch.cuda.amp import autocast, GradScaler

from domainbed import networks
import clip
from clip.model import AttentionPool2d
from .base import Algorithm
from .original import ERM
from .sma import MovingAvg, BetaMovingAvg

class CLIP_FinetuneWithTextFreeze(Algorithm): 
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CLIP_FinetuneWithTextFreeze, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.featurizer = networks.CLIP(self.hparams)
        self.featurizer.eval() # turn off bn update
        
        self.optimizer = torch.optim.AdamW(
            self.featurizer.clip_model.visual.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.lr_sheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5001, eta_min=0.0)
        self.prompt = torch.cat([clip.tokenize(f'a image of a {cls_name}') for cls_name in hparams['class_names']]).to(self.device)
        self.logit_scale = torch.ones([]) * np.log(1 / 0.01)
        
        self.scaler = GradScaler()
        
    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        with autocast():
            image_features = self.featurizer.forward_image(all_x)
            text_features = self.featurizer.forward_text(self.prompt)

            # normalized features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            logit_scale = self.logit_scale.exp().to(self.device)
            logits_per_image = logit_scale * image_features @ text_features.t()

            loss = F.cross_entropy(logits_per_image, all_y)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {'loss': loss.item()}

    def predict(self, x):
        logits_per_image, _ = self.featurizer(x, self.prompt)
        return logits_per_image



class CLIPood(Algorithm): 
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CLIPood, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.featurizer = networks.CLIP(self.hparams)
        self.featurizer.eval() # turn off bn update
        
        self.optimizer = torch.optim.AdamW(
            self.featurizer.clip_model.visual.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.lr_sheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5000, eta_min=0.0)
        self.prompt = torch.cat([clip.tokenize(f'a image of a {cls_name}') for cls_name in hparams['class_names']]).to(self.device)
        self.logit_scale = torch.ones([]) * np.log(1 / 0.01)
        self._lambda = hparams["lambda"]
        self.scaler = GradScaler()
        
    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        with autocast():
            image_features = self.featurizer.forward_image(all_x)
            text_features = self.featurizer.forward_text(self.prompt)

            # normalized features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            logit_scale = self.logit_scale.exp().to(self.device)
            logits_per_image = logit_scale * image_features @ text_features.t()
            adaptive_weights = self._lambda * logit_scale * (1 - (text_features @ text_features.t()))
            loss = F.cross_entropy(logits_per_image + adaptive_weights[all_y, :], all_y)
            

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {'loss': loss.item()}

    def predict(self, x):
        logits_per_image, _ = self.featurizer(x, self.prompt)
        return logits_per_image

class CLIP_FinetuneWithTextFreeze_EMA(CLIP_FinetuneWithTextFreeze, MovingAvg):
    """
    Empirical Risk Minimization (ERM) with Simple Moving Average (SMA) prediction model
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        CLIP_FinetuneWithTextFreeze.__init__(self, input_shape, num_classes, num_domains, hparams)
        MovingAvg.__init__(self, self.featurizer, rho=hparams.get("ema_rho", None))

    def update(self, minibatches, unlabeled=None):
        loss_dict = CLIP_FinetuneWithTextFreeze.update(self, minibatches, unlabeled=unlabeled)
        self.update_sma()
        return loss_dict
    
    def predict(self, x):
        self.network_sma.eval()
        logits_per_image, _ = self.network_sma(x, self.prompt)
        return logits_per_image

class CLIP_FinetuneWithTextFreeze_BetaEMA(CLIP_FinetuneWithTextFreeze, BetaMovingAvg):
    """
    Empirical Risk Minimization (ERM) with Simple Moving Average (SMA) prediction model
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        CLIP_FinetuneWithTextFreeze.__init__(self, input_shape, num_classes, num_domains, hparams)
        BetaMovingAvg.__init__(self, self.featurizer)

    def update(self, minibatches, unlabeled=None):
        loss_dict = CLIP_FinetuneWithTextFreeze.update(self, minibatches, unlabeled=unlabeled)
        self.update_sma()
        return loss_dict
    
    def predict(self, x):
        self.network_sma.eval()
        logits_per_image, _ = self.network_sma(x, self.prompt)
        return logits_per_image

class CLIPood_BetaEMA(CLIPood, BetaMovingAvg):
    """
    Empirical Risk Minimization (ERM) with Simple Moving Average (SMA) prediction model
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        CLIPood.__init__(self, input_shape, num_classes, num_domains, hparams)
        BetaMovingAvg.__init__(self, self.featurizer)

    def update(self, minibatches, unlabeled=None):
        loss_dict = CLIPood.update(self, minibatches, unlabeled=unlabeled)
        self.update_sma()
        return loss_dict
    
    def predict(self, x):
        self.network_sma.eval()
        logits_per_image, _ = self.network_sma(x, self.prompt)
        return logits_per_image
