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

class CLIP_FinetuneWithTextFreezeWithDomain(Algorithm): 
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CLIP_FinetuneWithTextFreezeWithDomain, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.featurizer = networks.CLIP(self.hparams)
        self.featurizer.eval() # turn off bn update
        
        self.optimizer = torch.optim.AdamW(
            self.featurizer.clip_model.visual.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.lr_sheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5001, eta_min=0.0)
        self.logit_scale = torch.ones([]) * np.log(1 / 0.01)
        
        self.class_names = hparams['class_names']
        self.domain_names = hparams['domain_names']
        print("class name: ", self.class_names)
        print("domain name: ", self.domain_names)
        self.scaler = GradScaler()
        
    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y, domain_idx in minibatches])
        all_y = torch.cat([y for x, y, domain_idx in minibatches])
        all_domain_idx = torch.cat([domain_idx for x, y, domain_idx in minibatches])
        
        with autocast():
            existing_domains = torch.unique(all_domain_idx)
            domain_names = [self.domain_names[i] for i in existing_domains]
            new_domain_idx = all_domain_idx.clone()
            for i, idx in enumerate(existing_domains):
                new_domain_idx[all_domain_idx == idx] = i
            
            class_prompt = torch.cat([clip.tokenize(f'a image of a {cls_name}') for cls_name in self.class_names]).to(self.device)
            domain_prompt = torch.cat([clip.tokenize(f'a image of a {domain_name}') for domain_name in domain_names]).to(self.device)
            domain_class_prompt = []
            for class_name in self.class_names:
                for domain_name in domain_names:
                    domain_class_prompt.append(clip.tokenize(f'a {domain_name} image of a {class_name}'))
            domain_class_prompt = torch.cat(domain_class_prompt).to(self.device)
            
            cls_domain_label = all_y * len(existing_domains) + new_domain_idx
            
            image_features = self.featurizer.forward_image(all_x)
            # normalized features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            
            with torch.no_grad():
                class_text_features = self.featurizer.forward_text(class_prompt)
                class_text_features /= class_text_features.norm(dim=-1, keepdim=True).detach()
                
                domain_text_features = self.featurizer.forward_text(domain_prompt)
                domain_text_features /= domain_text_features.norm(dim=-1, keepdim=True).detach()
                
                domain_class_text_features = self.featurizer.forward_text(domain_class_prompt)
                domain_class_text_features /= domain_class_text_features.norm(dim=-1, keepdim=True).detach()

            # cosine similarity as logits
            logit_scale = self.logit_scale.exp().to(self.device)
            
            loss_cls = F.cross_entropy(logit_scale * image_features @ class_text_features.T, all_y)
            loss_domain = F.cross_entropy(logit_scale * image_features @ domain_text_features.T, new_domain_idx)
            loss_cls_domain = F.cross_entropy(logit_scale * image_features @ domain_class_text_features.T, cls_domain_label)

            loss = (loss_cls + 0.5 * loss_domain + 0.5 * loss_cls_domain) / 2

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {'loss': loss.item()}

    def predict(self, x):
        class_prompt = torch.cat([clip.tokenize(f'a image of a {cls_name}') for cls_name in self.class_names]).to(self.device)
        logits_per_image, _ = self.featurizer(x, class_prompt)
        return logits_per_image

class CLIP_FinetuneWithTextFreezeWithDomain_BetaEMA(CLIP_FinetuneWithTextFreezeWithDomain, BetaMovingAvg):
    """
    Empirical Risk Minimization (ERM) with Simple Moving Average (SMA) prediction model
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        CLIP_FinetuneWithTextFreezeWithDomain.__init__(self, input_shape, num_classes, num_domains, hparams)
        BetaMovingAvg.__init__(self, self.featurizer)

    def update(self, minibatches, unlabeled=None):
        loss_dict = CLIP_FinetuneWithTextFreezeWithDomain.update(self, minibatches, unlabeled=unlabeled)
        self.update_sma()
        return loss_dict
    
    def predict(self, x):
        self.network_sma.eval()
        class_prompt = torch.cat([clip.tokenize(f'a image of a {cls_name}') for cls_name in self.class_names]).to(self.device)
        logits_per_image, _ = self.network_sma(x, class_prompt)
        return logits_per_image

class CLIP_FinetuneWithTextFreezeWithDomain_EMA(CLIP_FinetuneWithTextFreezeWithDomain, MovingAvg):
    """
    Empirical Risk Minimization (ERM) with Simple Moving Average (SMA) prediction model
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        CLIP_FinetuneWithTextFreezeWithDomain.__init__(self, input_shape, num_classes, num_domains, hparams)
        MovingAvg.__init__(self, self.featurizer)

    def update(self, minibatches, unlabeled=None):
        loss_dict = CLIP_FinetuneWithTextFreezeWithDomain.update(self, minibatches, unlabeled=unlabeled)
        self.update_sma()
        return loss_dict
    
    def predict(self, x):
        self.network_sma.eval()
        class_prompt = torch.cat([clip.tokenize(f'a image of a {cls_name}') for cls_name in self.class_names]).to(self.device)
        logits_per_image, _ = self.network_sma(x, class_prompt)
        return logits_per_image

class CLIP_FinetuneWithTextFreezeWithDomainV2(Algorithm): 
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CLIP_FinetuneWithTextFreezeWithDomainV2, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.featurizer = networks.CLIP(self.hparams)
        self.featurizer.eval() # turn off bn update
        
        self.optimizer = torch.optim.AdamW(
            self.featurizer.clip_model.visual.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.lr_sheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5001, eta_min=0.0)
        self.logit_scale = torch.ones([]) * np.log(1 / 0.01)
        
        self.class_names = hparams['class_names']
        self.domain_names = hparams['domain_names']
        print("class name: ", self.class_names)
        print("domain name: ", self.domain_names)
        
        self.scaler = GradScaler()
        
    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y, domain_idx in minibatches])
        all_y = torch.cat([y for x, y, domain_idx in minibatches])
        all_domain_idx = torch.cat([domain_idx for x, y, domain_idx in minibatches])
        
    
         
        with autocast():
            existing_domains = torch.unique(all_domain_idx)
            domain_names = [self.domain_names[i] for i in existing_domains]
            new_domain_idx = all_domain_idx.clone()
            for i, idx in enumerate(existing_domains):
                new_domain_idx[all_domain_idx == idx] = i
            
            class_prompt = torch.cat([clip.tokenize(f'a image of a {cls_name}') for cls_name in self.class_names]).to(self.device)
            domain_prompt = torch.cat([clip.tokenize(f'a image of a {domain_name}') for domain_name in domain_names]).to(self.device)
            
            image_features = self.featurizer.forward_image(all_x)
            # normalized features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            
            with torch.no_grad():
                class_text_features = self.featurizer.forward_text(class_prompt)
                class_text_features /= class_text_features.norm(dim=-1, keepdim=True).detach()
                
                domain_text_features = self.featurizer.forward_text(domain_prompt)
                domain_text_features /= domain_text_features.norm(dim=-1, keepdim=True).detach()
            
            # cosine similarity as logits
            logit_scale = self.logit_scale.exp().to(self.device)
            
            loss_cls = F.cross_entropy(logit_scale * image_features @ class_text_features.T, all_y)
            loss_domain = F.cross_entropy(logit_scale * image_features @ domain_text_features.T, new_domain_idx)

            loss = (loss_cls + loss_domain) / 2

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {'loss': loss.item()}

    def predict(self, x):
        class_prompt = torch.cat([clip.tokenize(f'a image of a {cls_name}') for cls_name in self.class_names]).to(self.device)
        logits_per_image, _ = self.featurizer(x, class_prompt)
        return logits_per_image

class CLIP_FinetuneWithTextFreezeWithDomainV2_BetaEMA(CLIP_FinetuneWithTextFreezeWithDomainV2, BetaMovingAvg):
    """
    Empirical Risk Minimization (ERM) with Simple Moving Average (SMA) prediction model
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        CLIP_FinetuneWithTextFreezeWithDomainV2.__init__(self, input_shape, num_classes, num_domains, hparams)
        BetaMovingAvg.__init__(self, self.featurizer)

    def update(self, minibatches, unlabeled=None):
        loss_dict = CLIP_FinetuneWithTextFreezeWithDomainV2.update(self, minibatches, unlabeled=unlabeled)
        self.update_sma()
        return loss_dict
    
    def predict(self, x):
        self.network_sma.eval()
        class_prompt = torch.cat([clip.tokenize(f'a image of a {cls_name}') for cls_name in self.class_names]).to(self.device)
        logits_per_image, _ = self.network_sma(x, class_prompt)
        return logits_per_image

class CLIP_FinetuneWithTextFreezeWithDomainV2_EMA(CLIP_FinetuneWithTextFreezeWithDomainV2, MovingAvg):
    """
    Empirical Risk Minimization (ERM) with Simple Moving Average (SMA) prediction model
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        CLIP_FinetuneWithTextFreezeWithDomainV2.__init__(self, input_shape, num_classes, num_domains, hparams)
        MovingAvg.__init__(self, self.featurizer)

    def update(self, minibatches, unlabeled=None):
        loss_dict = CLIP_FinetuneWithTextFreezeWithDomainV2.update(self, minibatches, unlabeled=unlabeled)
        self.update_sma()
        return loss_dict
    
    def predict(self, x):
        self.network_sma.eval()
        class_prompt = torch.cat([clip.tokenize(f'a image of a {cls_name}') for cls_name in self.class_names]).to(self.device)
        logits_per_image, _ = self.network_sma(x, class_prompt)
        return logits_per_image


class CLIP_FinetuneWithTextFreezeWithDomainV3(Algorithm): 
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CLIP_FinetuneWithTextFreezeWithDomainV3, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.featurizer = networks.CLIP(self.hparams)
        self.featurizer.eval() # turn off bn update
        
        self.optimizer = torch.optim.AdamW(
            self.featurizer.clip_model.visual.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.lr_sheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5001, eta_min=0.0)
        self.logit_scale = torch.ones([]) * np.log(1 / 0.01)
        
        self.class_names = hparams['class_names']
        self.domain_names = hparams['domain_names']
        print("class name: ", self.class_names)
        print("domain name: ", self.domain_names)
        self.scaler = GradScaler()
        
    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y, domain_idx in minibatches])
        all_y = torch.cat([y for x, y, domain_idx in minibatches])
        all_domain_idx = torch.cat([domain_idx for x, y, domain_idx in minibatches])
        
        
        with autocast():
            existing_domains = torch.unique(all_domain_idx)
            domain_names = [self.domain_names[i] for i in existing_domains]
            new_domain_idx = all_domain_idx.clone()
            for i, idx in enumerate(existing_domains):
                new_domain_idx[all_domain_idx == idx] = i

            class_prompt = torch.cat([clip.tokenize(f'a image of a {cls_name}') for cls_name in self.class_names]).to(self.device)
            domain_prompt = torch.cat([clip.tokenize(f'a image of a {domain_name}') for domain_name in domain_names]).to(self.device)
            
            domain_class_prompt = []
            for i in range(all_x.size(0)):
                domain_class_prompt.append(
                    torch.cat([clip.tokenize(f'a {self.domain_names[all_domain_idx[i]]} image of a {cls_name}') for cls_name in self.class_names]).to(self.device)
                )
            domain_class_prompt = torch.stack(domain_class_prompt, dim=0)
            
            image_features = self.featurizer.forward_image(all_x)
            # normalized features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            
            with torch.no_grad():
                class_text_features = self.featurizer.forward_text(class_prompt)
                class_text_features /= class_text_features.norm(dim=-1, keepdim=True).detach()
                
                domain_text_features = self.featurizer.forward_text(domain_prompt)
                domain_text_features /= domain_text_features.norm(dim=-1, keepdim=True).detach()
                
                domain_class_text_features = self.featurizer.forward_text(domain_class_prompt)
                domain_class_text_features /= domain_class_text_features.norm(dim=-1, keepdim=True).detach()

            # cosine similarity as logits
            logit_scale = self.logit_scale.exp().to(self.device)
            
            loss_cls = F.cross_entropy(logit_scale * image_features @ class_text_features.T, all_y)
            loss_domain = F.cross_entropy(logit_scale * image_features @ domain_text_features.T, new_domain_idx)
            loss_cls_domain = F.cross_entropy(logit_scale * image_features @ domain_class_text_features.T, all_y)

            loss = (loss_cls + 0.5 * loss_domain + 0.5 * loss_cls_domain) / 2

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {'loss': loss.item()}

    def predict(self, x):
        class_prompt = torch.cat([clip.tokenize(f'a image of a {cls_name}') for cls_name in self.class_names]).to(self.device)
        logits_per_image, _ = self.featurizer(x, class_prompt)
        return logits_per_image