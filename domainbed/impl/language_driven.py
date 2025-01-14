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

class LanguageDrivenDG(Algorithm): 
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(LanguageDrivenDG, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.hparams["return_feature"] = True
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        out_feature = self.featurizer.n_outputs
        
        self.clip_model = networks.CLIP(self.hparams)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        if self.clip_model.has_cls_token:
            out_feature_clip = self.clip_model.width
        else:
            out_feature_clip = self.clip_model.num_features
            
        self.atten_pool = AttentionPool2d(input_shape[-1] // 32, out_feature, 32, out_feature_clip)

        self.network = nn.Sequential(self.featurizer, self.atten_pool)
        
        
        
        
        t = hparams.get("t", 1.0)
        self.t = nn.Parameter(torch.ones([]) / t, requires_grad=True)
        self.prompt = torch.cat([clip.tokenize(f'a image of a {cls_name}') for cls_name in hparams['class_names']]).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            list(self.network.parameters()) + [self.t.data, ],
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        
    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        self.clip_model.eval()
        
        features = self.network(all_x) # b, d, h, w
        # features = features.flatten(2).permute(0, 2, 1) # bs N, d
        image_features = features / features.norm(dim=-1, keepdim=True) # bs N, d
        
        with torch.no_grad():
            text_features = self.clip_model.forward_text(self.prompt)
            text_features /= text_features.norm(dim=-1, keepdim=True).detach()
        
        similarity  = self.t * image_features @ text_features.T
        
        loss = F.cross_entropy(similarity, all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        features = self.network(x) # b, d, h, w
        # features = features.flatten(2).permute(0, 2, 1) # bs N, d
        image_features = features / features.norm(dim=-1, keepdim=True) # bs N, d
        
        text_features = self.clip_model.forward_text(self.prompt)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        similarity  = image_features @ text_features.T
        return similarity


class LanguageDrivenDGV2(Algorithm): 
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(LanguageDrivenDGV2, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hparams["return_feature"] = True
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        
        out_feature = self.featurizer.n_outputs
        
        self.clip_model = networks.CLIP(self.hparams)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        out_feature_clip = self.clip_model.num_features
            
        self.atten_pool = AttentionPool2d(input_shape[-1] // 32, out_feature, 32, out_feature_clip)

        self.network = nn.Sequential(self.featurizer, self.atten_pool)
        
        self.class_names = hparams['class_names']
        self.domain_names = hparams['domain_names']
        
        t = hparams.get("t", 1.0)
        self.t1 = nn.Parameter(torch.ones([]) / t, requires_grad=True)
        self.t2 = nn.Parameter(torch.ones([]) / t, requires_grad=True)
        self.t3 = nn.Parameter(torch.ones([]) / t, requires_grad=True)

        self.optimizer = torch.optim.Adam(
            list(self.network.parameters()) + [self.t1.data, self.t2.data, self.t3.data],
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        
    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y, domain_idx in minibatches])
        all_y = torch.cat([y for x, y, domain_idx in minibatches])
        all_domain_idx = torch.cat([domain_idx for x, y, domain_idx in minibatches])
        
        self.clip_model.eval()
        
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
        
        features = self.network(all_x) # b, d, h, w
        # features = features.flatten(2).permute(0, 2, 1) # bs N, d
        image_features = features / features.norm(dim=-1, keepdim=True) # bs N, d
        
        with torch.no_grad():
            class_text_features = self.clip_model.forward_text(class_prompt)
            class_text_features /= class_text_features.norm(dim=-1, keepdim=True).detach()
            
            domain_text_features = self.clip_model.forward_text(domain_prompt)
            domain_text_features /= domain_text_features.norm(dim=-1, keepdim=True).detach()
            
            domain_class_text_features = self.clip_model.forward_text(domain_class_prompt)
            domain_class_text_features /= domain_class_text_features.norm(dim=-1, keepdim=True).detach()
        
        
        loss_cls = F.cross_entropy(self.t1 * image_features @ class_text_features.T, all_y)
        loss_domain = F.cross_entropy(self.t2 * image_features @ domain_text_features.T, new_domain_idx)
        loss_cls_domain = F.cross_entropy(self.t3 * image_features @ domain_class_text_features.T, cls_domain_label)
        
        loss = (loss_cls + 0.5 * loss_domain + 0.5 * loss_cls_domain) / 2
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        prompt = torch.cat([clip.tokenize(f'a image of a {cls_name}') for cls_name in self.class_names]).to(self.device)
        
        features = self.network(x)# b, d, h, w
        # features = features.flatten(2).permute(0, 2, 1) # bs N, d
        image_features = features / features.norm(dim=-1, keepdim=True) # bs N, d
        
        text_features = self.clip_model.forward_text(prompt)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        similarity  = image_features @ text_features.T
        return similarity


class LanguageDrivenDGV2_EMA(LanguageDrivenDGV2, MovingAvg): 
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        LanguageDrivenDGV2.__init__(self, input_shape, num_classes, num_domains, hparams)
        MovingAvg.__init__(self, self.network)
        
    def update(self, minibatches, unlabeled=None):
        return_dict = LanguageDrivenDGV2.update(self, minibatches, unlabeled)
        self.update_sma()
        return return_dict

    def predict(self, x):
        self.network_sma.eval()
        
        prompt = torch.cat([clip.tokenize(f'a image of a {cls_name}') for cls_name in self.class_names]).to(self.device)
        
        features = self.network_sma(x) # b, d, h, w
        # features = features.flatten(2).permute(0, 2, 1) # bs N, d
        image_features = features / features.norm(dim=-1, keepdim=True) # bs N, d
        
        text_features = self.clip_model.forward_text(prompt)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        similarity  = image_features @ text_features.T
        return similarity


class LanguageDrivenDGV3(Algorithm): 
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(LanguageDrivenDGV3, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.hparams["return_feature"] = True
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        out_feature = self.featurizer.n_outputs
        self.classifier = networks.Classifier(
            out_feature,
            num_classes,
            self.hparams['nonlinear_classifier'])
        
        self.clip_model = networks.CLIP(self.hparams)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        if self.clip_model.has_cls_token:
            out_feature_clip = self.clip_model.width
        else:
            out_feature_clip = self.clip_model.num_features
            
        self.atten_pool = AttentionPool2d(input_shape[-1] // 32, out_feature, 32, out_feature_clip)
        self.network = nn.ModuleDict(
            {
                        'featurizer': self.featurizer,
                        'atten_pool': self.atten_pool,
                        'classifier': self.classifier
                })
        t = hparams.get("t", 1.0)
        self.t = nn.Parameter(torch.ones([]) / t, requires_grad=True)
        self.prompt = torch.cat([clip.tokenize(f'a image of a {cls_name}') for cls_name in hparams['class_names']]).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            list(self.network.parameters()) + [self.t.data, ] + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        
    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        self.clip_model.eval()
        
        feat, out = self.featurizer(all_x)
        image_features = self.atten_pool(feat) # b, d, h, w
        # features = features.flatten(2).permute(0, 2, 1) # bs N, d
        image_features = image_features / image_features.norm(dim=-1, keepdim=True) # bs N, d
        
        with torch.no_grad():
            text_features = self.clip_model.forward_text(self.prompt)
            text_features /= text_features.norm(dim=-1, keepdim=True).detach()
        
        similarity  = self.t * image_features @ text_features.T
        
        lang_loss = F.cross_entropy(similarity, all_y)
        

        logits = self.classifier(out)
        
        loss = lang_loss + F.cross_entropy(logits, all_y)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        _, out = self.featurizer(x)
        logits = self.classifier(out)
        return logits

class LanguageDrivenDGV3_EMA(LanguageDrivenDGV3, MovingAvg): 
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        LanguageDrivenDGV3.__init__(self, input_shape, num_classes, num_domains, hparams)
        MovingAvg.__init__(self, self.network)
        
    def update(self, minibatches, unlabeled=None):
        return_dict = LanguageDrivenDGV3.update(self, minibatches, unlabeled)
        self.update_sma()
        return return_dict

    def predict(self, x):
        self.network_sma.eval()
        _, out = self.network_sma["featurizer"](x)
        logits = self.network_sma["classifier"](out)
        return logits

