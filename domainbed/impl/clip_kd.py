import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

from domainbed import networks
import clip
from .base import Algorithm
from .original import ERM



class ERM_CLIP_Logits(ERM):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM_CLIP_Logits, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        
        self.model = networks.CLIP(self.hparams)

        for param in self.model.parameters():
            param.requires_grad = False

        self.prompt = torch.cat([clip.tokenize(f'a photo of a {cls_name}') for cls_name in hparams['class_names']]).to(self.device)

        self.T = self.hparams['T']
        self.alpha = self.hparams['alpha']
        
    def update(self, minibatches, unlabeled=None):
        self.train()
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        
        pred = self.predict(all_x)
        loss = F.cross_entropy(pred, all_y)
        
        logits_per_image, _ = self.model(all_x, self.prompt)

        teacher_prob = F.softmax(logits_per_image / self.T, dim = 1)
        student_log_prob = F.log_softmax(pred / self.T, dim = 1)
        kd_loss = F.kl_div(student_log_prob, teacher_prob, reduction = 'batchmean') * (self.T ** 2) 
        loss = loss + self.alpha * kd_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

class W2D_v2_CLIP_Logits(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(W2D_v2_CLIP_Logits, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.drop_f = (1 - hparams['rsc_f_drop_factor']) * 100
        self.drop_b = (1 - hparams['rsc_b_drop_factor']) * 100
        self.num_classes = num_classes
        self.p = hparams['worst_case_p']
        self.k = hparams['last_k_epoch']
        
        self.model = networks.CLIP(self.hparams)

        for param in self.model.parameters():
            param.requires_grad = False

        self.prompt = torch.cat([clip.tokenize(f'a photo of a {cls_name}') for cls_name in hparams['class_names']]).to(self.device)

        self.T = self.hparams['T']
        self.alpha = self.hparams['alpha']

    def update(self, minibatches, unlabeled=None, step=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        # inputs
        all_x = torch.cat([x for x, y in minibatches])
        # labels
        all_y = torch.cat([y for _, y in minibatches])
        
        # sample dim
        if step <= int(5000 * (1 - self.k)):
            with torch.no_grad():
                all_p = self.predict(all_x)
                loss_pre = F.cross_entropy(all_p, all_y, reduction='none')
            _, loss_sort_index = torch.sort(-loss_pre)
            loss_sort_index = loss_sort_index[:int(loss_pre.shape[0] * self.p)].long()
            all_x = all_x[loss_sort_index]
            all_y = all_y[loss_sort_index]
            
        # one-hot labels
        all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
        # features
        all_f = self.featurizer(all_x)
        # predictions
        all_p = self.classifier(all_f)
        
        logits_per_image, _ = self.model(all_x, self.prompt)

        teacher_prob = F.softmax(logits_per_image / self.T, dim = 1)
        student_log_prob = F.log_softmax(all_p / self.T, dim = 1)
        kd_loss = F.kl_div(student_log_prob, teacher_prob, reduction = 'batchmean') * (self.T ** 2) 

        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.to(device)).float()

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = self.classifier(all_f_muted)

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples
        all_p_muted_again = self.classifier(all_f * mask)

        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, all_y) + self.alpha * kd_loss
        
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
