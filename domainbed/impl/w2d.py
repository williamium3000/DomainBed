import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

import copy
import numpy as np
from collections import defaultdict

from .original import ERM
from .sma import MovingAvg
class W2D(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(W2D, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.drop_f = (1 - hparams['rsc_f_drop_factor']) * 100
        self.drop_b = (1 - hparams['rsc_b_drop_factor']) * 100
        self.num_classes = num_classes
        self.drop_spatial = hparams['rsc_f_drop_factor']
        self.drop_batch = hparams['rsc_b_drop_factor']
        self.p = hparams['worst_case_p']
        self.k = hparams['last_k_epoch']

    def update(self, minibatches, unlabeled=None, step=None, swa_model=None):
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

        all_x = self.featurizer.network.conv1(all_x)
        all_x = self.featurizer.network.bn1(all_x)
        all_x = self.featurizer.network.relu(all_x)
        all_x = self.featurizer.network.maxpool(all_x)
        all_x = self.featurizer.network.layer1(all_x)
        all_x = self.featurizer.network.layer2(all_x)
        all_x = self.featurizer.network.layer3(all_x)
        all_x = self.featurizer.network.layer4(all_x)

        # feature dim
        if self.training:
            self.eval()
            x_new = all_x.clone().detach()
            x_new = Variable(x_new.data, requires_grad=True)
            x_new_view = self.featurizer.network.avgpool(x_new)
            x_new_view = x_new_view.view(x_new_view.size(0), -1)
            output = self.classifier(x_new_view)
            class_num = output.shape[1]
            index = all_y
            num_rois = x_new.shape[0]
            num_channel = x_new.shape[1]
            H = x_new.shape[2]
            HW = x_new.shape[2] * x_new.shape[3]
            one_hot = torch.zeros((1), dtype=torch.float32).cuda()
            one_hot = Variable(one_hot, requires_grad=False)
            sp_i = torch.ones([2, num_rois]).long()
            sp_i[0, :] = torch.arange(num_rois)
            sp_i[1, :] = index
            sp_v = torch.ones([num_rois])
            one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()
            one_hot_sparse = Variable(one_hot_sparse, requires_grad=False)
            one_hot = torch.sum(output * one_hot_sparse)
            self.zero_grad()
            one_hot.backward()
            grads_val = x_new.grad.clone().detach()
            grad_channel_mean = torch.mean(grads_val.view(num_rois, num_channel, -1), dim=2)
            feature_map_channel = grad_channel_mean
            grad_channel_mean = grad_channel_mean.view(num_rois, num_channel, 1, 1)
            cam_all = torch.sum(x_new * grad_channel_mean, 1)
            cam_all = cam_all.view(num_rois, HW)
            self.zero_grad()

            spatial_drop_num = int(HW * self.drop_spatial)
            th18_mask_value = torch.sort(cam_all, dim=1, descending=True)[0][:, spatial_drop_num]
            th18_mask_value = th18_mask_value.view(num_rois, 1).expand(num_rois, HW)
            mask_all_cuda = torch.where(cam_all > th18_mask_value, torch.zeros(cam_all.shape).cuda(),torch.ones(cam_all.shape).cuda())
            mask_all = mask_all_cuda.reshape(num_rois, H, H).view(num_rois, 1, H, H)

            cls_prob_before = F.softmax(output, dim=1)
            x_new_view_after = x_new * mask_all
            x_new_view_after = self.featurizer.network.avgpool(x_new_view_after)
            x_new_view_after = x_new_view_after.view(x_new_view_after.size(0), -1)
            x_new_view_after = self.classifier(x_new_view_after)
            cls_prob_after = F.softmax(x_new_view_after, dim=1)
            sp_i = torch.ones([2, num_rois]).long()
            sp_i[0, :] = torch.arange(num_rois)
            sp_i[1, :] = index
            sp_v = torch.ones([num_rois])
            one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()
            before_vector = torch.sum(one_hot_sparse * cls_prob_before, dim=1)
            after_vector = torch.sum(one_hot_sparse * cls_prob_after, dim=1)
            change_vector = before_vector - after_vector - 0.0001
            change_vector = torch.where(change_vector > 0, change_vector, torch.zeros(change_vector.shape).cuda())
            th_fg_value = torch.sort(change_vector, dim=0, descending=True)[0][int(round(float(num_rois) * self.drop_batch))]
            drop_index_fg = change_vector.gt(th_fg_value).long()
            ignore_index_fg = 1 - drop_index_fg
            not_01_ignore_index_fg = ignore_index_fg.nonzero()[:, 0]
            mask_all[not_01_ignore_index_fg.long(), :] = 1
            self.train()
            mask_all = Variable(mask_all, requires_grad=True)
            all_x = all_x * mask_all

        all_x = self.featurizer.network.avgpool(all_x)
        all_x = all_x.view(all_x.size(0), -1)
        all_x = self.classifier(all_x)

        loss = F.cross_entropy(all_x, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
    


class W2D_v2(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(W2D_v2, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.drop_f = (1 - hparams['rsc_f_drop_factor']) * 100
        self.drop_b = (1 - hparams['rsc_b_drop_factor']) * 100
        self.num_classes = num_classes
        self.p = hparams['worst_case_p']
        self.k = hparams['last_k_epoch']

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
        loss = F.cross_entropy(all_p_muted_again, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class W2D_v2_EMA(W2D_v2, MovingAvg):
    """
    Empirical Risk Minimization (ERM) with Simple Moving Average (SMA) prediction model
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        W2D_v2.__init__(self, input_shape, num_classes, num_domains, hparams)
        MovingAvg.__init__(self, self.network)

    def update(self, minibatches, unlabeled=None):
        loss_dict = super().update(self, minibatches, unlabeled=None)
        self.update_sma()
        return loss_dict
    
    def predict(self, x):
        self.network_sma.eval()
        return self.network_sma(x)
    
class W2D_EMA(W2D, MovingAvg):
    """
    Empirical Risk Minimization (ERM) with Simple Moving Average (SMA) prediction model
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        W2D.__init__(self, input_shape, num_classes, num_domains, hparams)
        MovingAvg.__init__(self, self.network)

    def update(self, minibatches, unlabeled=None):
        loss_dict = super().update(self, minibatches, unlabeled=None)
        self.update_sma()
        return loss_dict
    
    def predict(self, x):
        self.network_sma.eval()
        return self.network_sma(x)