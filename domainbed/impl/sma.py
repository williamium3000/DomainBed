import torch
from torch import nn
import torch.nn.functional as F
import copy
import numpy as np
from domainbed import networks
from .base import Algorithm

class MovingAvg:
    def __init__(self, network, rho=None):
        self.network = network
        self.network_sma = copy.deepcopy(network)
        self.network_sma.eval()
        self.sma_start_iter = 100
        self.global_iter = 0
        self.sma_count = 0
        self.rho = rho
    
    def update_sma(self):
        self.global_iter += 1
        new_dict = {}
        if self.global_iter>=self.sma_start_iter:
            self.sma_count += 1
            curr_pho = self.rho if self.rho is not None else self.sma_count / (1.+self.sma_count)
            for (name,param_q), (_,param_k) in zip(self.network.state_dict().items(), self.network_sma.state_dict().items()):
                if 'num_batches_tracked' not in name:
                   new_dict[name] = param_k.data.detach().clone()* curr_pho + param_q.data.detach().clone() * (1 - curr_pho)
        else:
            for (name,param_q), (_,param_k) in zip(self.network.state_dict().items(), self.network_sma.state_dict().items()):
                if 'num_batches_tracked' not in name:
                    new_dict[name] = param_q.detach().data.clone()
        self.network_sma.load_state_dict(new_dict)


class BetaMovingAvg:
    def __init__(self, network, beta=0.5, total_step=5001):
        self.network = network
        self.network_sma = copy.deepcopy(network)
        self.network_sma.eval()
        self.sma_start_iter = 0
        self.global_iter = 0
        self.sma_count = 0
        self.beta = beta
        self.total_step = total_step - self.sma_start_iter
    
    def update_sma(self):
        self.global_iter += 1
        new_dict = {}
        if self.global_iter>=self.sma_start_iter:
            self.sma_count += 1
            alpha_0_t = np.random.beta(self.beta, self.beta, size=self.sma_count + 1) * (np.arange(0, self.sma_count + 1, 1) + 0.5) / (self.total_step + 1)
            curr_pho = np.sum(alpha_0_t[:-1]) / np.sum(alpha_0_t)
            for (name,param_q), (_,param_k) in zip(self.network.state_dict().items(), self.network_sma.state_dict().items()):
                if 'num_batches_tracked' not in name:
                   new_dict[name] = param_k.data.detach().clone()* curr_pho + param_q.data.detach().clone() * (1 - curr_pho)
        else:
            for (name,param_q), (_,param_k) in zip(self.network.state_dict().items(), self.network_sma.state_dict().items()):
                if 'num_batches_tracked' not in name:
                    new_dict[name] = param_q.detach().data.clone()
        self.network_sma.load_state_dict(new_dict)


class ERM_SMA(Algorithm, MovingAvg):
    """
    Empirical Risk Minimization (ERM) with Simple Moving Average (SMA) prediction model
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        Algorithm.__init__(self, input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
                    self.featurizer.n_outputs,
                    num_classes,
                    self.hparams['nonlinear_classifier'])
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
                        self.network.parameters(),
                        lr=self.hparams["lr"],
                        weight_decay=self.hparams['weight_decay']
                        )
        MovingAvg.__init__(self, self.network)

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss = F.cross_entropy(self.network(all_x), all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_sma()
        return {'loss': loss.item()}

    def predict(self, x):
        self.network_sma.eval()
        return self.network_sma(x)
    