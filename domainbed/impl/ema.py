import copy

class MovingAvg:
    def __init__(self, network, start_iter):
        self.network = network
        self.network_sma = copy.deepcopy(network)
        self.network_sma.eval()
        self.sma_start_iter = start_iter
        self.global_iter = 0
        self.sma_count = 0

    def update_sma(self):
        self.global_iter += 1
        new_dict = {}
        if self.global_iter>=self.sma_start_iter:
            self.sma_count += 1
            for (name,param_q), (_,param_k) in zip(self.network.state_dict().items(), self.network_sma.state_dict().items()):
                if 'num_batches_tracked' not in name:
                   new_dict[name] = ((param_k.data.detach().clone()* self.sma_count + param_q.data.detach().clone()) / (1. + self.sma_count))
        else:
            for (name,param_q), (_,param_k) in zip(self.network.state_dict().items(), self.network_sma.state_dict().items()):
                if 'num_batches_tracked' not in name:
                    new_dict[name] = param_q.detach().data.clone()
        self.network_sma.load_state_dict(new_dict)
