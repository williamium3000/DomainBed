import torch

from domainbed import networks
import clip
from .base import Algorithm
from .original import ERM

# zero-shot CLIP
class CLIP(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CLIP, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = networks.clip(self.hparams)

        for param in self.model.parameters():
            param.requires_grad = False

        self.prompt = torch.cat([clip.tokenize(f'a photo of a {cls_name}') for cls_name in hparams['class_names']]).to(self.device)

    def update(self, minibatches, unlabeled=None):
        return {'loss': 0}

    def predict(self, x):
        logits_per_image, _ = self.model(x, self.prompt)
        return logits_per_image.softmax(dim=-1)

class CLIP_LP(ERM): 
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CLIP_LP, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.featurizer = networks.CLIP(self.hparams)

        # linear probing, results showing that fune-tuning will distort the learned features and lead to worse performance.
        for param in self.featurizer.clip_model.parameters():
            param.requires_grad = False

        self.return_cls = self.featurizer.has_cls_token
        if self.return_cls:
            out_feature_shape = self.featurizer.width
        else:
            out_feature_shape = self.featurizer.num_features

        self.classifier = networks.Classifier(
            out_feature_shape,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
  
    def predict(self, x):
        return self.classifier(self.featurizer.forward_image(x))

class CLIP_LP(ERM): 
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CLIP_LP, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.featurizer = networks.CLIP(self.hparams)
        self.return_cls = self.featurizer.has_cls_token
        if self.return_cls:
            out_feature_shape = self.featurizer.width
        else:
            out_feature_shape = self.featurizer.num_features

        self.classifier = networks.Classifier(
            out_feature_shape,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
  
    def predict(self, x):
        return self.classifier(self.featurizer.forward_image(x))