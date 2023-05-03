# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from domainbed.lib import wide_resnet
import copy

from timm import create_model
from clip import clip

def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        checkpoint_path = hparams.get("checkpoint_path", None)
        if checkpoint_path is None:
            pretrained = True
        if hparams['model_name'] == "resnet18":
            self.network = torchvision.models.resnet18(pretrained=pretrained)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(pretrained=pretrained)
            self.n_outputs = 2048
        if checkpoint_path is not None:
            self.network.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
            
        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

class ViT(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ViT, self).__init__()
        checkpoint_path = hparams.get("checkpoint_path", None)
        self.network = create_model(hparams["model_name"], pretrained=True if checkpoint_path is None else False, checkpoint_path=checkpoint_path)
        self.n_outputs = self.network.num_features 
        
        # save memory
        del self.network.head 
        self.network.head = Identity()

        self.hparams = hparams

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.network(x)



class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
    elif input_shape[1:3] == (224, 224):
        if hparams["model"] == "resnet":
            return ResNet(input_shape, hparams)
        elif hparams["model"] == "vit":
            return ViT(input_shape[0], hparams)
    else:
        raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)


# function to renormalize the image for CLIP
def denormalize(images, type="imagenet"):
    # images [b, 3, H, W]
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1).type_as(images)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1).type_as(images)
    return std * images + mean


def normalize(images, type="clip"):
    # images [b, 3, h, w]
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device).view(1, 3, 1, 1).type_as(images)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device).view(1, 3, 1, 1).type_as(images)
    return (images - mean) / std

class CLIP(nn.Module):
    def __init__(self, hparams):
        super(CLIP, self).__init__()
        
        # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
        if hparams['clip_model'] not in clip.available_models():
            raise ValueError(f"backbone {hparams['clip_model']} not available")

        print(f'Using {hparams["clip_model"]}...')
        self.clip_model = clip.load(hparams['clip_model'])[0].float()
        #self.clip_model = self.clip_model.eval()

        # embedding dimensions based on CLIP paper https://arxiv.org/pdf/2103.00020.pdf
        if hparams['clip_model'] == 'RN50':
            self.num_features = 1024
        elif hparams['clip_model'] == 'RN101' or hparams['clip_model'] == 'ViT-B/32' or hparams['clip_model'] == 'ViT-B/16':
            self.num_features = 512
        elif hparams['clip_model'] == 'RN50x16' or hparams['clip_model'] == 'ViT-L/14' or hparams['clip_model'] == 'ViT-L/14@336px':
            self.num_features = 748
        elif hparams['clip_model'] == 'RN50x64':
            self.num_features = 1024
        elif hparams['clip_model'] == 'RN50x4':
            self.num_features = 640
        
        if hparams['clip_model'] == 'RN50' or hparams['clip_model'] == 'RN101':
            self.width = 2048
        elif hparams['clip_model'] == 'ViT-B/32' or hparams['clip_model'] == 'ViT-B/16':
            self.width = 768
        elif hparams['clip_model'] == 'ViT-L/14' or hparams['clip_model'] == 'ViT-L/14@336px':
            self.width = 1024
        elif hparams['clip_model'] == 'RN50x16':
            self.width = 3072
        elif hparams['clip_model'] == 'RN50x4':
            self.width = 3072
        elif hparams['clip_model'] == 'RN50x64':
            self.width = 4096
            
        if hparams['clip_model'] == 'RN50' or hparams['clip_model'] == 'RN101' or hparams['clip_model'] == 'RN50x16' or hparams['clip_model'] == 'RN50x64' \
            or hparams['clip_model'] == 'RN50x4':
            self.has_cls_token = False
        else:
            self.has_cls_token = True

    def forward_image(self, x):
        x = normalize(denormalize(x))
        return self.clip_model.encode_image(x)
    
    def forward_text(self, x):
        return self.clip_model.encode_text(x)
    
    def forward(self, img, text):
        img = normalize(denormalize(img))
        return self.clip_model(img, text)
    
    