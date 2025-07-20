import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from models.losses import prototype_loss


class conv_tsa(nn.Module):
    def __init__(self, orig_conv):
        super(conv_tsa, self).__init__()
        # the original conv layer
        self.conv = copy.deepcopy(orig_conv)
        self.conv.weight.requires_grad = False
        planes, in_planes, _, _ = self.conv.weight.size()
        stride, _ = self.conv.stride

        self.alpha = nn.Parameter(torch.ones(planes, in_planes, 1, 1))
        self.alpha.requires_grad = True

    def forward(self, x):
        y = self.conv(x)

        y = y + F.conv2d(x, self.alpha, stride=self.conv.stride)
        return y


class bn_light_film(nn.Module):
    def __init__(self, bn2):
        super(bn_light_film, self).__init__()
        # the original conv layer
        self.bn2 = copy.deepcopy(bn2)
        planes = self.bn2.weight.size()

        self.film_a = nn.Parameter(torch.ones(planes))
        self.film_a.requires_grad = True

    def forward(self, x):
        y = self.bn2(x)

        n, c, h, w = y.size()

        film_a = self.film_a.view(1, c, 1, 1)
        # print(film_a)
        y = torch.mul(y, film_a)
        return y


class pa(nn.Module):
    """
    pre-classifier alignment (PA) mapping from 'Universal Representation Learning from Multiple Domains for Few-shot Classification'
    (https://arxiv.org/pdf/2103.13841.pdf)
    """

    def __init__(self, feat_dim):
        super(pa, self).__init__()
        # define pre-classifier alignment mapping
        self.weight = nn.Parameter(torch.ones(feat_dim, feat_dim, 1, 1))
        self.weight.requires_grad = True

    def forward(self, x):
        if len(list(x.size())) == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.conv2d(x, self.weight.to(x.device))
        x = x.flatten(1)
        # print('finish pa ')
        return x


class reset_architecture(nn.Module):
    """ Attaching task-specific adapters (alpha) and/or PA (beta) to the ResNet backbone """

    def __init__(self, orig_backbone, model_name, use_lF):
        super(reset_architecture, self).__init__()
        # freeze the pretrained backbone
        for k, v in orig_backbone.named_parameters():
            v.requires_grad = False

        # attaching task-specific adapters (alpha) to each convolutional layers
        if model_name == 'resnet18':
            if use_lF:
                for i, block in enumerate(orig_backbone.layer1):
                    if i == 1:
                        for name, m in block.named_children():
                            if name == 'bn2':
                                # if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                                new_bn = bn_light_film(m)
                                setattr(block, name, new_bn)
                                print('finish resnet18 layer1 light_FiLM successfully!\n')

                for i, block in enumerate(orig_backbone.layer2):
                    if i == 1:
                        for name, m in block.named_children():
                            if name == 'bn2':
                                # if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                                new_bn = bn_light_film(m)
                                setattr(block, name, new_bn)
                                print('finish resnet18 layer2 light_FiLM successfully!\n')
            else:
                for block in orig_backbone.layer1:
                    for name, m in block.named_children():
                        if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                            new_conv = conv_tsa(m)
                            setattr(block, name, new_conv)
                            print('finish resnet18 layer1 rsa successfully!')
                for block in orig_backbone.layer2:
                    for name, m in block.named_children():
                        if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                            new_conv = conv_tsa(m)
                            setattr(block, name, new_conv)
                            print('finish resnet18 layer2 rsa successfully!')

            for block in orig_backbone.layer3:
                for name, m in block.named_children():
                    if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                        new_conv = conv_tsa(m)
                        setattr(block, name, new_conv)
                        print('finish resnet18 layer3 rsa successfully!')
            for block in orig_backbone.layer4:
                for name, m in block.named_children():
                    if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                        new_conv = conv_tsa(m)
                        setattr(block, name, new_conv)
                        print('finish resnet18 layer4 rsa successfully!')

            self.backbone = orig_backbone
            # attach pre-classifier alignment mapping (beta)
            self.feat_dim = orig_backbone.layer4[-1].bn2.num_features

        elif model_name == 'conv4':
            if use_lF:
                for name, m in orig_backbone.layer1.named_children():
                    if isinstance(m, nn.BatchNorm2d):
                        new_bn = bn_light_film(m)
                        setattr(orig_backbone.layer1, name, new_bn)
                        print('finish conv4 layer1 light_FiLM successfully!\n')

                for name, m in orig_backbone.layer2.named_children():
                    if isinstance(m, nn.BatchNorm2d):
                        new_bn = bn_light_film(m)
                        setattr(orig_backbone.layer2, name, new_bn)
                        print('finish conv4 layer2 light_FiLM successfully!\n')
            else:
                for name, m in orig_backbone.layer1.named_children():
                    if isinstance(m, nn.Conv2d):
                        new_conv = conv_tsa(m)
                        setattr(orig_backbone.layer1, name, new_conv)
                        print("finish conv4 layer1 rsa successfully!")

                for name, m in orig_backbone.layer2.named_children():
                    if isinstance(m, nn.Conv2d):
                        new_conv = conv_tsa(m)
                        setattr(orig_backbone.layer2, name, new_conv)
                        print("finish conv4 layer2 rsa successfully!")

            for name, m in orig_backbone.layer3.named_children():
                if isinstance(m, nn.Conv2d):
                    new_conv = conv_tsa(m)
                    setattr(orig_backbone.layer3, name, new_conv)
                    print("finish conv4 layer3 rsa successfully!")

            for name, m in orig_backbone.layer4.named_children():
                if isinstance(m, nn.Conv2d):
                    new_conv = conv_tsa(m)
                    setattr(orig_backbone.layer4, name, new_conv)
                    print("finish conv4 layer4 rsa successfully!")

            self.backbone = orig_backbone
            # attach pre-classifier alignment mapping (beta)
            self.feat_dim = 64
        else:
            pass
        beta = pa(self.feat_dim)
        setattr(self, 'beta', beta)
        beta.weight.requires_grad = True

    def forward(self, x):
        return self.backbone.forward(x=x)

    def embed(self, x):
        return self.backbone.embed(x)

    def reset(self):
        # initialize task-specific adapters (alpha)
        for k, v in self.backbone.named_parameters():
            if 'alpha' in k:
                v.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device) * 0.0001
                # print("reset alpha succeed!")
            if 'film_a' in k:
                v.data = torch.ones(v.size(0)).cuda()
                # print("reset film_a succeed!")

        # initialize pre-classifier alignment mapping (beta)
        v = self.beta.weight
        self.beta.weight.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device)



def stf(x, beta):
    zero_tensor = torch.zeros_like(x)
    x_pos = torch.maximum(x, zero_tensor)
    x_neg = torch.minimum(x, zero_tensor)
    x_pos = 1 / torch.pow(torch.log(1 / (x_pos + 1e-5) + 1), beta)
    x_neg = -1 / torch.pow(torch.log(1 / (-x_neg + 1e-5) + 1), beta)
    return x_pos + x_neg

def lla(x, k):
    abs_x = torch.abs(x)
    y = x * (1 + k * torch.exp(-abs_x))

    return y

def tsa(context_images, context_labels, model, max_iter=20, lr=0.1, lr_beta=1, distance='cos',
        tsa_opt='film+alpha+beta', rectify='lla', k=2):
    """
    Optimizing task-specific parameters attached to the ResNet backbone,
    e.g. adapters (alpha) and/or pre-classifier alignment mapping (beta)
    """
    model.eval()
    tsa_opt = tsa_opt
    alpha_params = [v for k, v in model.named_parameters() if 'alpha' in k]
    beta_params = [v for k, v in model.named_parameters() if 'beta' in k]
    film_params = [v for k, v in model.named_parameters() if 'film' in k]

    params = []
    if 'film' in tsa_opt:
        params.append({'params': film_params})
    if 'alpha' in tsa_opt:
        params.append({'params': alpha_params})

    if 'beta' in tsa_opt:
        params.append({'params': beta_params, 'lr': lr_beta})

    optimizer = torch.optim.Adadelta(params, lr=lr)

    if 'alpha' not in tsa_opt:
        with torch.no_grad():
            context_features = model.embed(context_images)
    for i in range(max_iter):
        optimizer.zero_grad()
        model.zero_grad()

        if 'alpha' in tsa_opt:
            # adapt features by task-specific adapters
            context_features = model.embed(context_images)
        if 'beta' in tsa_opt:
            # adapt feature by PA (beta)
            aligned_features = model.beta(context_features)
            # print('bbb')
        else:
            aligned_features = context_features

        """
        rectify the channel values by stf or lla
        """
        if rectify == 'stf':
            aligned_features = stf(aligned_features, 1.3)
        elif rectify == 'lla':
            aligned_features = lla(aligned_features, k)
        else:
            pass

        loss, stat, _ = prototype_loss(aligned_features, context_labels,
                                       aligned_features, context_labels, distance=distance)

        loss.backward()
        optimizer.step()

    return
