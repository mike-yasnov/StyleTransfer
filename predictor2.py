import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from skimage.transform import resize

import torch, torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.models as models


def image_loader(image_name, imsize, dtype):
    image = resize(np.array(image_name), [imsize[0], imsize[1]])
    image = image.transpose([2,0,1]) / image.max()
    image = Variable(dtype(image))
    image = image.unsqueeze(0)
    return image



class ContentLoss(nn.Module):

    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        self.target = target.detach() * weight
        self.weight = weight

    def forward(self, input):
        self.loss = F.mse_loss(input * self.weight, self.target)
        return input.clone()

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


def gram_matrix(input):
    a, b, c, d = input.size()

    features = input.view(a * b, c * d)

    G = torch.mm(features, features.t())

    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight

    def forward(self, input):
        self.G = gram_matrix(input)
        self.G.mul_(self.weight)
        self.loss = F.mse_loss(self.G, self.target)
        return input.clone()

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


class Predictor:
    def __init__(self):
        self.model = nn.Sequential()

    def get_image_predict(self, style_path='style.jpg', content_path='content.jpg', mode=None):

        use_cuda = torch.cuda.is_available()
        dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        imsize = np.array(np.array(content_path).shape[0:2]) // 2
        style_img = image_loader(style_path,imsize,dtype).type(dtype)
        content_img = image_loader(content_path,imsize,dtype).type(dtype)

        cnn = models.vgg19(pretrained=True).features
        
        if use_cuda:
            cnn = cnn.cuda()

        content_weight = 1 # coefficient for content loss
        style_weight = 1000  # coefficient for style loss
        content_layers = ('conv_4', 'conv_5', 'conv_6', "conv_15")  # use these layers for content loss
        style_layers = ('conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6', 'conv_7', 'conv_8', 'conv_9', 'conv_10', 'conv_11','conv_12', 'conv_13', 'conv_14', 'conv_15', 'conv_16')  # use these layers for style loss

        content_losses = []
        style_losses = []

        if use_cuda:
            self.model = self.model.cuda()

        i = 1
        for layer in list(cnn):
            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(i)
                self.model.add_module(name, layer)

                if name in content_layers:
                    # add content loss:
                    target = self.model(content_img).clone()
                    content_loss = ContentLoss(target, content_weight)
                    self.model.add_module("content_loss_" + str(i), content_loss)
                    content_losses.append(content_loss)

                if name in style_layers:
                    # add style loss:
                    target_feature = self.model(style_img).clone()
                    target_feature_gram = gram_matrix(target_feature)
                    style_loss = StyleLoss(target_feature_gram, style_weight)
                    self.model.add_module("style_loss_" + str(i), style_loss)
                    style_losses.append(style_loss)

            if isinstance(layer, nn.ReLU):
                name = "relu_" + str(i)
                self.model.add_module(name, layer)

                if name in content_layers:
                    # add content loss:
                    target = self.model(content_img).clone()
                    content_loss = ContentLoss(target, content_weight)
                    self.model.add_module("content_loss_" + str(i), content_loss)
                    content_losses.append(content_loss)

                if name in style_layers:
                    # add style loss:
                    target_feature = self.model(style_img).clone()
                    target_feature_gram = gram_matrix(target_feature)
                    style_loss = StyleLoss(target_feature_gram, style_weight)
                    self.model.add_module("style_loss_" + str(i), style_loss)
                    style_losses.append(style_loss)

                i += 1

            if isinstance(layer, nn.MaxPool2d):
                name = "pool_" + str(i)
                self.model.add_module(name, layer)  #
                
            
        print(self.model)    
        input_image = Variable(content_img.clone().data, requires_grad=True)
        optimizer = torch.optim.LBFGS([input_image])

        num_steps = 1000

        for i in range(1, num_steps + 1):
            # correct the values of updated input image
            input_image.data.clamp_(0, 1)

            self.model(input_image)
            style_score = 0
            content_score = 0
            for sl in style_losses:
                style_score += sl.backward()
            for cl in content_losses:
                content_score += cl.backward()
            if i % 200 == 0:  # <--- adjust the value to see updates more frequently

                yield input_image

            loss = style_score + content_score

            optimizer.step(lambda: loss)
            optimizer.zero_grad()

        # a last correction...
        input_image.data.clamp_(0, 1)