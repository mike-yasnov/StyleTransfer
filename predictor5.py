#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy
import h5py
import skimage
import os
from skimage import io,transform,img_as_float
from skimage.io import imread,imsave
from collections import OrderedDict
import decimal
notebook_dir = os.getcwd()
project_dir = os.path.split(notebook_dir)[0]
result_dir = project_dir + '/Results/Images/'
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
tmp_dir = project_dir + '/Tmp/'
if not os.path.isdir(tmp_dir):
    os.makedirs(tmp_dir)


# In[2]:


from matplotlib.pyplot import imread
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt

import torch, torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.models as models


# In[3]:


def match_color(target_img, source_img, mode='sym', eps=1e-5):
    '''
    Matches the colour distribution of the target image to that of the source image
    using a linear transform.
    Images are expected to be of form (w,h,c).
    Modes are chol, pca or sym for different choices of basis.
    '''
    mu_t = target_img.mean(0).mean(0)
    t = target_img - mu_t
    t = t.transpose(2,0,1).reshape(3,-1)
    Ct = t.dot(t.T) / t.shape[1] + eps * np.eye(t.shape[0])
    mu_s = source_img.mean(0).mean(0)
    s = source_img - mu_s
    s = s.transpose(2,0,1).reshape(3,-1)
    Cs = s.dot(s.T) / s.shape[1] + eps * np.eye(s.shape[0])
    if mode == 'chol':
        chol_t = np.linalg.cholesky(Ct)
        chol_s = np.linalg.cholesky(Cs)
        ts = chol_s.dot(np.linalg.inv(chol_t)).dot(t)
    if mode == 'pca':
        eva_t, eve_t = np.linalg.eigh(Ct)
        Qt = eve_t.dot(np.sqrt(np.diag(eva_t))).dot(eve_t.T)
        eva_s, eve_s = np.linalg.eigh(Cs)
        Qs = eve_s.dot(np.sqrt(np.diag(eva_s))).dot(eve_s.T)
        ts = Qs.dot(np.linalg.inv(Qt)).dot(t)
    if mode == 'sym':
        eva_t, eve_t = np.linalg.eigh(Ct)
        Qt = eve_t.dot(np.sqrt(np.diag(eva_t))).dot(eve_t.T)
        Qt_Cs_Qt = Qt.dot(Cs).dot(Qt)
        eva_QtCsQt, eve_QtCsQt = np.linalg.eigh(Qt_Cs_Qt)
        QtCsQt = eve_QtCsQt.dot(np.sqrt(np.diag(eva_QtCsQt))).dot(eve_QtCsQt.T)
        ts = np.linalg.inv(Qt).dot(QtCsQt).dot(np.linalg.inv(Qt)).dot(t)
    matched_img = ts.reshape(*target_img.transpose(2,0,1).shape).transpose(1,2,0)
    matched_img += mu_s
    matched_img[matched_img>1] = 1
    matched_img[matched_img<0] = 0
    return matched_img

def lum_transform(image):
    """
    Returns the projection of a colour image onto the luminance channel
    Images are expected to be of form (w,h,c) and float in [0,1].
    """
    img = image.transpose(2,0,1).reshape(3,-1)
    lum = np.array([.299, .587, .114]).dot(img).squeeze()
    img = tile(lum[None,:],(3,1)).reshape((3,image.shape[0],image.shape[1]))
    return img.transpose(1,2,0)

def rgb2luv(image):
    img = image.transpose(2,0,1).reshape(3,-1)
    luv = np.array([[.299, .587, .114],[-.147, -.288, .436],[.615, -.515, -.1]]).dot(img).reshape((3,image.shape[0],image.shape[1]))
    return luv.transpose(1,2,0)
def luv2rgb(image):
    img = image.transpose(2,0,1).reshape(3,-1)
    rgb = np.array([[1, 0, 1.139],[1, -.395, -.580],[1, 2.03, 0]]).dot(img).reshape((3,image.shape[0],image.shape[1]))
    return rgb.transpose(1,2,0)


# In[10]:


def image_loader(image_name, imsize):
    image = resize(np.array(image_name), [imsize[0], imsize[1]])
    return image

def process_image(image,dtype):
    image = image.transpose([2,0,1]) / image.max()
    image = Variable(dtype(image))
    image = image.unsqueeze(0)
    return image

def imshow(tensor, title=None):
    image = tensor.clone().cpu()  # we clone the tensor to not do changes on it
    image = image.view(tensor.shape[1:4])  # remove the fake batch dimension
    image = image.numpy().transpose([1,2,0])
    plt.figure(figsize=[10, 10])
    plt.imshow(image / np.max(image))
    if title is not None:
        plt.title(title)

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


class Predictor3:
    def __init__(self):
        self.model = nn.Sequential()

    def get_image_predict(self, style_path, content_path, mode='basic'):

        use_cuda = torch.cuda.is_available()
        dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        imsize = np.array(np.array(content_path).shape[0:2])// 2
        style_img = image_loader(style_path,imsize)
        content_img = image_loader(content_path,imsize)
        
        if mode == 'basic':
            style_img = process_image(style_img,dtype).type(dtype)
            content_img = process_image(content_img,dtype).type(dtype)
        elif mode == 'match':
            style_img = process_image(match_color(style_img,content_img,mode='pca'),dtype).type(dtype)
            content_img = process_image(content_img,dtype).type(dtype)
        elif mode == 'match_style':
            content_img = process_image(match_color(content_img,style_img,mode='pca'),dtype).type(dtype)
            style_img = process_image(style_img,dtype).type(dtype)
        
        cnn = models.vgg19(pretrained=True).features
        
        if use_cuda:
            cnn = cnn.cuda()

        content_weight = 3 # coefficient for content loss
        style_weight = 1e8  # coefficient for style loss
        content_layers = ('conv_12', 'conv_13', 'conv_14', "conv_15")  # use these layers for content loss
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
                
            
        print('start optimizing')    
        input_image = Variable(content_img.clone().data, requires_grad=True)
        optimizer = torch.optim.LBFGS([input_image])

        num_steps = 500

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
            if i % 100 == 0:  # <--- adjust the value to see updates more frequently
                print('{} iterations'.format(i))
 
                yield input_image
            loss = style_score + content_score

            optimizer.step(lambda: loss)
            optimizer.zero_grad()
            

        # a last correction...
        input_image.data.clamp_(0, 1)
        print('{} iterations'.format(i))
        yield input_image

        self.model = nn.Sequential()
        print('Finish optimizing')


# In[ ]:




