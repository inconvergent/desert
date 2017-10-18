#!/usr/bin/python3
# -*- coding: utf-8 -*-


import torch

# import torch.nn as nn
# from torch.autograd import Variable
# import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
# import torchvision.models as models

# import copy

from modules import Desert


# dtype = torch.cuda.FloatTensor #if torch.cuda.is_available()


unloader = transforms.ToPILImage()  # reconvert into PIL image


def imshow(tensor, imsize):
  image = tensor.clone().cpu()
  image = image.view(3, imsize, imsize)
  image = unloader(image)
  plt.imshow(image)

def box(s, xy, n):
  b = torch.cuda.FloatTensor(2, n).uniform_()
  b.mult_(s)


def main():

  # d = Desert(1000)

  imsize = 1000

  fig = plt.figure()

  fig = plt.figure()
  fig.patch.set_facecolor('gray')

  xy = torch.cuda.FloatTensor(2, 1)
  xy[]

  # box(0.1,  )


  # axes = fig.add_subplot(1, 1, 1, facecolor='red')

  # t = torch.cuda.FloatTensor(imsize, imsize, 4).uniform_()
  t = torch.cuda.FloatTensor(3, imsize, imsize).fill_(1)
  t[1, :, :].fill_(1)
  imshow(t, imsize)
  plt.pause(1) # pause a bit so that plots are updated


  # d.start()








if __name__ == '__main__':
  main()
