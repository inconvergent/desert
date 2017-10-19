# -*- coding: utf-8 -*-

# import cairocffi as cairo
# from cairocffi import OPERATOR_SOURCE

# import cairo as cairo
# from cairo import OP+ERATOR_SOURCE

# from gi import require_version
# require_version('Gtk', '3.0')

# from gi.repository import Gtk
# from gi.repository import GObject


# from numpy.random import random
# from numpy import pi
# from numpy import sqrt
# from numpy import linspace
# from numpy import arctan2
# from numpy import cos
# from numpy import sin
# from numpy import column_stack
# from numpy import square
# from numpy import array
from numpy import reshape
# from numpy import floor

import torch
from torch.cuda import IntTensor as cIntTensor
from torch.cuda import LongTensor as cLongTensor
from torch.cuda import FloatTensor as cFloatTensor
from torch.sparse import FloatTensor as sFloatTensor

# import torch.nn as nn
# from torch.autograd import Variable
# import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
# import torchvision.models as models

# import copy

unloader = transforms.ToPILImage()  # reconvert into PIL image
# dtype = torch.cuda.FloatTensor #if torch.cuda.is_available()

# TWOPI = pi*2


class Desert():

  def __init__(self, imsize, fg, bg):

    self.imsize = imsize
    self.fg = cFloatTensor(reshape(fg, (3, 1)))
    self.bg = cFloatTensor(reshape(bg, (3, 1)))
    self.vals = torch.cuda.FloatTensor(3, imsize, imsize).fill_(1)
    self.vals[1, :, :].fill_(1)

  def imshow(self):
    imsize = self.imsize
    image = self.vals.clone().cpu()
    image = image.view(3, imsize, imsize)
    image = unloader(image)
    plt.imshow(image)

  def box(self, s, xy, dens):


    try:
      sx, sy = s
    except TypeError:
      sx = s
      sy = s

    imsize = self.imsize
    n = int(sx*sy*dens*(imsize**2))

    coords = cFloatTensor(reshape(xy, (2, 1))) +\
             s*(1.0-2.0*cFloatTensor(2, n).uniform_())
    inds = coords.mul_(imsize).type(cLongTensor)

    self.vals[:,inds[0, :], inds[1, :]] += self.fg



    print(inds)

