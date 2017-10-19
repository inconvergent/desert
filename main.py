#!/usr/bin/python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from modules import Desert



def main():

  imsize = 1000
  d = Desert(imsize, fg=(1, 0.1, 0.5),
                     bg=(0.0, 0.0, 0.0))

  d.box(0.1, (0.5, 0.5), 0.5)

  fig = plt.figure()
  fig.patch.set_facecolor('gray')
  # axes = fig.add_subplot(1, 1, 1, facecolor='red')

  d.imshow()

  plt.pause(1) # pause a bit so that plots are updated


if __name__ == '__main__':
  main()
