#!/usr/bin/python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from modules import Desert


def main():
  imsize = 400
  d = Desert(imsize, fg=(0, 0.0, 0.0, 0.02),
                     bg=(1, 1, 1, 1.0))

  fig = plt.figure()
  fig.patch.set_facecolor('gray')

  d.box(0.5, (0.5, 0.5), 1.0)

  # axes = fig.add_subplot(1, 1, 1, facecolor='red')
  d.imshow()
  plt.pause(1)


if __name__ == '__main__':
  main()

