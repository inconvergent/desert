#!/usr/bin/python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from modules import Desert
from modules.shapes import Box


def main():
  imsize = 10
  d = Desert(imsize, fg=(0.0, 0.0, 0.0, 0.1),
                     bg=(1.0, 1.0, 1.0, 1.0))

  fig = plt.figure()
  fig.patch.set_facecolor('gray')

  # ll = []
  # for _ in range(10):
  #   b = Box(0.5, (0.5, 0.5), 1.0)
  #   ll.append(b)
  # d.draw(ll)

  print(d.img)
  d._box(Box(0.5, (0.5, 0.5), 1.0))
  print(d.img)
  # d._box(Box(0.5, (0.5, 0.5), 1.0))
  # print(d.img)
  # d._box(Box(0.5, (0.5, 0.5), 1.0))
  # d._box(Box(0.5, (0.5, 0.5), 1.0))
  # d._box(Box(0.5, (0.5, 0.5), 1.0))
  # d._box(Box(0.5, (0.5, 0.5), 1.0))

  # axes = fig.add_subplot(1, 1, 1, facecolor='red')
  d.imshow()
  plt.pause(1)


if __name__ == '__main__':
  main()

