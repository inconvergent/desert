#!/usr/bin/python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from modules import Desert
from modules.shapes import Box



def main():
  imsize = 2000
  d = Desert(imsize, fg=(1.0, 0.0, 0.0, 0.1),
                     bg=(1.0, 1.0, 1.0, 1.0))

  d.draw([
      Box(0.3, (0.3, 0.3), 1.0),
      Box(0.2, (0.7, 0.5), 0.1),
      Box(0.2, (0.9, 0.9), 1.0),
      Box((0.8, 0.2), (0.1, 0.9), 10.0),
      ])


  fig = plt.figure()
  fig.patch.set_facecolor('gray')
  d.imshow()
  plt.pause(1)


if __name__ == '__main__':
  main()

