#!/usr/bin/python3
# -*- coding: utf-8 -*-

from sys import argv

from numpy.random import random

from desert import Desert
from desert import box
from desert import stroke

from desert.color import black
from desert.color import white

from desert.rnd import in_circle

from desert.helpers import filename


VERBOSE = 'v'


def main(arg):

  fn = filename(arg)

  imsize = 1000
  with Desert(imsize, verbose=VERBOSE)\
      .init(fg=black(0.1),
            bg=white()) as desert:

    num = 20

    xya = random((num, 2))
    xyb = random((num, 2))

    stack = []

    drift = in_circle(1, 0, 0, 0.00001)

    for i in range(1, 10000):
      xya += in_circle(num, 0, 0, 0.001) + drift
      xyb += in_circle(num, 0, 0, 0.001) + drift
      stack.append(stroke(xya, xyb, 0.01))

      if not i%2000:
        print(i)
        desert.draw(stack)
        desert.show(0.1)
        stack = []

    desert.show(1)

    desert.save(fn)


if __name__ == '__main__':
  main(argv)

