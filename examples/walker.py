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


def main(arg):
  imsize = 1000
  d = Desert(imsize,
             fg=black(0.1),
             bg=white())

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
      d.draw(stack)
      d.imshow(0.1)
      stack = []

  d.imshow(1)

  try:
    fn = arg[1] + '.png'
  except Exception:
    fn = './tmp.png'

  d.imsave(fn)


if __name__ == '__main__':
  main(argv)

