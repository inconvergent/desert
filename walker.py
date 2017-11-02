#!/usr/bin/python3
# -*- coding: utf-8 -*-

from sys import argv

from numpy.random import random

from modules.desert import desert
from modules.desert import box
from modules.desert import strokes

from modules.color import black
from modules.color import white

from modules.rnd import in_circle


def main(arg):
  imsize = 1000
  d = desert(imsize,
             fg=black(0.1),
             bg=white())

  num = 20

  xya = random((num, 2))
  xyb = random((num, 2))

  stack = []

  drift = in_circle(1, 0, 0, 0.00001)

  for i in range(10000):
    if not i%1000:
      print(i)
    xya += in_circle(num, 0, 0, 0.001) + drift
    xyb += in_circle(num, 0, 0, 0.001) + drift
    stack.append(strokes(xya, xyb, 0.01))

  d.draw(stack)

  try:
    fn = arg[1] + '.png'
  except Exception:
    fn = './tmp.png'

  print('file:', fn)

  d.imsave(fn)


if __name__ == '__main__':
  main(argv)

