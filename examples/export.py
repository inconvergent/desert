#!/usr/bin/python3
# -*- coding: utf-8 -*-


from sys import argv

from desert.desert import Desert
from desert.primitives import box
from desert import stroke
from desert import circle

from desert.color import rgb
from desert.color import black
from desert.color import Rgba


VERBOSE = 'vv'

def main(arg):

  b = box(0.3, (0.3, 0.3), 1.0).rgb(rgb(0.1, 0.3, 0.4, 0.77)).json()
  print(b)
  print(box.from_json(b).json())
  print()

  s = stroke(((0.1, 0.1),
              (0.1, 0.1),
              (0.1, 0.9),
              (0.1, 0.1)),
             ((0.9, 0.9),
              (0.1, 0.9),
              (0.9, 0.9),
              (0.2, 0.15)), 2).rgb(black(0.4)).json()

  print(s)
  print(stroke.from_json(s).json())
  print()

  c = circle(0.05, ((0.5, 0.4), (0.8, 0.4)), 1.0).json()
  print(c)
  print(circle.from_json(c).json())
  print()

  r = rgb(0.1, 0.4, 0.3, 0.99).json()
  print(r)
  print(Rgba.from_json(r).json())
  print()


if __name__ == '__main__':
  main(argv)

