#!/usr/bin/python3
# -*- coding: utf-8 -*-


from sys import argv

from modules.desert import desert
from modules.desert import box
from modules.desert import stroke
from modules.desert import circle

from modules.color import rgb
from modules.color import white


VERBOSE = 'vv'

def main(arg):


  imsize = 1000
  d = desert(imsize,
             fg=rgb(1.0, 0.0, 0.0, 0.1),
             bg=white())

  d.draw([
      box(0.3, (0.3, 0.3), 1.0),
      box(0.2, (0.7, 0.5), 0.1),
      box(0.2, (0.9, 0.9), 1.0),
      box((0.8, 0.2), (0.1, 0.9), 2.0)], verbose=VERBOSE)

  d.set_fg(rgb(0, 0.5, 0.5, 0.1))

  d.draw([
      box(0.4, (0.3, 0.3), 1.0),
      box(0.3, (0.7, 0.5), 0.1),
      box(0.4, (0.9, 0.9), 1.0),
      box((0.9, 0.2), (0.1, 0.9), 2.0)], verbose=VERBOSE)

  d.set_fg(rgb(0, 0.0, 0.8, 0.1))

  d.draw([box(0.05, ((0.7, 0.3), (0.7, 0.8)), 1.0)], verbose=VERBOSE)

  d.draw([
      stroke([[0.1, 0.1],
              [0.1, 0.1],
              [0.1, 0.9],
              [0.1, 0.1]],
             [[0.9, 0.9],
              [0.1, 0.9],
              [0.9, 0.9],
              [0.2, 0.15]], 2)], verbose=VERBOSE)

  d.set_fg(rgb(0, 0.7, 0.2, 0.1))

  d.draw([circle(0.05, [[0.5, 0.4], [0.8, 0.4]], 1.0) ], verbose=VERBOSE)

  try:
    fn = arg[1] + '.png'
  except Exception:
    fn = './tmp.png'

  d.imsave(fn)


if __name__ == '__main__':
  main(argv)

