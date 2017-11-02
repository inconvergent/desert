#!/usr/bin/python3
# -*- coding: utf-8 -*-


from modules.desert import desert
from modules.desert import box
from modules.desert import strokes

from modules.color import rgb
from modules.color import white



def main():
  imsize = 1000
  d = desert(imsize,
             fg=rgb(1.0, 0.0, 0.0, 0.1),
             bg=white())

  d.draw([
      box(0.3, (0.3, 0.3), 1.0),
      box(0.2, (0.7, 0.5), 0.1),
      box(0.2, (0.9, 0.9), 1.0),
      box((0.8, 0.2), (0.1, 0.9), 2.0),
      ], verbose=True)

  d.set_fg(rgb(0, 0.5, 0.5, 0.1))

  d.draw([
      box(0.4, (0.3, 0.3), 1.0),
      box(0.3, (0.7, 0.5), 0.1),
      box(0.4, (0.9, 0.9), 1.0),
      box((0.9, 0.2), (0.1, 0.9), 2.0),
      ], verbose=True)

  d.draw([
    strokes([[0.1, 0.1],
             [0.1, 0.1],
             [0.1, 0.9],
             [0.1, 0.1]],
            [[0.9, 0.9],
             [0.1, 0.9],
             [0.9, 0.9],
             [0.2, 0.15]], 1)
    ], verbose=True)

  # d.imshow(pause=10)
  d.imsave('/tmp/test.png')


if __name__ == '__main__':
  main()

