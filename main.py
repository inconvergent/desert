#!/usr/bin/python3
# -*- coding: utf-8 -*-


from modules import Desert
from modules.color import rgb
from modules.color import white
from modules.shapes import box



def main():
  imsize = 1000
  d = Desert(imsize, fg=rgb(1.0, 0.0, 0.0, 0.1),
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


  d.imshow(pause=0.5)


if __name__ == '__main__':
  main()

