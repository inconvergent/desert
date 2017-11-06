#!/usr/bin/python3
# -*- coding: utf-8 -*-


from sys import argv

from desert import Desert
from desert import box
from desert import stroke
from desert import circle

from desert.color import rgb
from desert.color import white

from desert.helpers import filename


VERBOSE = 'vv'

def main(arg):

  fn = filename(arg)

  imsize = 1000
  with Desert(imsize, verbose=VERBOSE)\
      .init(fg=rgb(1.0, 0.0, 0.0, 0.1),
            bg=white()) as desert:

    desert.draw([
        box(0.3, (0.3, 0.3), 1.0),
        box(0.2, (0.7, 0.5), 0.1),
        box(0.2, (0.9, 0.9), 1.0),
        box((0.8, 0.2), (0.1, 0.9), 2.0)])

    desert.show()

    desert.draw([rgb(0, 0.5, 0.5, 0.1)])

    desert.draw([
        box(0.4, (0.3, 0.3), 1.0),
        box(0.3, (0.7, 0.5), 0.1),
        box(0.4, (0.9, 0.9), 1.0),
        box((0.9, 0.2), (0.1, 0.9), 2.0)])

    desert.show()

    desert.draw([rgb(0, 0.0, 0.8, 0.1)])

    desert.draw([box(0.05, ((0.7, 0.3), (0.7, 0.8)), 1.0)])

    desert.show()

    desert.draw([
        stroke(((0.1, 0.1),
                (0.1, 0.1),
                (0.1, 0.9),
                (0.1, 0.1)),
               ((0.9, 0.9),
                (0.1, 0.9),
                (0.9, 0.9),
                (0.2, 0.15)), 2)])

    desert.draw([rgb(0, 0.7, 0.2, 0.1)])

    desert.draw([circle(0.05, ((0.5, 0.4), (0.8, 0.4)), 1.0)])

    desert.show(1)

    desert.save(fn)


if __name__ == '__main__':
  main(argv)

