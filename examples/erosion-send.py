#!/usr/bin/python3
# -*- coding: utf-8 -*-


from erosion import Erosion
from desert import box
from desert import stroke
from desert import circle

from desert.color import rgb
from desert.color import white
from desert.color import black


VERBOSE = 'vv'

def main():

  with Erosion(verbose=VERBOSE)\
      .init(fg=rgb(1.0, 0.0, 0.0, 0.1),
            bg=white()) as erosion:

    erosion.send([
        box(0.3, (0.3, 0.3), 1.0),
        box(0.2, (0.7, 0.5), 0.1),
        box(0.2, (0.9, 0.9), 1.0),
        box((0.8, 0.2), (0.1, 0.9), 2.0)])

    erosion.set_fg(rgb(0, 0.5, 0.5, 0.1))

    erosion.send([
        box(0.4, (0.3, 0.3), 1.0),
        box(0.3, (0.7, 0.5), 0.1),
        box(0.4, (0.9, 0.9), 1.0),
        box((0.9, 0.2), (0.1, 0.9), 2.0)])

    erosion.set_fg(rgb(0, 0.0, 0.8, 0.1))

    erosion.send([box(0.05, ((0.7, 0.3), (0.7, 0.8)), 1.0)])

    erosion.send([
        stroke(((0.1, 0.1),
                (0.1, 0.1),
                (0.1, 0.9),
                (0.1, 0.1)),
               ((0.9, 0.9),
                (0.1, 0.9),
                (0.9, 0.9),
                (0.2, 0.15)), 2)])

    erosion.set_fg(rgb(0, 0.7, 0.2, 0.1))

    erosion.send([circle(0.05, ((0.5, 0.4), (0.8, 0.4)), 1.0)])

    erosion.send([
        circle(0.05, ((0.9, 0.1),
                      (0.9, 0.15),
                      (0.9, 0.2),
                      (0.9, 0.25),
                      (0.9, 0.3)), 2)\
          .rgb([
              rgb(0.2, 0.2, 0.9, 0.3),
              rgb(0.9, 0.2, 0.2, 0.3),
              rgb(0.2, 0.9, 0.2, 0.3),
              rgb(0.9, 0.9, 0.2, 0.3),
              rgb(0.2, 0.9, 0.9, 0.3),
              ]),

        circle(0.05, ((0.85, 0.1),
                      (0.85, 0.3)), 2)\
          .rgb([
              rgb(0.5, 0.2, 0.9, 0.3),
              rgb(0.9, 0.5, 0.2, 0.3),
              ])
        ])

    # filename is set by worker
    erosion.save()


if __name__ == '__main__':
  main()

