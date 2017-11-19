#!/usr/bin/python3
# -*- coding: utf-8 -*-


from sys import argv

from desert import Desert
from desert import box
from desert import bzspl
from desert import circle
from desert import stroke

from desert.color import rgb
from desert.color import white
from desert.color import black

from desert.helpers import filename


VERBOSE = 'vv'

def main(arg):

  imsize = 1000
  with Desert(imsize, verbose=VERBOSE)\
      .init(fg=rgb(1.0, 0.0, 0.0, 0.1),
            bg=white()) as desert:

    draw = desert.draw

    draw([box(0.15, (0.3, 0.3), 2.0),
          box(0.2, (0.2, 0.5), 0.1),
          box((0.05, 0.5), (0.25, 0.9), 1.0),
          box((0.3, 0.2), (0.1, 0.9), 2.0)])

    desert.show()

    desert.set_fg(rgb(0, 0.5, 0.5, 0.1))

    draw([box(0.15, (0.5, 0.3), 2.0),
          box(0.2, (0.5, 0.5), 0.1),
          box((0.05, 0.5), (0.6, 0.9), 1.0),
          box((0.1, 0.2), (0.3, 0.9), 1.0)])

    desert.show()

    desert.set_fg(rgb(0, 0.0, 0.8, 0.1))

    draw([box(0.05, ((0.7, 0.3), (0.7, 0.8)), 1.0)])

    desert.show()

    draw([stroke(((0.1, 0.1),
                  (0.1, 0.1),
                  (0.1, 0.9),
                  (0.1, 0.1)),
                 ((0.9, 0.9),
                  (0.1, 0.9),
                  (0.9, 0.9),
                  (0.2, 0.15)), 2)])

    desert.set_fg(rgb(0, 0.7, 0.2, 0.1))

    draw([circle(0.05, ((0.5, 0.4), (0.8, 0.4)), 1.0)])

    draw([circle(0.05, ((0.9, 0.1),
                        (0.9, 0.15),
                        (0.9, 0.2),
                        (0.9, 0.25),
                        (0.9, 0.3)), 2)\
          .rgb([rgb(0.2, 0.2, 0.9, 0.3),
                rgb(0.9, 0.2, 0.2, 0.3),
                rgb(0.2, 0.9, 0.2, 0.3),
                rgb(0.9, 0.9, 0.2, 0.3),
                rgb(0.2, 0.9, 0.9, 0.3)]),
          circle(0.05, ((0.85, 0.1),
                        (0.85, 0.3)), 2)\
          .rgb([rgb(0.5, 0.2, 0.9, 0.3),
                rgb(0.9, 0.5, 0.2, 0.3)])]).show()

    desert.set_fg(black())

    draw([bzspl([[(0.1, 0.2),
                  (0.3, 0.4),
                  (0.5, 0.6)],
                 [(0.15, 0.25),
                  (0.35, 0.45),
                  (0.55, 0.65)]], 2)]).show()

    draw([bzspl([[(0.1, 0.2),
                  (0.8, 0.3),
                  (0.3, 0.9)]], 2)]).show()

    draw([bzspl([[(0.1, 0.2),
                  (0.8, 0.3),
                  (0.3, 0.9)],
                 [(0.35, 0.25),
                  (0.85, 0.35),
                  (0.35, 0.95)]], 2)]).show()

    draw([bzspl([[(0.1, 0.2),
                  (0.4, 0.25),
                  (0.9, 0.15),
                  (0.9, 0.3),
                  (0.95, 0.45),
                  (0.8, 0.9),
                  (0.1, 0.87)]], 2, closed=True)]).show()

    draw([bzspl([[(0.15, 0.2),
                  (0.45, 0.25),
                  (0.95, 0.15),
                  (0.95, 0.3),
                  (0.98, 0.45),
                  (0.85, 0.9),
                  (0.15, 0.87)]], 2)]).show()

    desert.set_fg(rgb(1, 0, 1, 1))

    draw([bzspl([[(0.5, 0.1),
                  (0.3, 0.2),
                  (0.1, 0.3),
                  (0.4, 0.3),
                  (0.95, 0.45),
                  (0.8, 0.9),
                  (0.1, 0.87)]], 2)]).gforce().show(3).save(filename(arg))



if __name__ == '__main__':
  main(argv)

