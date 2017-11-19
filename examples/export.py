#!/usr/bin/python3
# -*- coding: utf-8 -*-


from desert.desert import Desert
from desert import box
from desert import stroke
from desert import circle
from desert import bzspl

from desert.color import rgb
from desert.color import black
from desert.color import Rgba

from desert.helpers import pprint



def main():

  b = box(0.3, (0.3, 0.3), 1.0).rgb(rgb(0.1, 0.3, 0.4, 0.77)).json()
  pprint(b)
  pprint(box.from_json(b).json())
  pprint()

  s = stroke(((0.1, 0.1),
              (0.1, 0.1),
              (0.1, 0.9),
              (0.1, 0.1)),
             ((0.9, 0.9),
              (0.1, 0.9),
              (0.9, 0.9),
              (0.2, 0.15)), 2).rgb(black(0.4)).json()

  pprint(s)
  pprint(stroke.from_json(s).json())
  pprint()

  c = circle(0.05, ((0.5, 0.4), (0.8, 0.4)), 1.0).json()
  pprint(c)
  pprint(circle.from_json(c).json())
  pprint()

  c = circle(0.05, ((0.5, 0.4), (0.8, 0.4)), 1.0)\
      .rgb([
          black(0.3),
          black(0.1)]).json()
  pprint(c)
  pprint(circle.from_json(c).json())
  pprint()

  r = rgb(0.1, 0.4, 0.3, 0.99).json()
  pprint(r)
  pprint(Rgba.from_json(r).json())
  pprint()

  b = bzspl([[(0.1, 0.2),
              (0.4, 0.25),
              (0.9, 0.15),
              (0.9, 0.3),
              (0.95, 0.45),
              (0.8, 0.9)]], 2, closed=True).json()

  pprint(b)
  pprint(bzspl.from_json(b).json())
  pprint()

  b2 = bzspl([[(0.1, 0.2),
               (0.3, 0.4),
               (0.5, 0.6)],
              [(0.15, 0.25),
               (0.35, 0.45),
               (0.55, 0.65)]], 2).json()

  pprint(b2)
  pprint(bzspl.from_json(b2).json())
  pprint()


if __name__ == '__main__':
  main()

