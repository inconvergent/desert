# -*- coding: utf-8 -*-

from .primitives import box
from .primitives import stroke
from .primitives import circle
from .color import Rgba


types = {
    'box': box.from_json,
    'circle': circle.from_json,
    'stroke': stroke.from_json,
    'rgba': Rgba.from_json,
    }


def type_router(o):
  _type = o['_type']
  return types[_type](o)

