# -*- coding: utf-8 -*-

from .primitives import box
from .primitives import bzspl
from .primitives import circle
from .primitives import stroke
from .color import Rgba


types = {
    'box': box.from_json,
    'circle': circle.from_json,
    'stroke': stroke.from_json,
    'bzspl': bzspl.from_json,
    'rgba': Rgba.from_json,
    }


def type_router(o):
  _type = o['_type']
  return types[_type](o)

