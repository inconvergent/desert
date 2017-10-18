# -*- coding: utf-8 -*-

# import cairocffi as cairo
# from cairocffi import OPERATOR_SOURCE

# import cairo as cairo
# from cairo import OPERATOR_SOURCE

from gi import require_version
require_version('Gtk', '3.0')

from gi.repository import Gtk
from gi.repository import GObject


from numpy.random import random
from numpy import pi
from numpy import sqrt
from numpy import linspace
from numpy import arctan2
from numpy import cos
from numpy import sin
from numpy import column_stack
from numpy import square
from numpy import array
# from numpy import reshape
# from numpy import floor


TWOPI = pi*2


# class Render(object):

  # def __init__(self,n, back, front):

  #   self.n = n
  #   self.front = front
  #   self.back = back
  #   self.pix = 1./float(n)

  #   self.colors = []
  #   self.ncolors = 0
  #   self.num_img = 0

  #   self.__init_cairo()

  # def __init_cairo(self):

  #   sur = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.n, self.n)
  #   ctx = cairo.Context(sur)
  #   ctx.scale(self.n, self.n)

  #   self.sur = sur
  #   self.ctx = ctx

  #   self.clear_canvas()

  # def clear_canvas(self):

  #   ctx = self.ctx

  #   ctx.set_source_rgba(*self.back)
  #   ctx.rectangle(0, 0, 1, 1)
  #   ctx.fill()
  #   ctx.set_source_rgba(*self.front)

  # def write_to_png(self, fn):

  #   self.sur.write_to_png(fn)
  #   self.num_img += 1

class Desert():

  def __init__(self, size):

    # Render.__init__(self, n, front, back)

    self.size = size

    window = Gtk.Window()
    self.window = window
    window.resize(self.size, self.size)

    window.connect("destroy", self.__destroy)
    darea = Gtk.DrawingArea()
    # darea.connect("expose-event", self.expose)
    self.darea = darea

    window.add(darea)
    window.show_all()

    #self.cr = self.darea.window.cairo_create()
    self.steps = 0
    GObject.idle_add(self.step_wrap)

  def __destroy(self, *args):
    Gtk.main_quit(*args)

  def start(self):
    Gtk.main()

  # def expose(self, *args):

  #   #cr = self.cr
  #   # cr = self.darea.window.cairo_create()
  #   cr = self.darea.get_property('window').cairo_create()
  #   cr.set_source_surface(self.sur, 0, 0)
  #   cr.paint()

  def step_wrap(self):
    # res = self.step(self)
    self.steps += 1
    # self.expose()

    return True

