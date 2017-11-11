# -*- coding: utf-8 -*-

from json import dumps
from json import loads

from time import gmtime
from time import strftime

from redis import Redis

from desert import Desert
from desert.color import Rgba
from desert import type_router

from fn import Fn


__ALL__ = ['ErosionWorker', 'Erosion']


def _parse_con(con):
  con = str(con)
  host, port = con.split(':')
  return host, int(port)


class ErosionWorker():
  def __init__(
      self,
      con='localhost:6379',
      chan='erosion',
      resolution=1000,
      gsamples=100000,
      show=False,
      path='./',
      verbose=False):

    self.con = str(con)
    host, port = _parse_con(self.con)
    self.host = host
    self.port = port
    self.chan = str(chan)

    self.imsize = int(resolution)
    self.gsamples = int(gsamples)
    self.verbose = verbose
    self.show = show
    self.red = None
    self.count = 0
    self.desert = None
    self.fn = Fn(prefix=path, postfix='.png')

  def __enter__(self):
    print('>> running erosion worker.')
    print('>> listening at: {:s}/{:s}.'.format(self.con, self.chan))
    print('>> resolution: ({:d} {:d}).'.format(self.imsize, self.imsize))
    print('>> gsamples: ', self.gsamples)
    print('>> show: ', self.show)

    self.red = Redis(self.host, self.port)
    self.desert = Desert(self.imsize,
                         show=self.show,
                         gsamples=self.gsamples,
                         verbose=self.verbose).init()
    return self

  def __exit__(self, _type, val, tb):
    self.save()
    print('>> exited. total: {:d}'.format(self.count))
    del self.red
    del self.desert

  def save(self):
    self.desert.save(self.fn.name())
    if self.show and self.desert._gupdated:
      self.desert.show()

  def clear_chan(self):
    l = self.red.llen(self.chan)
    print('>> cleared: {:d}'.format(l))
    self.red.delete(self.chan)

  def _show_test(self, j):
    print('** rec test: {:s}'.format(j['_data']['time']))

  def _erosion_cmd(self, _type, j):
    if _type == '_test':
      self._show_test(j)
    elif _type == '_init':
      self.desert.init(
          fg=Rgba.from_json(j['_data']['fg']),
          bg=Rgba.from_json(j['_data']['bg']),
          )
    elif _type == '_save':
      self.save()
    else:
      print('## warn: erosion. unknown cmd: {:s}'.format(_type))

  def listen(self):
    while True:
      try:
        _, v = self.red.blpop(self.chan)
        self.count += 1
        j = loads(v.decode('utf8'))

        _type = j['_type']
        if _type.startswith('_'):
          self._erosion_cmd(_type, j)
          continue

        try:
          p = type_router(j)
          self.desert.gdraw([p])
          if self.show and self.desert._gupdated:
            self.desert.show()
        except Exception as e:
          print('## err: erosion:\n{:s}'.format(str(j)))
      except KeyboardInterrupt:
        return


class Erosion():
  def __init__(self,
               con='localhost:6379',
               chan='erosion',
               verbose=False):

    self.con = str(con)
    host, port = _parse_con(self.con)
    self.host = host
    self.port = port
    self.chan = str(chan)
    self.count = 0
    self.verbose = verbose
    self.fg = None
    self.bg = None

    print('>> running erosion client.')
    print('>> sending to: {:s}/{:s}.'.format(self.con, self.chan))
    self.red = Redis(self.host, self.port)

  def __enter__(self):
    self.count = 0
    return self

  def __exit__(self, _type, val, tb):
    del self.red

  def _send(self, j):
    self.red.rpush(self.chan, dumps(j))
    self.count += 1

  def clear_chan(self):
    l = self.red.llen(self.chan)
    print('>> cleared: {:d}'.format(l))
    self.red.delete(self.chan)
    return self

  def init(self, fg, bg):
    self.fg = fg
    self.bg = bg
    self._send({
        '_type': '_init',
        '_data': {
            'fg': fg.json(),
            'bg': bg.json(),
            }
        })
    return self

  def save(self):
    print('>> sending save cmd.')
    self._send({'_type': '_save'})
    return self

  def test(self):
    print('** sent test.')
    self.red.rpush(self.chan, dumps({
        '_type': '_test',
        '_data': {'time': strftime("%Y-%m-%dT%H:%M:%S", gmtime())}
        }))
    return self

  def set_fg(self, c):
    assert isinstance(c, Rgba)
    self.fg = c
    return self

  def set_bg(self, c):
    assert isinstance(c, Rgba)
    self.bg = c
    return self

  def send(self, cmds):
    for cmd in cmds:
      if not cmd.has_rgb():
        cmd.rgb(self.fg)
      self._send(cmd.json())
    return self

