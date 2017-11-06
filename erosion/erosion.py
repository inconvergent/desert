# -*- coding: utf-8 -*-

from json import dumps
from json import loads

from time import gmtime
from time import strftime

from redis import Redis

from desert import Desert
from desert.color import Rgba
from desert import type_router


__ALL__ = ['ErosionServer', 'Erosion']


def _parse_con(con):
  con = str(con)
  host, port = con.split(':')
  return host, int(port)


class ErosionServer():
  def __init__(
      self,
      con='localhost:6379',
      chan='erosion',
      resolution=1000,
      verbose=False):

    self.con = str(con)
    host, port = _parse_con(self.con)
    self.host = host
    self.port = port
    self.chan = str(chan)

    self.imsize = int(resolution)
    self.verbose = verbose
    self.red = None
    self.count = 0
    self.desert = None

  def __enter__(self):
    print('>> running erosion server.')
    print('>> listening at: {:s}/{:s}.'.format(self.con, self.chan))
    print('>> resolution: ({:d} {:d}).'.format(self.imsize, self.imsize))
    self.red = Redis(self.host, self.port)
    self.desert = Desert(self.imsize, show=False, verbose=self.verbose).init()
    return self

  def __exit__(self, _type, val, tb):
    # TODO: save on exit.
    print('>> exited. total: {:d}'.format(self.count))
    del self.red
    del self.desert

  def clear_chan(self):
    l = self.red.llen(self.chan)
    print('>> cleared: {:d}'.format(l))
    self.red.delete(self.chan)

  def test(self):
    print('** sent test.')
    self.red.rpush(self.chan, dumps({
        '_type': '_test',
        '_data': {'time': strftime("%Y-%m-%dT%H:%M:%S", gmtime())}
        }))

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
      self.desert.imsave('./tmp.png')
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
          cmd = type_router(j)
          self.desert.draw([cmd])
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

  def init(self, fg, bg):
    self._send({
        '_type': '_init',
        '_data': {
            'fg': fg.json(),
            'bg': bg.json(),
            }
        })
    return self

  def send(self, cmds):
    for cmd in cmds:
      self._send(cmd.json())

  def save(self):
    self._send({'_type': '_save'})

  # TODO: clear?

