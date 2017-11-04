# -*- coding: utf-8 -*-

from json import dumps
from json import loads

from time import gmtime
from time import strftime

from redis import Redis

from desert import Desert

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
    print('>> resolution: {:d}.'.format(self.imsize))
    self.red = Redis(self.host, self.port)
    self.desert = Desert(self.imsize, show=False, verbose=self.verbose)
    return self

  def __exit__(self, _type, val, tb):
    print('>> exited. total: {:d}'.format(self.count))
    del self.red

  def clear(self):
    l = self.red.llen(self.chan)
    print('>> cleared: {:d}'.format(l))
    self.red.delete(self.chan)

  def test(self):
    print('** sent test.')
    self.red.rpush(self.chan, dumps({
        '_type': 'test',
        '_data': {'time': strftime("%Y-%m-%dT%H:%M:%S", gmtime())}
        }))

  def _show_test(self, j):
    print('** rec test: {:s}'.format(j['_data']['time']))

  def listen(self):
    while True:
      try:
        _, v = self.red.blpop(self.chan)
        self.count += 1
        j = loads(v.decode('utf8'))
        _type = j['_type']
        if _type == 'test':
          self._show_test(j)
          continue

        if _type == 'save':
          self.desert.imsave('./tmp.png')
          continue

        if self.verbose:
          cmd = type_router(j)
          self.desert.draw([cmd])
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

  def send(self, cmds):
    for cmd in cmds:
      self.red.rpush(self.chan, dumps(cmd.json()))
      self.count += 1

  def save(self):
    self.red.rpush(self.chan, dumps({'_type': 'save'}))

  # TODO: clear?

