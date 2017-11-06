# -*- coding: utf-8 -*-

"""erosion

Usage:
  erosion [--chan=<c>] [--resolution=<r>] [--clear] [--v | --vv]
  erosion [--chan=<c>] --test
  erosion [--chan=<c>] --save
  erosion [--chan=<c>] --clear-exit
  erosion -h

Options:
  -h                  Show this screen.
  --chan=<c>          Use this channel [default: erosion]
  --resolution=<r>    Canvas resolution. [default: 1000]
  --clear             Clear channel.
  --test              Send a test message to the server, then exit.
  --save              Save then exit.
  --clear-exit        Clear channel, then exit.
  --v                 Verbose.
  --vv                Even more verbose.
  --version           Show version.

Examples:
  erosion --test
  CON='localhost:6379' erosion --chan erosion --clear --v

"""

from os import getenv
import sys
import traceback


from erosion.erosion import ErosionServer
from erosion.erosion import Erosion


__ALL__ = ['ErosionServer', 'Erosion']


def run():
  from docopt import docopt
  args = docopt(__doc__, version='erosion 0.0.1')
  main(args)


def main(args):
  con = str(getenv('CON', 'localhost:6379'))
  chan = str(args['--chan'])
  res = int(args['--resolution'])

  verbose = None
  if args['--vv']:
    verbose = 'vv'
  elif args['--v']:
    verbose = True

  erosion = ErosionServer(con,
                          chan,
                          resolution=res,
                          verbose=verbose)

  try:
    with erosion as er:

      if args['--test']:
        erosion.test()
        exit(1)

      if args['--clear']:
        erosion.clear_chan()

      er.listen()

  except Exception:
    traceback.print_exc(file=sys.stdout)
    exit(1)

