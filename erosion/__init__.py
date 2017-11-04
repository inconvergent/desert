# -*- coding: utf-8 -*-

"""erosion

Usage:
  erosion [--chan=<c>] [--resolution=<r>] [--v | --vv]
  erosion [--chan=<c>] --test
  erosion [--chan=<c>] --clear
  erosion -h

Options:
  -h                  Show this screen.
  --chan=<c>          Use this channel [default: erosion]
  --v                 Verbose.
  --vv                More verbose.
  --resolution=<r>    Canvas resolution. [default: 1000]
  --test              Send a test message to the server then exit.
  --clear             Clear channel.
  --version           Show version.

Examples:
  erosion --test
  CON='localhost:6379' erosion --chan erosion

"""

from os import getenv
# from sys import stderr

from erosion.erosion import ErosionServer
from erosion.erosion import Erosion


__ALL__ = ['ErosionServer', 'Erosion']


def run():
  from docopt import docopt
  args = docopt(__doc__, version='erosion 0.0.1')
  main(args)


def main(args):

  con = getenv('CON', 'localhost:6379')
  chan = args['--chan']
  res = int(args['--resolution'])

  verbose = None
  if args['--vv']:
    verbose = 'vv'
  elif args['--v']:
    verbose = True

  erosion = ErosionServer(
      con,
      chan,
      resolution=res,
      verbose=verbose
      )

  with erosion as er:

    if args['--test']:
      erosion.test()
      exit(1)

    if args['--clear']:
      erosion.clear()
      exit(1)

    er.listen()

  # except Exception as e:
  #   print(e, file=stderr)
  #   exit(1)

