# -*- coding: utf-8 -*-

"""erosion

Erosion is a tool for drawing Desert primitives from a Redis queue. This util
can be used to start the Erosion worker, as well as to perform a few other
tasks.

You can read more about Desert at https://github.com/inconvergent/desert

Erosion uses Redis. By default it expects Redis to be available at
localhost:6379. You can override it by setting the "CON" environment variable.
See examples below.

Similarly you can set the number of cuda threads by assiging an integer value
to the "THREADS" environment variable. The default is 512.


Usage:
  erosion worker [--chan=<c>] [--resolution=<r>] [--gsamples=<s>]
                 [--path=<p>] [--clear] [--show] [--v | --vv]
  erosion cli [--chan=<c>] --clear [--v | --vv]
  erosion cli [--chan=<c>] --save [--v | --vv]
  erosion cli [--chan=<c>] --test [--v | --vv]
  erosion -h

Options:
  -h                  Show this screen.
  --v                 Verbose.
  --vv                Even more verbose.
  --version           Show version.
  --chan=<c>          Use this channel [default: erosion]
  --resolution=<r>    Canvas resolution. [default: 1000]
  --gsamples=<s>      Group together samples before drawing. [default: 100000]
  --path=<p>          Store results to this path [default: ./]
  --clear             Send a clear cmd to the worker.
  --show              Show the result while drawing.
  --test              Send a test cmd to the worker.
  --save              Send a save cmd to the worker.

Examples:
  erosion cli --test
  erosion cli --clear
  CON='localhost:6379' erosion cli --save
  erosion worker
  erosion worker --path=/tmp/
  CON='localhost:6379' erosion worker --chan erosion --clear --v

"""

from os import getenv
import sys
import traceback


from erosion.erosion import ErosionWorker
from erosion.erosion import Erosion


__ALL__ = ['ErosionWorker', 'Erosion']

def run_worker(args, con, chan, verbose):
  res = int(args['--resolution'])
  path = str(args['--path'])
  show = bool(args['--show'])
  gsamples = int(args['--gsamples'])
  erosion_worker = ErosionWorker(con, chan,
                                 resolution=res,
                                 gsamples=gsamples,
                                 show=show,
                                 path=path,
                                 verbose=verbose)

  with erosion_worker as er:

    if args['--clear']:
      er.clear_chan()

    er.listen()

def run_cli(args, con, chan, verbose):

  erosion = Erosion(con, chan, verbose=verbose)

  with erosion as er:

    if args['--test']:
      er.test()

    elif args['--clear']:
      er.clear_chan()

    elif args['--save']:
      er.save()

    exit(0)


def run():
  from docopt import docopt
  args = docopt(__doc__, version='erosion 0.1.0')
  main(args)


def main(args):

  con = str(getenv('CON', 'localhost:6379'))
  chan = str(args['--chan'])

  verbose = None
  if args['--vv']:
    verbose = 'vv'
  elif args['--v']:
    verbose = True

  try:
    if args['worker']:
      run_worker(args, con, chan, verbose)
    elif args['cli']:
      run_cli(args, con, chan, verbose)

  except Exception:
    traceback.print_exc(file=sys.stdout)
    exit(1)

