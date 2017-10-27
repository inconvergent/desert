

from numpy import zeros
from numpy.random import random
from numpy import dstack
from numpy import reshape
from numpy import array
from numpy import uint8
from numpy import float32
from numpy import transpose
from numpy import int as npint
from PIL import Image as image


# def unpack2(img, sx, sy):
#   im = uint8(reshape(img, (sy, sx, 4))*255)
#   return image.fromarray(im)


def unpack(img, sx, sy):
  im = uint8(
      transpose(
      dstack((
        reshape(img[:, 0], (sy, sx)),
        reshape(img[:, 1], (sy, sx)),
        reshape(img[:, 2], (sy, sx)),
        ))*255 , (0, 1, 2)))
  print(im.shape)
  print(im)
  return image.fromarray(im)


def rnd(img, sx, sy, rgba, w, h, n=10):
  r, g, b, a = rgba

  ia = 1.0-a
  vals = random((n, 2))*reshape((sx*w, sy*h), (1,2))
  vals = (vals[:, 0]).astype(npint) + sx*(vals[:, 1]).astype(npint)

  for v in vals:
    img[v, 0] = img[v, 0]*ia + r
    img[v, 1] = img[v, 1]*ia + g
    img[v, 2] = img[v, 2]*ia + b



def main():

    sx = 200
    sy = 300

    val = zeros((sx*sy, 4), float32)

    val[:, :] = 0.5

    rnd(val, sx, sy, (0, 0, 0, 0.5), 0.5, 1.0 ,100000)

    val[0, 0] = 1
    val[1, 1] = 1
    val[250, 1] = 1
    print(50+1*200)
    val[sx*sy-1, 2] = 1

    res = unpack(val, sx, sy)

    res.save("/tmp/res.png")


if __name__ == "__main__":
    main()

