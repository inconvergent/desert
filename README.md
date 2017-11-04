# Desert

Desert consists of two parts.

The main library is simply called `Desert`. It is a CUDA accelerated library
for sandpainting: http://inconvergent.net/grains-of-sand/

The second part is called `Erosion`. A Redis-based client and server that can
accept and draw `Desert` primitives encoded as JSON objects.

You do not need to use libraries. `Erosion` is most useful to seperate the
drawing process from the generative algorithm. Eg. if you want to program in a
different language, while still having a fast drawing engine that benefits from
CUDA.

## Examples

Tentative examples can be seen in `examples`.


## Dependencies

The code also depends on:

*    `Redis`
*    `python-redis`
*    `numpy`
*    `pycuda`
*    `matplotlib`


## Install

Use Setuptools to install.

    ./setup.py install --user

or

    ./setup.py develop --user

This will install python libraries `desert` and `erosion`. As well as a shell
command called `erosion` that runs the drawing server. It will be available as
`~/.local/bin/erosion`


## TODO

Desert:

- [x] Box
- [ ] Box: varying size
- [x] Stroke
- [x] Circle
- [ ] Circle: varying rad
- [ ] Spline
- [x] Color
- [ ] Color: hsv, cmyk
- [x] Json import/export of classes


Erosion:

- [x] Basic example using Redis
- [ ] Send clear/bg/fg


## Notes

If cuda is not working try `sudo ldconfig`. and check $LD_LIBRARY_PATH

