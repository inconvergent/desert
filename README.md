# Desert

Desert consists of two parts.

The main library is simply called `Desert`. It is a CUDA accelerated library
for sandpainting: http://inconvergent.net/grains-of-sand/

The second part is called `Erosion`. A Redis-based client and server that can
accept and draw `Desert` primitives and commands encoded as JSON objects. That
means that you can use the `Erosion` server from any platform as long as you
can construct JSON and it to a Redis queue. Eg. if you want to program in a
different language, while still having a fast drawing engine that benefits from
CUDA.

## Examples

Tentative examples can be seen in `./examples`.


## Dependencies

The code also depends on:

*    `Redis`
*    `python-redis`
*    `numpy`
*    `pycuda`
*    `matplotlib`


## Install

Use Setuptools to install:

    ./setup.py install --user

This will install python libraries `desert` and `erosion`. As well as a shell
command called `erosion` that runs the drawing server. It will be available as
`~/.local/bin/erosion` if you installed with the `--user` flag.


## On Use and Contributions

This code is a tool that I have written for my own use. I release it publicly
in case people find it useful. It is not however intended as a
collaboration/Open Source project. As such I am unlikely to accept PRs, reply
to issues, or take requests.


## Todo

Desert:

- [x] Box
- [x] Stroke
- [x] Circle
- [ ] Spline
- [ ] Circle: varying rad
- [ ] Box: varying size
- [x] Color
- [ ] Color: hsv, cmyk?
- [ ] Accept primitive color
- [x] Json import/export of classes
- [ ] aggregate primitives


Erosion:

- [x] Basic example using Redis
- [x] Init
- [ ] Send clear/bg/fg
- [ ] Move pfloat to erosion (from .json())


## Notes

If cuda is not working try `sudo ldconfig`. and check $LD_LIBRARY_PATH

https://documen.tician.de/pycuda/tutorial.html#executing-a-kernel

http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#kernels

