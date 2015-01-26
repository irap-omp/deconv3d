
WHAT IS IT
==========

This is a work in progress.

FIXME


HOW TO INSTALL
==============

Dependencies
------------

- astropy
- hyperspectral


DEBIAN PACKAGES
---------------

Most of the usual packages can be installed system-wise from repositories.

```
python2.7 python-numpy python-scipy python-astropy python-matplotlib
```

You can also install them via `pip`, it's your choice.


PYTHON PACKAGES
---------------

The following are packages that are not (yet) present in the repositories.

- `bottleneck` : https://pypi.python.org/pypi/Bottleneck/0.7.0

You can install bottleneck via `pip`, too :

    $ sudo pip install bottleneck


MPDAF PACKAGES
--------------

_Optional._

Deconv3D provides a `MUSELineSpreadFunction` class that depends on `mpdaf.MUSE.LSF`.
Follow [MPDAF install instructions](http://urania1.univ-lyon1.fr/mpdaf/chrome/site/DocCoreLib/installation.html).

Deconv3D also accepts MPDAF's Cubes as input.


HOW TO TEST
===========

Simply run `nosetests -v --nocapture --nologcapture` from project's root :

- `-v` is verbose
- `--nocapture` means `print` statements will print
- `--nologcapture` means `logging` statements will print

These options are not mandatory for the tests to pass, but they are useful.

If you don't have `nose`, you can either
```
$ apt-get install python-nose
```
or
```
$ pip install nose
```


HOW TO DOCUMENT
===============

Install sphinx :

```
$ apt-get install python-sphinx
```

Make your changes into the `doc/source` files.

Once its done, go to `doc` directory, run `make html`.


ACRONYMS
========

<sarcasm>
Real geniuses never define acronyms. They understand them genetically.
</sarcasm>


FFT     Fast Fourier Transform
FITS    Flexible Image Transport System
FWHM    Full Width at Half Maximum
HDU     Header Data Unit
LSF     Line Spread Function
MCMC    Markov Chain Monte Carlo
MPDAF   MUSE Python Data Analysis Framework
MUSE    Multi Unit Spectroscopic Explorer
NFM     Narrow Field Mode
PA      Position Angle
PC      ParseC
PSF     Point Spread Function
SNR     Signal to Noise Ratio
        The relative intensity of the signal from the noise
        Should be > 1, or the data is useless
WFM     Wide Field Mode
WSC     World Coordinates System
