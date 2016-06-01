.. eblur/dust documentation master file, created by
   sphinx-quickstart on Thu Jan 28 13:40:11 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Home doc page for *eblur/dust*
==============================

The *eblur/dust* set of python modules calculate scattering absorption
and scattering efficiencies for dust from the infrared to the X-ray.
This code can also be used to calculate dust scattering halo images in
the X-ray, in both interstellar and intergalactic (cosmological)
contexts.

**First published version of this code** (released with `Corrales & Paerels, 2015 <http://adsabs.harvard.edu/abs/2015MNRAS.453.1121C>`_)
http://dx.doi.org/10.5281/zenodo.15991

**Source code:** github.com/eblur/dust

**Support:** If you are having issues, please contact lia@space.mit.edu


Features
--------

A number of dust grain size distributions and optical constants are
provided, but they can be fully customized by the user by invoking
custom objects of the approporiate class.  Provided dust models
include:

* A power law distribution of dust grain sizes
* `Weingartner & Draine (2001) <http://adsabs.harvard.edu/abs/2001ApJ...548..296W>`_ 
  grain size distributions for Milky Way dust
* Optical constants (complex index of refraction) for 0.1 um sized
  `graphite and astrosilicate grains <https://www.astro.princeton.edu/~draine/dust/dust.diel.html>`_

* Rayleigh-Gans scattering physics

  * `Smith & Dwek (1998) <http://adsabs.harvard.edu/abs/1998ApJ...503..831S>`_
  * `Mauche & Gorenstein (1986) <http://adsabs.harvard.edu/abs/1986ApJ...302..371M>`_

* Mie scattering physics using the algorithms of 
  `Bohren & Huffman (1986) <http://adsabs.harvard.edu/abs/1983asls.book.....B>`_

  * Converted from `fortran and IDL
    <http://www.met.tamu.edu/class/atmo689-lc/bhmie.pro>`_ 
    to python


Installation
------------

As of yet there is no static install version.  I recommend cloning the
github repo into a directory in your python path.::

    cd /path/to/python/libraries/
    git clone git@github.com:eblur/dust.git .


Modules
-------
.. toctree::
   :maxdepth: 1
   
   dust
   cmindex
   scatmodels
   sigma_scat
   halo
   galhalo
   analytic
   halodict
   model_halo

..
  License
  -------
  
  Copyright (c) 2014, Lia Corrales
  All rights reserved.
  
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:
  
  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
  
  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
  
  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
  IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
  PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
..

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


