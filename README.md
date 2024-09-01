# CloudSBP

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jehicken.github.io/CloudSBP.jl/)
[![Build Status](https://github.com/jehicken/CloudSBP.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jehicken/CloudSBP.jl/actions/workflows/CI.yml?query=branch%3Amain)

This Julia package is a research code for constructing summation-by-parts (SBP) finite-difference operators on point clouds over complex geometries.  It is the library that implements the algorithm and produced the results in the following paper:

> Jason Hicken, Ge Yan, and Sharanjeet Kaur, _"Constructing stable, high-order finite-difference operators on point clouds over complex geometries,"_ under review.

We are working toward registering the package, but, in the meantime, you will have the follow the instructions [here](https://pkgdocs.julialang.org/v1/managing-packages/#Adding-unregistered-packages) in order to use it.

**Note**: You will also have to install the unregistered `CutQuad.jl` package, which is a dependency used to construct quadrature rules on cut cells.  You can find it in the github repo https://github.com/jehicken/CutQuad.jl.

