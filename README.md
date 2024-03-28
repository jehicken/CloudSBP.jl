# CutDGD

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jehicken.github.io/CutDGD.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jehicken.github.io/CutDGD.jl/dev/)
[![Build Status](https://github.com/jehicken/CutDGD.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jehicken/CutDGD.jl/actions/workflows/CI.yml?query=branch%3Amain)


## Notes

* SBP diagonal norm operators here have increased density vs DGD; however, because the DGD mass matrix is non-diagonal, this adds to its memory footprint too (although, in 3D the difference in memory may still favor DGD).