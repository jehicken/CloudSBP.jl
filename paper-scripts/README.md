# Scripts used for the paper

This folder contains the scripts used to generate the date for the paper 

> Jason Hicken, Ge Yan, and Sharanjeet Kaur, _"Constructing stable, high-order finite-difference operators on point clouds over complex geometries,"_ submitted (see also this [preprint](http://arxiv.org/abs/2409.00809) on arxiv)

The scripts are provided in folders with names that match the relevant section of the paper.  Hopefully they are documented and commented well enough to be self-explanatory.

## Running the scripts

My apologies in advance if these scripts do not run "out of the box."  I am not a Julia packaging expert, so there may be some additional steps you need to take to get the scripts to run.  For instance:

* You will have to tell Julia to use packages listed in the `Project.toml`.  When starting Julia, provide the path to this file using `julia --project=/path/to/project` or use `]` and then `activate` to the appropriate path.
* Clone the version of `CloudSBP.jl` with the commit hash `d0baa98908431cd4a14a617bb77e85d08192afba`
* Since `CloudSBP.jl` is not registered at this time, you will need to use `add` or `dev` manually with it to run the scripts here.
* `CloudSBP.jl` uses `CutQuad.jl`, which is our wrapper for Robert Saye's Algoim library.  `CutQuad` itself relies on `CxxWrap`, and sometimes the former breaks when the latter is upgraded.  The version of `CxxWrap` that is needed is indicated by `CutQuad`, but I still find issues arise from time to time.







