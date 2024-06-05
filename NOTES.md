
* Increasing the stencil size generally improves the conditioning of the Vandermonde matrix.  This also helps the optimization problem, as it becomes easier to solve.  For example for degree=5, I tried using n* + degree and n* + 2*degree nodes in the stencil, and the former needed 87 iterations (on the last opt) and the latter needed 22 iterations.  So, it is a tradeoff between the difficulty of finding an operator, and having an operator with a small stencil.

* Furthermore, it is a good idea to include more than the minimum number of nodes in the stencil, because as the optimization progresses, the Vandermonde matrix for a cell can become large if the stencil is not sufficiently large; that is, the stencil size guards against future changes to the node locations and improves robustness.

* What can be said regarding existence?  If the number of nodes can be related to a element-based mesh, perhaps we can say there exists a quadrature, but the process we use to find the quadrature (that is, using cell) may make it unreachable.  

* We can build diagonal norm SBP operators, or sparse DGD operators.  The usual disadvantages of diagonal-norm SBP operators apply.  However, this must be weighed against the advantages of the diagonal norm for explicit time marching, and, more importantly perhaps, viscous terms.

* If the level-set is aligned with a face, we can have a situation where the face is not cut, but the cell is marked as cut, and then we end up with two faces (one from the background mesh and one from Algorim).  This is more of a concern in 1D, but it seems possible for planar faces in 2D.