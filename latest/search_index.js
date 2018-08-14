var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#IRAM.jl-1",
    "page": "Home",
    "title": "IRAM.jl",
    "category": "section",
    "text": "IRAM.jl finds multiple approximate solutions to the eigenproblem  Ax = lambda x where A is a large, sparse and non-symmetric matrix.  It is a matrix-free method, and only requires multiplications with A.  It is based on the implicitly restarted Arnoldi method, which be viewed as a  mix between a subspace accelerated version of the power method and a truncated  version of the dense QR algorithm.Via spectral transformations one can use this package to solve generalized eigenvalue problems Ax = lambda Bx see  Eigenvalue problems and transformations."
},

{
    "location": "index.html#Pure-Julia-implementation-1",
    "page": "Home",
    "title": "Pure Julia implementation",
    "category": "section",
    "text": "The algorithm is a pure Julia implementation of the implicitly restarted  Arnoldi method and is loosely based on ARPACK. It is not our goal to make an  exact copy of ARPACK. With \"pure Julia\" we mean that we do not rely on LAPACK  for linear algebra routines. This allows us to use any number type. In some  occasions we do rely on BLAS.When this project started, ARPACK was still a dependency of the Julia language,  and the main goal was to get rid of this. Currently ARPACK has moved to a  separate repository called  Arpack.jl, but still it  would be great to have a native Julia implementation of this algorithm."
},

{
    "location": "index.html#Status-1",
    "page": "Home",
    "title": "Status",
    "category": "section",
    "text": "Still a work in progress! Currently we have:An efficient dense QR algorithm natively in Julia, used to do implicit restarts and to compute the low-dimensional dense eigenproblem involving the Hessenberg matrix. It is based on implicit shifts and handles real arithmetic efficiently;\nLocking of converged Ritz vectorsWork in progress:Efficient transformation of real Schur vectors to eigenvectors.\nSearch targets."
},

{
    "location": "theory/transformations.html#",
    "page": "Eigenvalue problems",
    "title": "Eigenvalue problems",
    "category": "page",
    "text": ""
},

{
    "location": "theory/transformations.html#Eigenvalue-problems-and-transformations-1",
    "page": "Eigenvalue problems",
    "title": "Eigenvalue problems and transformations",
    "category": "section",
    "text": "In this section we explore several ways to use IRAM.jl to solve the generalized eigenvalue problemAx = lambda BxThis problem arises for instance in:Finite element discretizations, with B a symmetric, positive definite mass  matrix;\nStability analysis of Navier-Stokes equations, where B is semi-definite  and singular;\nSimple finite differences discretizations where typically B = IBecause IRAM.jl only deals with the standard formCx = lambda xwe have to do a spectral transformation whenever B neq I Secondly, to get fast convergence, one typically applies shift-and-invert techniques, which also requires a spectral transformation."
},

{
    "location": "theory/transformations.html#Transformation-to-standard-form-for-non-singular-B-1",
    "page": "Eigenvalue problems",
    "title": "Transformation to standard form for non-singular B",
    "category": "section",
    "text": "If B is nonsingular and easy to factorize, one can define the matrix C = B^-1A and apply IRAM to the eigenproblemCx = lambda xwhich is in standard form. Of course C should not be formed explicity! One only has to provide the action of the matrix-vector product by implementing LinearAlgebra.mul!(y, C, x). The best way to do so is to factorize B up front.IRAM.jl does not yet provide helper functions for this transformation."
},

{
    "location": "theory/transformations.html#Targeting-eigenvalues-with-shift-and-invert-1",
    "page": "Eigenvalue problems",
    "title": "Targeting eigenvalues with shift-and-invert",
    "category": "section",
    "text": "When looking for eigenvalues near a specific target sigma, one can get fast  convergence by using a shift-and-invert technique:Ax = lambda Bx iff (A - sigma B)^-1Bx = theta x text where  theta = frac1lambda - sigmaHere we have casted the generalized eigenvalue problem into standard form with a matrix C = (A - sigma B)^-1B Note that theta is large whenever sigma is close to lambda. This means that we have to target the largest  eigenvalues of C in absolute magnitude. Fast convergence is guaranteed  whenever sigma is close enough to an eigenvalue lambda.Again, one should not construct the matrix C explicitly, but rather implement the matrix-vector product LinearAlgebra.mul!(y, C, x). The best way to do so is to factorize A - sigma B up front.Note that this shift-and-invert strategy simplifies when B = I in which case the matrix in standard form is just C = (A - sigma I)^-1IRAM.jl does not transform the eigenvalues of C back to the eigenvalues of (A B) However, the relation is simply lambda = sigma + theta^-1."
},

{
    "location": "theory/transformations.html#Purification-1",
    "page": "Eigenvalue problems",
    "title": "Purification",
    "category": "section",
    "text": "If B is exactly singular or very ill-conditioned, one cannot work with  C = B^-1A. One can however apply the shift-and-invert method. There is a  catch, since C = (A - sigma B)^-1B has eigenvalues close to zero or exactly zero. When transformed back, these values would corrspond to  lambda = infty The process to remove these eigenvalues is called  purification.IRAM.jl does not yet support this purification idea, but it could in principle be put together along the following lines [MLA]:Start the Arnoldi method with C times a random vector, such that the  initial vector has numerically no components in the null space of C.\nExpand the Krylov subspace with one additional vector and add a zero shift. This corresponds to implicitly multiplying the first vector of the Krylov subspace with C at restart.[MLA]: Meerbergen, Karl, and Alastair Spence. \"Implicitly restarted Arnoldi with purification for the shift-invert transformation.\" Mathematics of Computation of the American Mathematical Society 66.218 (1997): 667-689."
},

{
    "location": "theory/partial_schur.html#",
    "page": "Schur decomposition",
    "title": "Schur decomposition",
    "category": "page",
    "text": ""
},

{
    "location": "theory/partial_schur.html#Partial-Schur-decomposition-1",
    "page": "Schur decomposition",
    "title": "Partial Schur decomposition",
    "category": "section",
    "text": "Internally the Arnoldi method first builds a partial Schur decomposition of a matrix A, and only then transforms this to an eigendecomposition. In some cases one might wish to just have the Schur vectors, because they form a stable, orthonormal basis for the eigenspace.By default IRAM.jl returns a partial Schur decompositionAQ = QRwhere Q is orthonormal of size n times textttnev and R is upper  triangular of size textttnev times textttnev In real arithmetic R is quasi upper triangular, with 2 times 2 blocks on the diagonal  corresponding to conjugate complex-valued eigenpairs.Via [some function] one can transform the partial Schur decomposition to an eigendecomposition. Suppose RY = YS is the eigendecomposition of R, then the eigendecomposition of A isA(QY) = (QY)SNote that Y is upper triangular as well, so the product QY can be computed efficiently and even in-place."
},

{
    "location": "usage/usage.html#",
    "page": "Using IRAM.jl",
    "title": "Using IRAM.jl",
    "category": "page",
    "text": ""
},

{
    "location": "usage/usage.html#Using-IRAM.jl-1",
    "page": "Using IRAM.jl",
    "title": "Using IRAM.jl",
    "category": "section",
    "text": "You can compute the partial Schur form of a matrix with partial_schur:partial_schur(A; min, max, nev, tol, maxiter, which)where A is the n×n matrix whose eigenvalues and eigenvectors you want; min specifies the minimum dimension to which the Hessenberg matrix is reduced after implicit_restart!; max specifies the maximum dimension of the Hessenberg matrix at which point iterate_arnoldi! stops increasing the dimension of the Krylov subspace; nev specifies the minimum amount of eigenvalues the method gives you; tol specifies the criterion that determines when the eigenpairs are considered converged (in practice, smaller tol forces the eigenpairs to converge even more); maxiter specifies the maximum amount of restarts the method can perform before ending the iteration; and which is a Target structure and specifies which eigenvalues are desired (Largest magnitude, smallest real part, etc.).The function call partial_schur returns a tuple (P,prods), where P is a PartialSchur struct with fields Q and R for which A * P.Q ≈ P.Q * P.R. The upper triangular matrix R is of size nev×nev and the unitary matrix Q is of size n×nev. The amount of matrix-vector products computed during the iterations is stored in prods.You can compute the eigenvalues and eigenvectors from the Schur form with schur_to_eigen:schur_to_eigen(P)where P is a PartialSchur struct that contains the partial Schur decomposition of a matrix A. This computes the eigenvalues and eigenvectors of matrix A from the Schur decomposition P.An example of how to use IRAM.jl\'s function partial_schur:julia> using IRAM, LinearAlgebra\n# Generate a sparse matrix\njulia> A = spdiagm(-1 => fill(-1.0, 99), 0 => fill(2.0, 100), 1 => fill(-1.001, 99));\n# Compute Schur form of A\njulia> schur_form,  = partial_schur(A, min = 12, max = 30, nev = 10, tol = 1e-10, maxiter = 20, which=LM());\njulia> Q,R = schur_form.Q, schur_form.R;\njulia> norm(A*Q - Q*R)\n6.336794280593682e-11\n# Compute eigenvalues and eigenvectors of A\njulia> vals, vecs = schur_to_eigen(schur_form);\n# Show that Ax = λx\njulia> norm(A*vecs - vecs*Diagonal(vals))\n6.335460143979987e-11"
},

{
    "location": "usage/usage.html#Applying-shift-and-invert-to-target-smallest-eigenvalues-with-LinearMaps.jl-1",
    "page": "Using IRAM.jl",
    "title": "Applying shift-and-invert to target smallest eigenvalues with LinearMaps.jl",
    "category": "section",
    "text": "using IRAM, LinearMaps\n\n# Define a matrix whose eigenvalues you want\nA = rand(100,100)\n\n# Inverts the problem\nfunction construct_linear_map(A)\n    a = factorize(A)\n    LinearMap{eltype(A)}((y, x) -> ldiv!(y, a, x), size(A,1), ismutating=true)\nend\n\n# Target the largest eigenvalues of the inverted problem\nschur_form,  = partial_schur(construct_linear_map(A), min=11, max=22, nev=10, tol=1e-5, maxiter=100, which=LM())\ninv_vals, vecs = schur_to_eigen(schur_form)\n\n# Eigenvalues have to be inverted to find the smallest eigenvalues of the non-inverted problem.\nvals = ones(length(inv_vals))./inv_vals\n\n# Show that Ax = λx\n@assert norm(A*vecs - vecs*Diagonal(vals)) < 1e-5"
},

{
    "location": "usage/usage.html#Solving-a-generalized-eigenvalue-problem-targeting-smallest-eigenvalues-with-LinearMaps.jl-1",
    "page": "Using IRAM.jl",
    "title": "Solving a generalized eigenvalue problem targeting smallest eigenvalues with LinearMaps.jl",
    "category": "section",
    "text": "using IRAM, LinearMaps\n\n# Define the matrices of the generalized eigenvalue problem\nA, B = rand(100,100), rand(100,100)\n\nstruct ShiftAndInvert{TA,TB,TT}\n    A_lu::TA\n    B::TB\n    temp::TT\nend\n\nfunction (M::ShiftAndInvert)(y,x)\n    mul!(M.temp, M.B, x)\n    ldiv!(y, M.A_lu, M.temp)\nend\n\nfunction construct_linear_map(A,B)\n    a = ShiftAndInvert(factorize(A),B,Vector{eltype(A)}(undef, size(A,1)))\n    LinearMap{eltype(A)}(a, size(A,1), ismutating=true)\nend\n\n# Target the largest eigenvalues of the inverted problem\nschur_form,  = partial_schur(construct_linear_map(A,B), min=11, max=22, nev=10, tol=1e-5, maxiter=100, which=LM())\ninv_vals, vecs = schur_to_eigen(schur_form)\n\n# Eigenvalues have to be inverted to find the smallest eigenvalues of the non-inverted problem.\nvals = ones(length(inv_vals))./inv_vals\n\n# Show that Ax = λBx\n@assert norm(A*vecs - B*vecs*Diagonal(vals)) < 1e-5"
},

]}
