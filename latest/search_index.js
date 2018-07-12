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
    "text": "IRAM.jl approximately solves the eigenproblem Ax = λx where A is a large, sparse and non-symmetric matrix. It is a matrix-free method, and only requires multiplications with  A. It is based on the implicitly restarted Arnoldi method, which be viewed as a mix between a subspace accelerated version of the power method and a truncated version of the dense QR algorithm."
},

{
    "location": "index.html#Pure-Julia-implementation-1",
    "page": "Home",
    "title": "Pure Julia implementation",
    "category": "section",
    "text": "The algorithm is a pure Julia implementation of the implicitly restarted Arnoldi method and is loosely based on ARPACK. It is not our goal to make an exact copy of ARPACK. With \"pure Julia\" we mean that we do not rely on LAPACK for linear algebra routines. This allows us to use any number type.When this project started, ARPACK was still a dependency of the Julia language, and the main goal was to get rid of this. Currently ARPACK has moved to a separate repository called  Arpack.jl, but still it would be great to have a native Julia implementation of this algorithm."
},

{
    "location": "index.html#Status-1",
    "page": "Home",
    "title": "Status",
    "category": "section",
    "text": "Still a work in progress![x] Efficient QR iterations in implicit restart;\n[x] Real arithmetic with real matrices by handling conjugate eigenpairs efficiently;\n[ ] Targeting of eigenvalues;\n[x] Locking of converged Ritz vectors;\n[ ] Generalized eigenvalue problems."
},

]}
