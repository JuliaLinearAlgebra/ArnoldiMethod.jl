var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#ArnoldiMethod.jl-1",
    "page": "Home",
    "title": "ArnoldiMethod.jl",
    "category": "section",
    "text": "ArnoldiMethod.jl provides an iterative method to find a few approximate  solutions to the eigenvalue problem in standard form:Ax = xlambdawhere A is a general matrix of size n times n; and x in mathbbC^n and lambda in mathbbC are eigenvectors and eigenvalues respectively. By  general matrix we mean that A has no special structure. It can be symmetric or non-symmetric and either real or complex.The method is matrix-free, meaning that it only requires multiplication with  the matrix A.The package exports just two functions:partialschur to compute a stable basis for an eigenspace;\npartialeigen to compute an eigendecomposition from a partial Schur decomposition.See Using ArnoldiMethod.jl  on how to use these  functions."
},

{
    "location": "index.html#What-algorithm-is-ArnoldiMethod.jl?-1",
    "page": "Home",
    "title": "What algorithm is ArnoldiMethod.jl?",
    "category": "section",
    "text": "The underlying algorithm is the restarted Arnoldi method, which be viewed as a mix between a subspace accelerated version of the power method and a truncated  version of the dense QR algorithm.Initially the method was based on the Implicitly Restarted Arnoldi Method (or IRAM for short), which is the algorithm implemented by ARPACK. This method has a very elegant restarting scheme based on exact QR iterations, but is  unfortunately susceptible to forward instabilities of the QR algorithm.For this reason the Krylov-Schur method is currently embraced in this package, which is mathematically equivalent to IRAM, but has better stability by  replacing exact QR iterations with a direct method that reorders the Schur form. In fact we see Krylov-Schur just as an implementation detail of the Arnoldi  method."
},

{
    "location": "index.html#What-problems-does-this-package-solve-specifically?-1",
    "page": "Home",
    "title": "What problems does this package solve specifically?",
    "category": "section",
    "text": "By design the Arnoldi method is best at finding eigenvalues on the boundary of the convex hull of eigenvalues. For instance eigenvalues of largest modulus and largest or smallest real part. In the case of complex matrices one can target eigenvalues of largest and smallest imaginary part as well.The scope is much broader though, since there is a whole zoo of spectral  transformations possible to find for instance interior eigenvalues or  eigenvalues closest to the imaginary axis.Further, one can solve generalized eigenvalue problems Ax = Bx lambda by applying a suitable spectral transformation as well. Also quadratic eigenvalue  problems can be casted to standard form.See Theory for more information."
},

{
    "location": "index.html#Goal-of-this-package:-a-pure-Julia-implementation-1",
    "page": "Home",
    "title": "Goal of this package: a pure Julia implementation",
    "category": "section",
    "text": "This project started with two goals:Having a native Julia implementation of the eigs function that performs as well as ARPACK. With native we mean that its implementation should be generic and support any number type. Currently the partialschur function  does not depend on LAPACK, and removing the last remnants of direct calls to  BLAS is in the pipeline.\nRemoving the dependency of the Julia language on ARPACK. This goal was already achieved before the package was stable enough, since ARPACK moved to a  separate repository  Arpack.jl."
},

{
    "location": "index.html#Status-1",
    "page": "Home",
    "title": "Status",
    "category": "section",
    "text": "An overview of what we have, how it\'s done and what we\'re missing."
},

{
    "location": "index.html#Implementation-details-1",
    "page": "Home",
    "title": "Implementation details",
    "category": "section",
    "text": "The method does not make assumptions about the type of the matrix; it is  matrix-free.\nConverged Ritz vectors are locked (or deflated).\nImportant matrices and vectors are pre-allocated and operations on the  Hessenberg matrix are in-place; Julia\'s garbage collector can sit back.\nKrylov basis vectors are orthogonalized with repeated classical Gram-Schmidt to ensure they are orthogonal up to machine precision; this is a BLAS-2 operation.\nTo compute the Schur decomposition of the Hessenberg matrix we use a dense  QR algorithm written natively in Julia. It is based on implicit (or Francis)  shifts and handles real arithmetic efficiently.\nLocking and purging of Ritz vectors is done by reordering the Schur form,  which is also implemented natively in Julia. In the real case it is done by casting tiny Sylvester equations to linear systems and solving them with  complete pivoting.\nShrinking the size of the Krylov subspace and changing its basis is done by accumulating all rotations and reflections in a unitary matrix Q, and then simply computing the matrix-matrix product V := V * Q, where V is the  original orthonormal basis. This is not in-place in V, but with good reason:  the dense matrix-matrix product is not memory-bound."
},

{
    "location": "index.html#Not-implemented-(yet)-and-future-ideas-1",
    "page": "Home",
    "title": "Not implemented (yet) and future ideas",
    "category": "section",
    "text": "Being able to kickstart the method from a given Arnoldi relation. This also captures:\nMaking an initial guess by providing a known approximate eigenvector;\nDeflating some subspace by starting the Arnoldi method with a given partial Schur decomposition.\nMatrix-induced inner product for generalized eigenvalue problems.\nEfficient implementation of symmetric problems with Lanczos.On my wish list is to allow custom vector or matrix types, so that we can  delegate expensive but trivial work to hardware that can do it faster  (distributed memory / GPU). The basic concept would be: The core Arnoldi method performs tedious linear algebra on the projected,  low-dimensional problem, but finally just outputs a change of basis in the form of a unitary matrix Q.\nAppropriate hardware does the change of basis V := V * Q.Similar things should happen for expansion of the subspace and  orthogonalization."
},

{
    "location": "theory.html#",
    "page": "Theory",
    "title": "Theory",
    "category": "page",
    "text": ""
},

{
    "location": "theory.html#theory-1",
    "page": "Theory",
    "title": "Standard-form eigenvalue problems",
    "category": "section",
    "text": "ArnoldiMethod.jl is intended to find a few approximate solutions to the  eigenvalue problemAx = x lambdaThis problem is handled in two steps:For numerical stability, the method firstly constructs a partial Schur form\nAQ = QR\nwhere Q is orthonormal of size n times textttnev and R is upper  triangular of size textttnev times textttnev In real arithmetic R is quasi upper triangular, with 2 times 2 blocks on the diagonal  corresponding to conjugate complex-valued eigenpairs.\nThe user can transform the partial Schur form into an eigendecomposition via a helper function. The basic math is to determine the eigendecomposition of the upper triangular matrix RY = YLambda such that\nA(QY) = (QY)Lambda\nforms the full eigendecomposition.Step 2 is a cheap post-processing step. Also note that it is not necessary when  the matrix is symmetric, because in that case the Schur decomposition coincides  with the eigendecomposition."
},

{
    "location": "theory.html#Stopping-criterion-1",
    "page": "Theory",
    "title": "Stopping criterion",
    "category": "section",
    "text": "ArnoldiMethod.jl considers an approximate eigenpair converged when the  conditionAx - xlambda_2  texttttollambdais met, where tol is a user provided tolerance. Note that this stopping  criterion is scale-invariant. For a scaled matrix B = alpha A the same  approximate eigenvector together with the scaled eigenvalue alphalambda  would satisfy the stopping criterion."
},

{
    "location": "theory.html#Spectral-transformations-1",
    "page": "Theory",
    "title": "Spectral transformations",
    "category": "section",
    "text": "There are multiple reasons to use a spectral transformation. Firstly, consider the generalized eigenvalue problemAx = BxlambdaThis problem arises for instance in:Finite element discretizations, with B a symmetric, positive definite mass  matrix;\nStability analysis of Navier-Stokes equations, where B is semi-definite  and singular;\nSimple finite differences discretizations where typically B = IBecause ArnoldiMethod.jl only deals with the standard formCx = xlambdawe have to do a spectral transformation whenever B neq I Secondly, to get fast convergence, one typically applies shift-and-invert  techniques, which also requires a spectral transformation."
},

{
    "location": "theory.html#Transformation-to-standard-form-for-non-singular-B-1",
    "page": "Theory",
    "title": "Transformation to standard form for non-singular B",
    "category": "section",
    "text": "If B is nonsingular and easy to factorize, one can define the matrix  C = B^-1A and apply the Arnoldi method to the eigenproblemCx = xlambdawhich is in standard form. Of course C should not be formed explicity! One only has to provide the action of the matrix-vector product by implementing LinearAlgebra.mul!(y, C, x). The best way to do so is to factorize B up front.See an example here."
},

{
    "location": "theory.html#Targeting-eigenvalues-with-shift-and-invert-1",
    "page": "Theory",
    "title": "Targeting eigenvalues with shift-and-invert",
    "category": "section",
    "text": "When looking for eigenvalues near a specific target sigma, one can get fast  convergence by using a shift-and-invert technique:Ax = lambda Bx iff (A - sigma B)^-1Bx = theta x text where  theta = frac1lambda - sigmaHere we have casted the generalized eigenvalue problem into standard form with a matrix C = (A - sigma B)^-1B Note that theta is large whenever sigma is close to lambda. This means that we have to target the largest  eigenvalues of C in absolute magnitude. Fast convergence is guaranteed  whenever sigma is close enough to an eigenvalue lambda.Again, one should not construct the matrix C explicitly, but rather implement the matrix-vector product LinearAlgebra.mul!(y, C, x). The best way to do so is to factorize A - sigma B up front.Note that this shift-and-invert strategy simplifies when B = I in which case the matrix in standard form is just C = (A - sigma I)^-1ArnoldiMethod.jl does not transform the eigenvalues of C back to the  eigenvalues of (A B) However, the relation is simply  lambda = sigma + theta^-1."
},

{
    "location": "theory.html#Purification-1",
    "page": "Theory",
    "title": "Purification",
    "category": "section",
    "text": "If B is exactly singular or very ill-conditioned, one cannot work with  C = B^-1A. One can however apply the shift-and-invert method. There is a  catch, since C = (A - sigma B)^-1B has eigenvalues close to zero or exactly zero. When transformed back, these values would corrspond to  lambda = infty The process to remove these eigenvalues is called  purification.ArnoldiMethod.jl does not yet support this purification idea, but it could in  principle be put together along the following lines [MLA]:Start the Arnoldi method with C times a random vector, such that the  initial vector has numerically no components in the null space of C.\nExpand the Krylov subspace with one additional vector and add a zero shift. This corresponds to implicitly multiplying the first vector of the Krylov subspace with C at restart.[MLA]: Meerbergen, Karl, and Alastair Spence. \"Implicitly restarted Arnoldi with purification for the shift-invert transformation.\" Mathematics of Computation of the American Mathematical Society 66.218 (1997): 667-689."
},

{
    "location": "usage/01_getting_started.html#",
    "page": "Getting started",
    "title": "Getting started",
    "category": "page",
    "text": ""
},

{
    "location": "usage/01_getting_started.html#getting_started-1",
    "page": "Getting started",
    "title": "Getting started",
    "category": "section",
    "text": ""
},

{
    "location": "usage/01_getting_started.html#Installing-1",
    "page": "Getting started",
    "title": "Installing",
    "category": "section",
    "text": "In Julia open the package manager in the REPL via ] and run:(v1.0) pkg> add ArnoldiMethodThen use the package.using ArnoldiMethod"
},

{
    "location": "usage/01_getting_started.html#ArnoldiMethod.partialschur",
    "page": "Getting started",
    "title": "ArnoldiMethod.partialschur",
    "category": "function",
    "text": "partialschur(A; nev, which, tol, mindim, maxdim, restarts) → PartialSchur, History\n\nFind nev approximate eigenpairs of A with eigenvalues near a specified target.\n\nThe matrix A can be any linear map that implements mul!(y, A, x), eltype and size.\n\nThe method will run iteratively until the eigenpairs are approximated to the prescribed tolerance or until restarts restarts have passed.\n\nArguments\n\nThe most important keyword arguments:\n\nKeyword Type Default Description\nnev Int min(6, size(A, 1)) Number of eigenvalues\nwhich Target LM() One of LM(), LR(), SR(), LI(), SI(), see below.\ntol Real √eps Tolerance for convergence: ‖Ax - xλ‖₂ < tol * ‖λ‖\n\nThe target which can be any of subtypes(ArnoldiMethod.Target):\n\nTarget Description\nLM() Largest magnitude: abs(λ) is largest\nLR() Largest real part: real(λ) is largest\nSR() Smallest real part: real(λ) is smallest\nLI() Largest imaginary part: imag(λ) is largest\nSI() Smallest imaginary part: imag(λ) is smallest\n\nnote: Note\nThe targets LI() and SI() only make sense in complex arithmetic. In real arithmetic λ is an eigenvalue iff conj(λ) is an eigenvalue and this  conjugate pair converges simultaneously.\n\nReturn values\n\nThe function returns a tuple\n\ndecomp, history = partialschur(A, ...)\n\nwhere decomp is a PartialSchur struct which  forms a partial Schur decomposition of A to a prescribed tolerance:\n\n> norm(A * decomp.Q - decomp.Q * decomp.R)\n\nhistory is a History struct that holds some basic information about convergence of the method:\n\n> history.converged\ntrue\n> @show history\nConverged after 359 matrix-vector products\n\nAdvanced usage\n\nFurther there are advanced keyword arguments for tuning the algorithm:\n\nKeyword Type Default Description\nmindim Int min(max(10, nev), size(A,1)) Minimum Krylov dimension (≥ nev)\nmaxdim Int min(max(20, 2nev), size(A,1)) Maximum Krylov dimension (≥ min)\nrestarts Int 200 Maximum number of restarts\n\nWhen the algorithm does not converge, one can increase restarts. When the  algorithm converges too slowly, one can play with mindim and maxdim. It is  suggested to keep mindim equal to or slightly larger than nev, and maxdim is usually about two times mindim.\n\n\n\n\n\n"
},

{
    "location": "usage/01_getting_started.html#Construct-a-partial-Schur-decomposition-1",
    "page": "Getting started",
    "title": "Construct a partial Schur decomposition",
    "category": "section",
    "text": "ArnoldiMethod.jl exports the partialschur function which can be used to  obtain a partial Schur decomposition of any matrix A.partialschur"
},

{
    "location": "usage/01_getting_started.html#ArnoldiMethod.partialeigen",
    "page": "Getting started",
    "title": "ArnoldiMethod.partialeigen",
    "category": "function",
    "text": "partialeigen(P::PartialSchur) → (Vector{<:Union{Real,Complex}}, Matrix{<:Union{Real,Complex}})\n\nTransforms a partial Schur decomposition into an eigendecomposition.\n\nnote: Note\nFor real-symmetric and Hermitian matrices the Schur vectors coincide with  the eigenvectors, and hence it is not necessary to call this function in  that case.\n\nThe method still relies on LAPACK to compute the eigenvectors of the (quasi) upper triangular matrix R from the partial Schur decomposition.\n\nnote: Note\nThis method is currently type unstable for real matrices, since we have not yet decided how to deal with complex conjugate pairs of eigenvalues. E.g. if almost all eigenvalues are real, but there are just a few conjugate  pairs, should all eigenvectors be complex-valued?\n\n\n\n\n\n"
},

{
    "location": "usage/01_getting_started.html#From-a-Schur-decomposition-to-an-eigendecomposition-1",
    "page": "Getting started",
    "title": "From a Schur decomposition to an eigendecomposition",
    "category": "section",
    "text": "The eigenvalues and eigenvectors are obtained from the Schur form with the  partialeigen function that is exported by ArnoldiMethod.jl:partialeigen"
},

{
    "location": "usage/01_getting_started.html#Example-1",
    "page": "Getting started",
    "title": "Example",
    "category": "section",
    "text": "Here we compute the first ten eigenvalues and eigenvectors of a tridiagonal sparse matrix.julia> using ArnoldiMethod, LinearAlgebra, SparseArrays\njulia> A = spdiagm(\n           -1 => fill(-1.0, 99),\n            0 => fill(2.0, 100), \n            1 => fill(-1.0, 99)\n       );\njulia> decomp, history = partialschur(A, nev=10, tol=1e-6, which=SR());\njulia> decomp\nPartialSchur decomposition (Float64) of dimension 10\neigenvalues:\n10-element Array{Complex{Float64},1}:\n 0.0009674354160236865 + 0.0im\n  0.003868805732811139 + 0.0im\n  0.008701304061962657 + 0.0im\n   0.01546025527344699 + 0.0im\n  0.024139120518486677 + 0.0im\n    0.0347295035554728 + 0.0im\n   0.04722115887278571 + 0.0im\n   0.06160200160067088 + 0.0im\n    0.0778581192025522 + 0.0im\n   0.09597378493453936 + 0.0im\njulia> history\nConverged: 10 of 10 eigenvalues in 174 matrix-vector products\njulia> norm(A * decomp.Q - decomp.Q * decomp.R)\n6.39386920955869e-8\njulia> λs, X = partialeigen(decomp);\njulia> norm(A * X - X * Diagonal(λs))\n6.393869211477937e-8"
},

{
    "location": "usage/01_getting_started.html#ArnoldiMethod.PartialSchur",
    "page": "Getting started",
    "title": "ArnoldiMethod.PartialSchur",
    "category": "type",
    "text": "PartialSchur(Q, R, eigenvalues)\n\nHolds an orthonormal basis Q and a (quasi) upper triangular matrix R.\n\nFor convenience the eigenvalues that appear on the diagonal of R are also  listed as eigenvalues, which is in particular useful in the case of real  matrices with complex eigenvalues. Note that the eigenvalues are always a  complex, even when the matrix R is real.\n\n\n\n\n\n"
},

{
    "location": "usage/01_getting_started.html#ArnoldiMethod.History",
    "page": "Getting started",
    "title": "ArnoldiMethod.History",
    "category": "type",
    "text": "History(mvproducts, nconverged, converged, nev)\n\nHistory shows whether the method has converged (when nconverged ≥ nev) and how many matrix-vector products were necessary to do so.\n\n\n\n\n\n"
},

{
    "location": "usage/01_getting_started.html#The-PartialSchur-and-History-structs-1",
    "page": "Getting started",
    "title": "The PartialSchur and History structs",
    "category": "section",
    "text": "For completeness, the return values of the partialschur function:ArnoldiMethod.PartialSchur\nArnoldiMethod.History"
},

{
    "location": "usage/02_spectral_transformations.html#",
    "page": "Transformations",
    "title": "Transformations",
    "category": "page",
    "text": ""
},

{
    "location": "usage/02_spectral_transformations.html#Spectral-transformations-1",
    "page": "Transformations",
    "title": "Spectral transformations",
    "category": "section",
    "text": "ArnoldiMethod.jl by default only solves the standard-form eigenvalue problem  Ax = xlambda for lambda close to the boundary of the convex hull of  eigenvalues.Whenever one targets eigenvalues close to a specific point in the complex plane, or whenever one solves generalized eigenvalue problems, spectral transformations will enable you to recast the problem into something that ArnoldiMethod.jl can  handle well. In this section we provide some examples."
},

{
    "location": "usage/02_spectral_transformations.html#Shift-and-invert-with-LinearMaps.jl-1",
    "page": "Transformations",
    "title": "Shift-and-invert with LinearMaps.jl",
    "category": "section",
    "text": "To find eigenvalues closest to the origin of A, one can find the eigenvalues of largest magnitude of A^-1. LinearMaps.jl  is a neat way to implement this.using ArnoldiMethod, LinearAlgebra, LinearMaps\n\n# Define a matrix whose eigenvalues you want\nA = rand(100,100)\n\n# Factorizes A and builds a linear map that applies inv(A) to a vector.\nfunction construct_linear_map(A)\n    F = factorize(A)\n    LinearMap{eltype(A)}((y, x) -> ldiv!(y, F, x), size(A,1), ismutating=true)\nend\n\n# Target the largest eigenvalues of the inverted problem\ndecomp, = partialschur(construct_linear_map(A), nev=4, tol=1e-5, restarts=100, which=LM())\nλs_inv, X = partialeigen(decomp)\n\n# Eigenvalues have to be inverted to find the smallest eigenvalues of the non-inverted problem.\nλs = 1 ./ λs_inv\n \n# Show that Ax = xλ\n@show norm(A * X - X * Diagonal(λs)) # 7.38473677258669e-6"
},

{
    "location": "usage/02_spectral_transformations.html#generalized_shift_invert-1",
    "page": "Transformations",
    "title": "Smallest eigenvalues of generalized eigenvalue problem",
    "category": "section",
    "text": "When targeting the eigenvalues closest to the origin of a generalized eigenvalue problem Ax = Bxlambda, one can apply the shift-and-invert trick, recasting  the problem to A^-1Bx = xtheta where lambda = 1  theta.using ArnoldiMethod, LinearAlgebra, LinearMaps\n\n# Define the matrices of the generalized eigenvalue problem\nA, B = rand(100,100), rand(100,100)\n\nstruct ShiftAndInvert{TA,TB,TT}\n    A_lu::TA\n    B::TB\n    temp::TT\nend\n\nfunction (M::ShiftAndInvert)(y,x)\n    mul!(M.temp, M.B, x)\n    ldiv!(y, M.A_lu, M.temp)\nend\n\nfunction construct_linear_map(A,B)\n    a = ShiftAndInvert(factorize(A),B,Vector{eltype(A)}(undef, size(A,1)))\n    LinearMap{eltype(A)}(a, size(A,1), ismutating=true)\nend\n\n# Target the largest eigenvalues of the inverted problem\ndecomp,  = partialschur(construct_linear_map(A, B), nev=4, tol=1e-5, restarts=100, which=LM())\nλs_inv, X = partialeigen(decomp)\n\n# Eigenvalues have to be inverted to find the smallest eigenvalues of the non-inverted problem.\nλs = 1 ./ λs_inv\n\n# Show that Ax = Bxλ\n@show norm(A * X - B * X * Diagonal(λs)) # 2.8043149027575927e-6"
},

]}
