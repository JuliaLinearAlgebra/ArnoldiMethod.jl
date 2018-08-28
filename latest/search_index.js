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
    "text": "IRAM.jl provides an iterative method to find a few approximate solutions to the  eigenvalue problem in standard form:Ax = xlambdawhere A is a general matrix of size n times n; and x in mathbbC^n and lambda in mathbbC are eigenvectors and eigenvalues respectively. By  general matrix we mean that A has no special structure. It can be symmetric or non-symmetric and either real or complex.The method is matrix-free, meaning that it only requires multiplication with  the matrix A.See Using IRAM.jl on how to use the package."
},

{
    "location": "index.html#What-algorithm-is-IRAM.jl?-1",
    "page": "Home",
    "title": "What algorithm is IRAM.jl?",
    "category": "section",
    "text": "The underlying algorithm is the Implicitly Restarted Arnoldi Method, which be  viewed as a mix between a subspace accelerated version of the power method and  a truncated version of the dense QR algorithm."
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
    "text": "The algorithm is a pure Julia implementation of the implicitly restarted  Arnoldi method and is loosely based on ARPACK. It is not our goal to make an  exact copy of ARPACK. With \"pure Julia\" we mean that we do not rely on LAPACK  for linear algebra routines. This allows us to use any number type. In some  occasions we do rely on BLAS.When this project started, ARPACK was still a dependency of the Julia language,  and the main goal was to get rid of this. Currently ARPACK has moved to a  separate repository called  Arpack.jl, but still it  would be great to have a native Julia implementation of this algorithm."
},

{
    "location": "index.html#Status-1",
    "page": "Home",
    "title": "Status",
    "category": "section",
    "text": "Currently features:An efficient dense QR algorithm natively in Julia, used to do implicit restarts and to compute the low-dimensional dense eigenproblem involving the Hessenberg matrix. It is based on implicit shifts and handles real arithmetic efficiently;\nTransforming converged Ritz values to a partial Schur form.Work in progress:Native transformation of real Schur vectors to eigenvectors.\nUsing a matrix-induced inner product in the case of generalized eigenvalue problems.\nEfficient implementation of symmetric problems with Lanczos."
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
    "text": "The IRAM.jl is intended to find a few approximate solutions to the eigenvalue  problemAx = x lambdaThis problem is handled in two steps:For numerical stability, the method firstly constructs a partial Schur form\nAQ = QR\nwhere Q is orthonormal of size n times textttnev and R is upper  triangular of size textttnev times textttnev In real arithmetic R is quasi upper triangular, with 2 times 2 blocks on the diagonal  corresponding to conjugate complex-valued eigenpairs.\nThe user can transform the partial Schur form into an eigendecomposition via a helper function. The basic math is to determine the eigendecomposition of the upper triangular matrix RY = YLambda such that\nA(QY) = (QY)Lambda\nforms the full eigendecomposition.Step 2 is a cheap post-processing step. Also note that it is not necessary when  the matrix is symmetric, because in that case the Schur decomposition coincides  with the eigendecomposition."
},

{
    "location": "theory.html#Stopping-criterion-1",
    "page": "Theory",
    "title": "Stopping criterion",
    "category": "section",
    "text": "IRAM.jl considers an approximate eigenpair converged when the conditionAx - xlambda_2  texttttollambdais met, where tol is a user provided tolerance. Note that this stopping  criterion is scale-invariant. For a scaled matrix B = alpha A the same  approximate eigenvector together with the scaled eigenvalue alphalambda  would satisfy the stopping criterion."
},

{
    "location": "theory.html#Spectral-transformations-1",
    "page": "Theory",
    "title": "Spectral transformations",
    "category": "section",
    "text": "There are multiple reasons to use a spectral transformation. Firstly, consider the generalized eigenvalue problemAx = BxlambdaThis problem arises for instance in:Finite element discretizations, with B a symmetric, positive definite mass  matrix;\nStability analysis of Navier-Stokes equations, where B is semi-definite  and singular;\nSimple finite differences discretizations where typically B = IBecause IRAM.jl only deals with the standard formCx = xlambdawe have to do a spectral transformation whenever B neq I Secondly, to get fast convergence, one typically applies shift-and-invert  techniques, which also requires a spectral transformation."
},

{
    "location": "theory.html#Transformation-to-standard-form-for-non-singular-B-1",
    "page": "Theory",
    "title": "Transformation to standard form for non-singular B",
    "category": "section",
    "text": "If B is nonsingular and easy to factorize, one can define the matrix C = B^-1A and apply IRAM to the eigenproblemCx = xlambdawhich is in standard form. Of course C should not be formed explicity! One only has to provide the action of the matrix-vector product by implementing LinearAlgebra.mul!(y, C, x). The best way to do so is to factorize B up front.See an example here."
},

{
    "location": "theory.html#Targeting-eigenvalues-with-shift-and-invert-1",
    "page": "Theory",
    "title": "Targeting eigenvalues with shift-and-invert",
    "category": "section",
    "text": "When looking for eigenvalues near a specific target sigma, one can get fast  convergence by using a shift-and-invert technique:Ax = lambda Bx iff (A - sigma B)^-1Bx = theta x text where  theta = frac1lambda - sigmaHere we have casted the generalized eigenvalue problem into standard form with a matrix C = (A - sigma B)^-1B Note that theta is large whenever sigma is close to lambda. This means that we have to target the largest  eigenvalues of C in absolute magnitude. Fast convergence is guaranteed  whenever sigma is close enough to an eigenvalue lambda.Again, one should not construct the matrix C explicitly, but rather implement the matrix-vector product LinearAlgebra.mul!(y, C, x). The best way to do so is to factorize A - sigma B up front.Note that this shift-and-invert strategy simplifies when B = I in which case the matrix in standard form is just C = (A - sigma I)^-1IRAM.jl does not transform the eigenvalues of C back to the eigenvalues of (A B) However, the relation is simply lambda = sigma + theta^-1."
},

{
    "location": "theory.html#Purification-1",
    "page": "Theory",
    "title": "Purification",
    "category": "section",
    "text": "If B is exactly singular or very ill-conditioned, one cannot work with  C = B^-1A. One can however apply the shift-and-invert method. There is a  catch, since C = (A - sigma B)^-1B has eigenvalues close to zero or exactly zero. When transformed back, these values would corrspond to  lambda = infty The process to remove these eigenvalues is called  purification.IRAM.jl does not yet support this purification idea, but it could in principle be put together along the following lines [MLA]:Start the Arnoldi method with C times a random vector, such that the  initial vector has numerically no components in the null space of C.\nExpand the Krylov subspace with one additional vector and add a zero shift. This corresponds to implicitly multiplying the first vector of the Krylov subspace with C at restart.[MLA]: Meerbergen, Karl, and Alastair Spence. \"Implicitly restarted Arnoldi with purification for the shift-invert transformation.\" Mathematics of Computation of the American Mathematical Society 66.218 (1997): 667-689."
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
    "text": "In Julia open the package manager in the REPL via ] and run:(v1.0) pkg> add git@github.com:haampie/IRAM.jl.gitThen use the package.using IRAM"
},

{
    "location": "usage/01_getting_started.html#IRAM.partial_schur",
    "page": "Getting started",
    "title": "IRAM.partial_schur",
    "category": "function",
    "text": "partial_schur(A; nev, which, tol, min, max, restarts) -> PartialSchur, History\n\nFind nev approximate eigenpairs of A with eigenvalues near a specified target.\n\nThe matrix A can be any linear map that implements mul!(y, A, x), eltype and size.\n\nThe method will run iteratively until the eigenpairs are approximated to the prescribed tolerance or until restarts restarts have passed.\n\nArguments\n\nThe most important keyword arguments:\n\nKeyword Type Default Description\nnev Int 6 Number of eigenvalues\nwhich Target LM() One of LM(), LR(), SR(), LI(), SI(), see below.\ntol Real √eps Tolerance for convergence: ‖Ax - xλ‖₂ < tol * ‖λ‖\n\nThe target which can be any of subtypes(IRAM.Target):\n\nTarget Description\nLM() Largest magnitude: abs(λ) is largest\nLR() Largest real part: real(λ) is largest\nSR() Smallest real part: real(λ) is smallest\nLI() Largest imaginary part: imag(λ) is largest\nSI() Smallest imaginary part: imag(λ) is smallest\n\nnote: Note\nThe targets LI() and SI() only make sense in complex arithmetic. In real arithmetic λ is an eigenvalue iff conj(λ) is an eigenvalue and this  conjugate pair converges simultaneously.\n\nReturn values\n\nThe function returns a tuple\n\ndecomp, history = partial_schur(A, ...)\n\nwhere decomp is a PartialSchur struct which forms a partial Schur  decomposition of A to a prescribed tolerance:\n\n> norm(A * decomp.Q - decomp.Q * decomp.R)\n\nhistory is a History struct that holds some basic information about convergence of the method:\n\n> history.converged\ntrue\n> @show history\nConverged after 359 matrix-vector products\n\nAdvanced usage\n\nFurther there are advanced keyword arguments for tuning the algorithm:\n\nKeyword Type Default Description\nmin Int max(10, nev) Minimum Krylov dimension (≥ nev)\nmax Int max(20, 2nev) Maximum Krylov dimension (> min)\nrestarts Int 200 Maximum number of restarts\n\nWhen the algorithm does not converge, one can increase restarts. When the  algorithm converges too slowly, one can play with min and max. It is  suggested to keep min equal to or slightly larger than nev, and max is  usually about two times min.\n\n\n\n\n\n"
},

{
    "location": "usage/01_getting_started.html#Construct-a-partial-Schur-decomposition-1",
    "page": "Getting started",
    "title": "Construct a partial Schur decomposition",
    "category": "section",
    "text": "IRAM.jl exports the partial_schur function which can be used to obtain a  partial Schur decomposition of any matrix A.partial_schur"
},

{
    "location": "usage/01_getting_started.html#From-a-Schur-decomposition-to-an-eigendecomposition-1",
    "page": "Getting started",
    "title": "From a Schur decomposition to an eigendecomposition",
    "category": "section",
    "text": "The eigenvalues and eigenvectors are obtained from the Schur form with the  schur_to_eigen function that is exported by IRAM.jl:λs, X = schur_to_eigen(decomp::PartialSchur)Note that whenever the matrix A is real-symmetric or Hermitian, the partial  Schur decomposition coincides with the partial eigendecomposition, so in that  case there is no need for the transformation."
},

{
    "location": "usage/01_getting_started.html#Example-1",
    "page": "Getting started",
    "title": "Example",
    "category": "section",
    "text": "Here we compute the first ten eigenvalues and eigenvectors of a tridiagonal sparse matrix.julia> using IRAM, LinearAlgebra, SparseArrays\njulia> A = spdiagm(\n           -1 => fill(-1.0, 99),\n            0 => fill(2.0, 100), \n            1 => fill(-1.0, 99)\n       );\njulia> decomp, history = partial_schur(A, nev=10, tol=1e-6, which=SR());\njulia> history\nConverged after 178 matrix-vector products\njulia> norm(A * decomp.Q - decomp.Q * decomp.R)\n3.717314639756976e-8\njulia> λs, X = schur_to_eigen(decomp);\njulia> norm(A * X - X * Diagonal(λs))\n3.7173146389810755e-8"
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
    "text": "IRAM.jl by default only solves the standard-form eigenvalue problem  Ax = xlambda for lambda close to the boundary of the convex hull of  eigenvalues.Whenever one targets eigenvalues close to a specific point in the complex plane, or whenever one solves generalized eigenvalue problems, spectral transformations will enable you to recast the problem into something that IRAM.jl can handle well. In this section we provide some examples."
},

{
    "location": "usage/02_spectral_transformations.html#Shift-and-invert-with-LinearMaps.jl-1",
    "page": "Transformations",
    "title": "Shift-and-invert with LinearMaps.jl",
    "category": "section",
    "text": "To find eigenvalues closest to the origin of A, one can find the eigenvalues of largest magnitude of A^-1. LinearMaps.jl  is a neat way to implement this.using IRAM, LinearAlgebra, LinearMaps\n\n# Define a matrix whose eigenvalues you want\nA = rand(100,100)\n\n# Factorizes A and builds a linear map that applies inv(A) to a vector.\nfunction construct_linear_map(A)\n    F = factorize(A)\n    LinearMap{eltype(A)}((y, x) -> ldiv!(y, F, x), size(A,1), ismutating=true)\nend\n\n# Target the largest eigenvalues of the inverted problem\ndecomp, = partial_schur(construct_linear_map(A), nev=4, tol=1e-5, restarts=100, which=LM())\nλs_inv, X = schur_to_eigen(decomp)\n\n# Eigenvalues have to be inverted to find the smallest eigenvalues of the non-inverted problem.\nλs = 1 ./ λs_inv\n \n# Show that Ax = xλ\n@show norm(A * X - X * Diagonal(λs)) # 7.38473677258669e-6"
},

{
    "location": "usage/02_spectral_transformations.html#generalized_shift_invert-1",
    "page": "Transformations",
    "title": "Smallest eigenvalues of generalized eigenvalue problem",
    "category": "section",
    "text": "When targeting the eigenvalues closest to the origin of a generalized eigenvalue problem Ax = Bxlambda, one can apply the shift-and-invert trick, recasting  the problem to A^-1Bx = xtheta where lambda = 1  theta.using IRAM, LinearAlgebra, LinearMaps\n\n# Define the matrices of the generalized eigenvalue problem\nA, B = rand(100,100), rand(100,100)\n\nstruct ShiftAndInvert{TA,TB,TT}\n    A_lu::TA\n    B::TB\n    temp::TT\nend\n\nfunction (M::ShiftAndInvert)(y,x)\n    mul!(M.temp, M.B, x)\n    ldiv!(y, M.A_lu, M.temp)\nend\n\nfunction construct_linear_map(A,B)\n    a = ShiftAndInvert(factorize(A),B,Vector{eltype(A)}(undef, size(A,1)))\n    LinearMap{eltype(A)}(a, size(A,1), ismutating=true)\nend\n\n# Target the largest eigenvalues of the inverted problem\ndecomp,  = partial_schur(construct_linear_map(A, B), nev=4, tol=1e-5, restarts=100, which=LM())\nλs_inv, X = schur_to_eigen(decomp)\n\n# Eigenvalues have to be inverted to find the smallest eigenvalues of the non-inverted problem.\nλs = 1 ./ λs_inv\n\n# Show that Ax = Bxλ\n@show norm(A * X - B * X * Diagonal(λs)) # 2.8043149027575927e-6"
},

]}
