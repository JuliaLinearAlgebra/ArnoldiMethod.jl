"""
```julia
partial_schur(A; nev, which, tol, min, max, restarts) -> PartialSchur, History
```

Find `nev` approximate eigenpairs of `A` with eigenvalues near a specified target.

The matrix `A` can be any linear map that implements `mul!(y, A, x)`, `eltype`
and `size`.

The method will run iteratively until the eigenpairs are approximated to
the prescribed tolerance or until `restarts` restarts have passed.

## Arguments

The most important keyword arguments:

| Keyword | Type | Default | Description |
|------:|:-----|:----|:------|
| `nev` | `Int` | `6` |Number of eigenvalues |
| `which` | `Target` | `LM()` | One of `LM()`, `LR()`, `SR()`, `LI()`, `SI()`, see below. |
| `tol` | `Real` | `√eps` | Tolerance for convergence: ‖Ax - xλ‖₂ < tol * ‖λ‖ |

The target `which` can be any of `subtypes(ArnoldiMethod.Target)`:

| Target | Description |
|------:|:-----|
| `LM()` | Largest magnitude: `abs(λ)` is largest |
| `LR()` | Largest real part: `real(λ)` is largest |
| `SR()` | Smallest real part: `real(λ)` is smallest |
| `LI()` | Largest imaginary part: `imag(λ)` is largest|
| `SI()` | Smallest imaginary part: `imag(λ)` is smallest|

!!! note

    The targets `LI()` and `SI()` only make sense in complex arithmetic. In real
    arithmetic `λ` is an eigenvalue iff `conj(λ)` is an eigenvalue and this 
    conjugate pair converges simultaneously.

## Return values

The function returns a tuple

```julia
decomp, history = partial_schur(A, ...)
```

where `decomp` is a `PartialSchur` struct which forms a partial Schur 
decomposition of `A` to a prescribed tolerance:

```julia
> norm(A * decomp.Q - decomp.Q * decomp.R)
```

`history` is a `History` struct that holds some basic information about
convergence of the method:

```julia
> history.converged
true
> @show history
Converged after 359 matrix-vector products
```

## Advanced usage

Further there are advanced keyword arguments for tuning the algorithm:

| Keyword | Type | Default | Description |
|------:|:-----|:---|:------|
| `min` | `Int` | `max(10, nev)` | Minimum Krylov dimension (≥ nev) |
| `max` | `Int` | `max(20, 2nev)` | Maximum Krylov dimension (> min) |
| `restarts` | `Int` | `200` | Maximum number of restarts |

When the algorithm does not converge, one can increase `restarts`. When the 
algorithm converges too slowly, one can play with `min` and `max`. It is 
suggested to keep `min` equal to or slightly larger than `nev`, and `max` is 
usually about two times min.

"""
partial_schur(A; nev::Int = 6,
                 which::Target = LM(),
                 tol::Real = sqrt(eps(real(eltype(A)))), 
                 min::Int = max(10, nev),
                 max::Int = max(20, 2nev),
                 restarts::Int = 200) =
    _partial_schur(A, eltype(A), min, max, nev, tol, restarts, which)

"""
    IsConverged(ritz, tol)

Functor to test whether Ritz values satisfy the convergence criterion. Current
convergence condition is ‖Ax - xλ‖₂ < tol * |λ|. This is supposed to be scale
invariant: the matrix `B = αA` for some constant `α` has the same eigenvectors
with eigenvalue λα, so this scaling with `α` cancels in the inequality.
"""
struct IsConverged{RV<:RitzValues,T}
    ritz::RV
    tol::T
end

function (r::IsConverged)(i::Integer)
    @inbounds begin
        idx = r.ritz.ord[i]
        return r.ritz.rs[idx] < r.tol * abs(r.ritz.λs[idx])
    end
end

struct History
    mvproducts::Int
    converged::Bool
end

function _partial_schur(A, ::Type{T}, mindim::Int, maxdim::Int, nev::Int, tol::Ttol, restarts::Int, which::Target) where {T,Ttol<:Real}
    n = size(A, 1)

    # Pre-allocated Arnoldi decomp
    arnoldi = Arnoldi{T}(n, maxdim)

    # Approximate residual norms for all Ritz values, and Ritz values
    ritz = RitzValues{T}(maxdim)
    isconverged = IsConverged(ritz, tol)

    # Some temporaries
    Vtmp = Matrix{T}(undef, n, maxdim)
    Htmp = Matrix{T}(undef, maxdim + 1, maxdim)
    Qtmp = Matrix{T}(undef, maxdim + 1, maxdim + 1)

    # Initialize an Arnoldi relation of size `min`
    reinitialize!(arnoldi)
    iterate_arnoldi!(A, arnoldi, 1:mindim)

    # First index of non-locked basis vector in V.
    # This just means it is the first index for which H[active + 1, active] != 0
    active = 1

    # Number of converged eigenvalues (not necessarily deflated!)
    converged = 0

    # Bookkeeping for number of mv-products
    prods = mindim

    # Effective smallest size of the Arnoldi decomp.
    k = mindim

    # Ordering used in sort!
    ordering = get_order(ritz, which)

    for iter = 1 : restarts

        # Expand Krylov subspace dimension from `k` to `max`.
        iterate_arnoldi!(A, arnoldi, k+1:maxdim)
        
        # Bookkeeping
        prods += length(k+1:maxdim)

        # Compute the Ritz values and residuals
        # E.g. we compute the eigenvalues of H[active:max,active:max]
        H_active = view(Htmp, active:maxdim, active:maxdim)
        Q_active = view(Qtmp, active:maxdim, active:maxdim)
        copyto!(H_active, view(arnoldi.H, active:maxdim, active:maxdim))
        copyto!(Q_active, I)

        # Construct Schur decomp of inplace
        local_schurfact!(H_active, Q_active)
        
        # Update the Ritz values
        indices = view(ritz.ord, active:maxdim)
        copy_eigenvalues!(view(ritz.λs, active:maxdim), H_active)
        copy_residuals!(view(ritz.rs, active:maxdim), H_active, Q_active, @inbounds arnoldi.H[maxdim+1,maxdim])
        copyto!(indices, active:maxdim)

        # Partition the Ritz values in converged & not converged
        # We never shift a converged Ritz value because the Arnoldi relation might lose
        # as many digits as the converged Ritz value had (there's probably theory on this,
        # but this is what we observed)
        # Note that this means we might have converged Ritz values we don't want;
        # currently we do not remove these converged but unwanted Ritz values and vectors.
        first_not_converged = partition!(isconverged, indices)

        # This never happens, but now the compiler knows that as well
        first_not_converged === nothing && break

        # Total number of converged Ritz values
        converged = (active - 1) + (first_not_converged - 1)

        # Ritz values are converged, but not all of them are deflated, so we still
        # have to bring the Hessenberg matrix to upper triangular form.
        # For now just shift away those Ritz values that have not converged
        # and then act like H[converged+1,converged] = 0, so that V[:,1:converged]
        # spans an invariant subspace for A.
        if converged ≥ nev
            implicit_restart!(arnoldi, Vtmp, ritz, converged, maxdim, active)
            transform_converged!(arnoldi, active, converged, Vtmp)
            hist = History(prods, true)
            schur = PartialSchur(view(arnoldi.V, :, 1:converged), view(arnoldi.H, 1:converged, 1:converged))
            return schur, hist
        end

        # We will reduce the the size of the Krylov subspace from `max` to `k`
        # and in the special case of a conjugate pair sometimes to `k+1`
        # We allow `k` to be larger than `mindim` whenever Ritz values have converged;
        # It's basically heuristics, but once one eigenvector is converged, the effective
        # size of the Krylov subspace can be seen as one less, so the quality of the 
        # subspace might be worse. So we compensate by keeping an effective Krylov subspace 
        # of `mindim` excluding converged eigenvectors.
        # However, we must also keep some room for improving the subspace, so in the end
        # we don't allow the minimum dimension to grow beyond halfway `mindim` and `maxdim`.
        k = min(mindim + converged, (mindim + maxdim) ÷ 2)
        
        # Now determine `maxdim - k` exact shifts.
        # TODO: worry about the order of the exact shifts -- maybe there is value in
        # a particular order such as from worst converged to best converged. Would not be
        # surprised if ARPACK did this.
        sort!(ritz.ord, converged + 1, maxdim, MergeSort, ordering)

        # Shrink the subspace. Note that implicit_restart! returns the effective size of
        # the shrunken Krylov subspace. In complex arithmetic it will always be the old `k`
        # but in real arithmetic a conjugate pair make `k ← k + 1`.
        k = implicit_restart!(arnoldi, Vtmp, ritz, k, maxdim, active)
        
        # Check whether some off-diagonal value is small enough and if so, bring the new 
        # locked part of H into upper triangular form.
        new_active = max(active, detect_convergence!(arnoldi.H, tol)) # max is superfluous here...
        transform_converged!(arnoldi, active, new_active-1, Vtmp)
        active = new_active

        active > nev && break
    end

    schur = PartialSchur(view(arnoldi.V, :, 1:converged), view(arnoldi.H, 1:converged, 1:converged))
    hist = History(prods, false)
    return schur, hist
end

"""
    update_residual_norms!(rs, H, Q, hₖ₊₁ₖ) -> rs

Computes the Ritz residuals ‖Ax - λx‖₂ = |yₖ| * |hₖ₊₁ₖ| for each eigenvalue
"""
function copy_residuals!(rs::AbstractVector{T}, H, Q, hₖ₊₁ₖ) where {T<:Real}
    m = size(H, 1)
    x = zeros(complex(T), m)
    @inbounds for i = 1:m
        fill!(x, zero(T))
        len = collect_eigen!(x, H, i)
        tmp = zero(complex(T))
        for j = 1 : len
            tmp += Q[m, j] * x[j]
        end
        rs[i] = abs(tmp * hₖ₊₁ₖ)
    end

    rs
end

"""
    transform_converged!(arnoldi, from, to, Vtmp) -> nothing

Whenever we have found an invariant subspace V[:, 1:to], we want to bring V[:, 1:to]
and H[1:to, 1:to] to partial Schur form, in the sense that H[1:to,1:to] is upper triangular
and A * V[:, 1:to] = V[:, 1:to] * H[1:to,1:to].

In this function we assume (V[:, 1:from-1], H[1:from-1,1:from-1]) is already in partial
Schur form, and we only have to touch V[:, from:to] and the blocks H[from:to, from:to],
H[1:from-1,from:to] and H[from:to,to+1:end].

If only one vector has converged (i.e. from == to), then we don't have to do any work!
"""
function transform_converged!(arnoldi::Arnoldi{T}, from::Int, to::Int, Vtmp) where {T}

    # Nothing to transform
    to ≤ from && return nothing
    
    # H = Q R Q'

    # A V = V H
    # A V = V Q R Q'
    # A (V Q) = (V Q) R
    
    # V <- V Q
    # H_right <- Q' H_right
    # H_lock <- Q' H_lock Q
    # H_above <- H_above Q

    Q_large = Matrix{T}(I, to, to)
    Q_small = view(Q_large, from:to, from:to)
    V_locked = view(arnoldi.V, :, from:to)

    local_schurfact!(arnoldi.H, from, to, Q_large)
    mul!(view(Vtmp, :, from:to), V_locked, Q_small)
    copyto!(V_locked, view(Vtmp, :, from:to))

    return nothing
end

