using LinearAlgebra: checksquare

"""
    vtype(A) -> T

Do some arithmetic to get a proper number type that a matrix A should operate
on.
"""
function vtype(A)
    T = eltype(A)
    typeof(zero(T)/sqrt(one(T)))
end

"""
```julia
partialschur(A; nev, which, tol, mindim, maxdim, restarts) -> PartialSchur, History
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
| `nev` | `Int` | `min(6, size(A, 1))` |Number of eigenvalues |
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
decomp, history = partialschur(A, ...)
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
| `mindim` | `Int` | `min(max(10, nev), size(A,1))` | Minimum Krylov dimension (≥ nev) |
| `maxdim` | `Int` | `min(max(20, 2nev), size(A,1))` | Maximum Krylov dimension (≥ min) |
| `restarts` | `Int` | `200` | Maximum number of restarts |

When the algorithm does not converge, one can increase `restarts`. When the 
algorithm converges too slowly, one can play with `mindim` and `maxdim`. It is 
suggested to keep `mindim` equal to or slightly larger than `nev`, and `maxdim`
is usually about two times `mindim`.

"""
function partialschur(A;
                       nev::Int = min(6, size(A, 1)),
                       which::Target = LM(),
                       tol::Real = sqrt(eps(real(vtype(A)))), 
                       mindim::Int = min(max(10, nev), size(A, 1)),
                       maxdim::Int = min(max(20, 2nev), size(A, 1)),
                       restarts::Int = 200)
    s = checksquare(A)
    nev ≤ mindim ≤ maxdim ≤ s || throw(ArgumentError("nev ≤ mindim ≤ maxdim does not hold, got $nev ≤ $mindim ≤ $maxdim"))
    _partialschur(A, vtype(A), mindim, maxdim, nev, tol, restarts, which)
end

"""
    IsConverged(ritz, tol)

Functor to test whether Ritz values satisfy the convergence criterion. Current
convergence condition is ‖Ax - xλ‖₂ < max(ε‖H‖, tol * |λ|). This is supposed to
be scale invariant: the matrix `B = αA` for some constant `α` has the same 
eigenvectors with eigenvalue λα, so this scaling with `α` cancels in the 
inequality.
"""
struct IsConverged{RV<:RitzValues,T}
    ritz::RV
    tol::T
    H_frob_norm::RefValue{T}

    IsConverged(ritz::R, tol::T) where {R<:RitzValues,T} = new{R,T}(ritz, tol, RefValue(zero(T)))
end

function (r::IsConverged{RV,T})(i::Integer) where {RV,T}
    @inbounds begin
        idx = r.ritz.ord[i]
        return r.ritz.rs[idx] < max(eps(T) * r.H_frob_norm[], r.tol * abs(r.ritz.λs[idx]))
    end
end

"""
History gives some general information about how much matrix-vector product were necessary
to converged, plus the number of converged eigenvalues.
"""
struct History
    mvproducts::Int
    converged::Bool
end

function _partialschur(A, ::Type{T}, mindim::Int, maxdim::Int, nev::Int, tol::Ttol, restarts::Int, which::Target) where {T,Ttol<:Real}
    n = size(A, 1)

    # Pre-allocated Arnoldi decomp
    arnoldi = Arnoldi{T}(n, maxdim)

    # Unpack for convenience
    H = arnoldi.H
    V = arnoldi.V

    # For a change of basis we have Vtmp as working space
    Vtmp = Matrix{T}(undef, n, maxdim)

    # Unitary matrix used for change of basis of V.
    Q = Matrix{T}(undef, maxdim, maxdim)

    # Approximate residual norms for all Ritz values, and Ritz values
    ritz = RitzValues{T}(maxdim)
    isconverged = IsConverged(ritz, tol)
    ordering = get_order(ritz, which)

    # V[:,1:active-1   ] ⸺ Locked vectors / approximately invariant subspace
    # V[:,active:maxdim] ⸺ Active part of decomposition.
    # V[:,active:k     ] ⸺ Smallest size of Arnoldi relation; k will increase over time.
    #                        We keep length(active:k) ≈ mindim, but we should also retain
    #                        space to add new directions to the Krylov subspace, so 
    #                        length(k+1:maxdim) should not be too small either.
    active = 1
    k = mindim
    effective_nev = nev

    # Bookkeeping for number of mv-products
    prods = mindim

    # Initialize an Arnoldi relation of size `mindim`
    reinitialize!(arnoldi)
    iterate_arnoldi!(A, arnoldi, 1:mindim)

    for iter = 1 : restarts

        # Expand Krylov subspace dimension from `k` to `maxdim`.
        iterate_arnoldi!(A, arnoldi, k+1:maxdim)
        
        # Bookkeeping
        prods += length(k+1:maxdim)

        # Q accumulates the changes of basis via relfectors; initially it's just I.
        copyto!(Q, one(T) * I)

        # Construct Schur decomposition of H[active:maxdim,active:maxdim] in-place
        local_schurfact!(view(H, 1:maxdim, 1:maxdim), active, maxdim, Q)
        
        # Update the Ritz values
        copyto!(view(ritz.ord, active:maxdim), active:maxdim)
        copy_eigenvalues!(ritz.λs, H)
        copy_residuals!(ritz.rs, H, Q, H[maxdim+1,maxdim])

        # Sort the Ritz values from most wanted to least wanted in the active part of the
        # factorization. TODO: use quicksort and let ordering induce stability by itself.
        sort!(ritz.ord, active, maxdim, MergeSort, ordering)

        # Compute the Frobenius norm of H for the stopping criterion
        isconverged.H_frob_norm[] = norm(view(arnoldi.H, 1:maxdim, 1:maxdim))

        ### LOCKING: fixing the converged Ritz values in the front of H

        # From the potentialy converged Ritz values in active:nev+{0,1} we partition 
        # them in converged and not converged. Note: we partition the permutation.
        potentials = view(ritz.ord, active:effective_nev)
        effective_nev = include_conjugate_pair(T, ritz, nev)
        first_not_converged = partition!(isconverged, potentials)
        
        # Count how many have converged
        nconv_this_iteration = first_not_converged === nothing ? length(active:effective_nev) : first_not_converged - 1

        # Rotate the converged Ritz values to first entries of the diagonal of H
        lock!(H, Q, active, view(potentials, 1:nconv_this_iteration))

        # Total number of converged Ritz values
        total_converged = active - 1 + nconv_this_iteration

        # We are done :)
        if total_converged ≥ nev 
            # But still do the change of basis.
            @views mul!(Vtmp[:,active:total_converged], V[:,active:maxdim], Q[active:maxdim,active:total_converged])
            @views copyto!(V[:,active:total_converged], Vtmp[:,active:total_converged])
            return PartialSchur(V[:,1:total_converged], H[1:total_converged,1:total_converged]), History(prods, true)
        end


        # We have locked `first_not_converged - 1` Ritz vectors, so the active factorization
        # active:maxdim shrinks in length.
        new_active = active + first_not_converged - 1

        ### RESTART: truncation of the unwanted Ritz values

        # Determine the new length `k` of the truncated Krylov subspace:
        # 1. The dimension of the active part should be roughly `mindim`; so `k` will be 
        #    larger than `mindim` when converged Ritz vectors have been locked.
        # 2. But `k` can't be so large that the expansion would barely give new information;
        #    hence we meet in the middle: `k` is at most halfway `mindim` and `maxdim`.
        # 3. If `k` ends up on the boundary of a conjugate pair, we increase `k` by 1.
        k = include_conjugate_pair(T, ritz, min(mindim + new_active - 1, (mindim + maxdim) ÷ 2))

        # Rotate the unwantend Ritz values to the end of the diagonal of H
        truncate!(H, Q, maxdim, view(ritz.ord, k+1:maxdim))

        # Restore the Hessenberg matrix via Householder reflections.
        restore_hessenberg!(H, new_active, k, Q)

        # Finally do the change of basis to get the length `k` Arnoldi relation.
        @views mul!(Vtmp[:,active:k], V[:,active:maxdim], Q[active:maxdim,active:k])
        @views copyto!(V[:,active:k], Vtmp[:,active:k])
        @views copyto!(V[:,k+1], V[:,maxdim+1])

        active = new_active
    end

    return PartialSchur(V, H), History(prods, false)
end

"""
    include_conjugate_pair(T, ritz, i) → {i, i + 1}

Returns i or i + 1 depending on whether Ritz value i and i + 1 form a conjugate pair
together
"""
@inline function include_conjugate_pair(::Type{<:Real}, ritz::RitzValues, i)
    @inbounds λ1 = ritz.λs[ritz.ord[i+0]]
    @inbounds λ2 = ritz.λs[ritz.ord[i+1]]
    return imag(λ1) != 0 && λ1' == λ2 ? i + 1 : i
end

@inline include_conjugate_pair(::Type{T}, ritz::RitzValues, i) where {T} = i

@inline function lock!(R, Q, from::Integer, list::AbstractVector{<:Integer})
    sort!(list, QuickSort, Base.Order.Forward)
    @inbounds for idx in list
        rotate_right!(R, from, idx, Q)
        from += 1
    end

    nothing
end

@inline function lock!(R::AbstractMatrix{<:Real}, Q, from::Integer, list::AbstractVector{<:Integer})
    # We assume `from` points to the start of a block
    sort!(list, MergeSort, Base.Order.Forward)
    i = 1
    @inbounds while i ≤ length(list)
        idx = list[i]
        is11 = is_start_of_11_block(R, idx)
        rotate_right!(R, from, idx, Q)
        if is11
            from += 1
            i += 1
        else
            from += 2
            i += 2
        end
    end
end

"""
    truncate!(R, Q, to, list) → nothing

Rotate eigenvalues occuring in `list` to the end of the Schur decomp
"""
function truncate!(R, Q, to::Integer, list::AbstractVector{<:Integer})
    sort!(list, QuickSort, Base.Order.Backward)
    @inbounds for idx in list
        rotate_left!(R, idx, to, Q)
        to -= 1
    end

    nothing
end

# Real case, some edge cases with conjugate pairs etc etc
function truncate!(R::AbstractMatrix{<:Real}, Q, to::Integer, list::AbstractVector{<:Integer})
    # VERY convoluted, should really be refactored
    sort!(list, MergeSort, Base.Order.Forward)
    i = length(list)
    @inbounds while i > 0
        idx = list[i]
        isfrom11 = is_end_of_11_block(R, idx)
        isto11 = is_end_of_11_block(R, to)
        rotate_left!(R, isfrom11 ? idx : idx - 1, isto11 ? to : to - 1, Q)
        if isfrom11
            to -= 1
            i -= 1
        else
            to -= 2
            i -= 2
        end
    end

    nothing
end

"""
    update_residual_norms!(rs, H, Q, hₖ₊₁ₖ) → rs

Computes the Ritz residuals ‖Ax - λx‖₂ = |yₖ| * |hₖ₊₁ₖ| for each eigenvalue
"""
function copy_residuals!(rs::AbstractVector{T}, H, Q, hₖ₊₁ₖ) where {T<:Real}
    m = size(H, 2)
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
