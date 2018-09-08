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

(r::IsConverged{RV,T})(i::Integer) where {RV,T} =
    @inbounds return r.ritz.rs[i] < max(eps(T) * r.H_frob_norm[], r.tol * abs(r.ritz.λs[i]))

"""
History gives some general information about how much matrix-vector product were necessary
to converged, plus the number of converged eigenvalues.
"""
struct History
    mvproducts::Int
    nconverged::Int
    converged::Bool
    nev::Int
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
    groups = zeros(Int, maxdim)

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
        # TODO: only compute residual of *active* eigenvalues
        copyto!(ritz.ord, 1:maxdim)
        copy_eigenvalues!(ritz.λs, H)
        copy_residuals!(ritz.rs, H, Q, H[maxdim+1,maxdim])

        # Sort the Ritz values from most wanted to least wanted in the active part of the
        # factorization.
        sort!(ritz.ord, active, maxdim, QuickSort, ordering)

        # Compute the Frobenius norm of H for the stopping criterion
        isconverged.H_frob_norm[] = norm(view(arnoldi.H, 1:maxdim, 1:maxdim))

        ### PARTITIONING OF SCHUR FORM IN [LOCKED | RETAINED | PURGED]

        # Plan de campagne: reorder the permutation `ritz.ord` such that
        # ritz.ord[1:nlock] gives the indices of the locked ritz values
        # ritz.ord[nlock+1:k] is the indices of the ritz values we wanna retain
        # ritz.ord[k+1:maxdim] is the indices of the ritz values we wanna truncate
        # Then we make ritz.groups[i] = {1,2,3} by iterating over them.

        # We keep at most `nev` or `nev+1` eigenvalues, depending on the split being
        # halfway a conjugate pair.
        effective_nev = include_conjugate_pair(T, ritz, nev)

        # Partition in converged & not converged.
        first_not_conv_idx = partition!(isconverged, ritz.ord, active:effective_nev)

        # Now ritz.ord[1:nlock] are converged eigenvalues that we want to lock, and 
        # nlock ≤ effective_nev, so it's really just these that we are after!
        nlock = first_not_conv_idx === nothing ? effective_nev : first_not_conv_idx - 1

        # Next, purge the converged eigenvalues we do not want by moving them to the back.
        partition!(i -> !isconverged(i), ritz.ord, nlock+1:maxdim)

        # Determine the new length `k` of the truncated Krylov subspace:
        # 1. The dimension of the active part should be roughly `mindim`; so `k` will be 
        #    larger than `mindim` when converged Ritz vectors have been locked.
        # 2. But `k` can't be so large that the expansion would barely give new information;
        #    hence we meet in the middle: `k` is at most halfway `mindim` and `maxdim`.
        # 3. If `k` ends up on the boundary of a conjugate pair, we increase `k` by 1.
        k = include_conjugate_pair(T, ritz, min(nlock + mindim, (mindim + maxdim) ÷ 2))

        # Locked ritz values
        @inbounds for i = 1:nlock
            groups[ritz.ord[i]] = 1
        end

        # Retained ritz values
        @inbounds for i = nlock+1:k
            groups[ritz.ord[i]] = 2
        end

        # Truncated ritz values
        @inbounds for i = k+1:maxdim
            groups[ritz.ord[i]] = 3
        end

        partition_schur_three_way!(H, Q, groups)

        # Restore the Hessenberg matrix via Householder reflections.
        # Note that we restore the new active part only -- Q[end, 1:nlock] is small enough
        # by convergence criterion.
        restore_hessenberg!(H, nlock + 1, k, Q)

        # Finally do the change of basis to get the length `k` Arnoldi relation.
        @views mul!(Vtmp[:,active:k], V[:,active:maxdim], Q[active:maxdim,active:k])
        @views copyto!(V[:,active:k], Vtmp[:,active:k])
        @views copyto!(V[:,k+1], V[:,maxdim+1])

        # The active part 
        active = nlock + 1

        @views nlock ≥ nev && return PartialSchur(V[:, 1:nlock], H[1:nlock,1:nlock]), History(prods, nlock, true, nev)
    end

    @views return PartialSchur(V[:, 1:active-1], H[1:active-1,1:active-1]), History(prods, active-1, false, nev)
end

function partition_schur_three_way!(R, Q, groups::AbstractVector{Int})
    # Partitioning goes like this:
    # |2|33|1|2|33|2|1|
    #  h
    #  m
    #  l

    # |2|33|1|2|33|2|1|  -> |2|33|1|2|33|2|1| 
    #    h               ->       h             
    #  m                 ->    m             
    #  l                 ->  l               

    # |2|33|1|2|33|2|1|  -> |1|2|33|2|33|2|1| 
    #       h            ->         h              
    #    m               ->      m
    #  l                 ->    l                   
    
    # |1|2|33|2|33|2|1|  -> |1|2|2|33|33|2|1| 
    #         h          ->           h        
    #      m             ->        m         
    #    l               ->    l             

    # |1|2|2|33|33|2|1|  -> |1|2|2|33|33|2|1| 
    #           h        ->              h              
    #        m           ->        m          
    #    l               ->    l                   

    # |1|2|2|33|33|2|1|  -> |1|2|2|2|33|33|1| 
    #              h     ->                h   
    #        m           ->          m       
    #    l               ->    l             
        
    
    # |1|2|2|2|33|33|1|  -> |1|1|2|2|2|33|33|
    #                h   ->                  h             
    #          m         ->            m       
    #    l               ->      l                  
     

    hi = 1
    mi = 1
    lo = 1

    while hi ≤ length(groups)
        group = groups[hi]
        blocksize = is_start_of_11_block(R, hi) ? 1 : 2

        if group == 3
            hi += blocksize
        elseif group == 2
            rotate_right!(R, mi, hi, Q)
            hi += blocksize
            mi += blocksize
        else # group == 1
            rotate_right!(R, lo, hi, Q)
            hi += blocksize
            mi += blocksize
            lo += blocksize
        end
    end

    nothing
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
