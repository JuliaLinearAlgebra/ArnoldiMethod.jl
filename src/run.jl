using LinearAlgebra: checksquare

"""
    vtype(A) → T

Do some arithmetic to get a proper number type that a matrix A should operate
on.
"""
function vtype(A)
    T = eltype(A)
    typeof(zero(T) / sqrt(one(T)))
end

"""
```julia
partialschur(A; v1, workspace, nev, which, tol, mindim, maxdim, restarts) → PartialSchur, History
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
| `which` | `Symbol` or `Target` | `:LM` | One of `:LM`, `:LR`, `:SR`, `:LI`, `:SI`, see below. |
| `tol` | `Real` | `√eps` | Tolerance for convergence: ‖Ax - xλ‖₂ < tol * ‖λ‖ |
| `v1` | `AbstractVector` | `nothing` | Optional starting vector for the Krylov subspace |

Regarding the initial vector `v1`: it will not be mutated, and it does not have to be normalized.

The target `which` can be any of:

| Target          | Description                                    |
|----------------:|:-----------------------------------------------|
| `:LM` or `LM()` | Largest magnitude: `abs(λ)` is largest         |
| `:LR` or `LR()` | Largest real part: `real(λ)` is largest        |
| `:SR` or `SR()` | Smallest real part: `real(λ)` is smallest      |
| `:LI` or `LI()` | Largest imaginary part: `imag(λ)` is largest   |
| `:SI` or `SI()` | Smallest imaginary part: `imag(λ)` is smallest |

!!! note

    The targets `:LI` and `:SI` only make sense in complex arithmetic. In real
    arithmetic `λ` is an eigenvalue iff `conj(λ)` is an eigenvalue and this 
    conjugate pair converges simultaneously.

## Return values

The function returns a tuple

```julia
decomp, history = partialschur(A, ...)
```

where `decomp` is a [`PartialSchur`](@ref) struct which 
forms a partial Schur decomposition of `A` to a prescribed tolerance:

```julia
> norm(A * decomp.Q - decomp.Q * decomp.R)
```

`history` is a [`History`](@ref) struct that holds some basic information about
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
function partialschur(
    A;
    v1::Union{AbstractVector,Nothing} = nothing,
    nev::Int = min(6, size(A, 1)),
    which::Union{Target,Symbol} = LM(),
    tol::Real = sqrt(eps(real(vtype(A)))),
    mindim::Int = min(max(10, nev), size(A, 1)),
    maxdim::Int = min(max(20, 2nev), size(A, 1)),
    restarts::Int = 200,
)
    s = checksquare(A)
    nev < 1 && throw(ArgumentError("nev cannot be less than 1"))
    nev ≤ mindim ≤ maxdim ≤ s || throw(
        ArgumentError(
            "nev ≤ mindim ≤ maxdim ≤ size(A, 1) does not hold, got $nev ≤ $mindim ≤ $maxdim ≤ $s",
        ),
    )
    _which = which isa Target ? which : _symbol_to_target(which)

    if v1 === nothing
        arnoldi = ArnoldiWorkspace(vtype(A), size(A, 1), maxdim)
        reinitialize!(arnoldi, 0, rand!)
    else
        length(v1) == size(A, 1) ||
            throw(ArgumentError("v1 should have the same dimension as A"))
        arnoldi = ArnoldiWorkspace(v1, maxdim)
        reinitialize!(arnoldi, 0, v -> copyto!(v, v1))
    end
    _partialschur(A, arnoldi, mindim, maxdim, nev, tol, restarts, _which)
end

"""
```julia
partialschur!(A, arnoldi; start_from, initialize, nev, which, tol, mindim, maxdim, restarts) → PartialSchur, History
```

This is a variant of [`partialschur`](@ref) that operates on a pre-allocated
[`ArnoldiWorkspace`](@ref) instance. This is useful in the following cases:

1. To provide an initial partial Schur decomposition. In that case, set `start_from` to the
   number of Schur vectors, and `partialschur` will continue from there. Notice that if you have
   only one Schur vector, it can be simpler to pass it as `v1` instead.
2. To provide a custom array type for the basis of the Krylov subspace, the Hessenberg matrix, and
   some temporaries.
3. To avoid allocations when calling `partialschur` repeatedly.

You can also provide the initial vector that induces the Krylov subspace in the first column
arnoldi.V[:, 1]. If you do that, set `initialize` explicitly to `false`.

Upon return, note that the PartialSchur struct will contain views of arnoldi.V and arnoldi.H, no
copies are made.
"""
function partialschur!(
    A,
    arnoldi::ArnoldiWorkspace{T};
    start_from::Int = 1,
    initialize::Bool = start_from == 1,
    nev::Int = min(6, size(A, 1)),
    which::Union{Target,Symbol} = LM(),
    tol::Real = sqrt(eps(real(vtype(A)))),
    mindim::Int = min(max(10, nev), size(A, 1), size(arnoldi.V, 2) - 1),
    maxdim::Int = min(max(20, 2nev), size(A, 1), size(arnoldi.V, 2) - 1),
    restarts::Int = 200,
) where {T}
    s = checksquare(A)
    nev < 1 && throw(ArgumentError("nev cannot be less than 1"))
    nev ≤ mindim ≤ maxdim ≤ s || throw(
        ArgumentError(
            "nev ≤ mindim ≤ maxdim ≤ size(A, 1) does not hold, got $nev ≤ $mindim ≤ $maxdim ≤ $s",
        ),
    )
    maxdim < size(arnoldi.V, 2) ||
        throw(ArgumentError("maxdim should be strictly less than size(arnoldi.V, 2)"))
    1 ≤ start_from ≤ maxdim ||
        throw(ArgumentError("start_from should be between 1 and maxdim"))
    _which = which isa Target ? which : _symbol_to_target(which)
    fill!(view(arnoldi.H, :, start_from:size(arnoldi.H, 2)), zero(T))
    initialize && reinitialize!(arnoldi, start_from - 1)
    _partialschur(A, arnoldi, mindim, maxdim, nev, tol, restarts, _which, start_from)
end

_symbol_to_target(sym::Symbol) =
    sym == :LM ? LM() :
    sym == :LR ? LR() :
    sym == :SR ? SR() :
    sym == :LI ? LI() : sym == :SI ? SI() : throw(ArgumentError("Unknown target: $sym"))


"""
    IsConverged(ritz, tol)

Functor to test whether Ritz values satisfy the convergence criterion. Current
convergence condition is ‖Ax - xλ‖₂ < max(ε‖H‖, tol * |λ|). This is supposed to
be scale invariant: the matrix `B = αA` for some constant `α` has the same 
eigenvectors with eigenvalue λα, so this scaling with `α` cancels in the 
inequality.
"""
struct IsConverged{Tv,Tr,Tt}
    ritz::RitzValues{Tv,Tr}
    tol::Tt
    H_frob_norm::RefValue{Tr}

    IsConverged(ritz::RitzValues{Tv,Tr}, tol::Tt) where {Tv,Tr,Tt} =
        new{Tv,Tr,Tt}(ritz, tol, RefValue(zero(Tr)))
end

function (r::IsConverged{Tv,Tr})(i::Integer) where {Tv,Tr}
    @inbounds r.ritz.rs[i] <= max(eps(Tr) * r.H_frob_norm[], r.tol * abs(r.ritz.λs[i]))
end


"""
    History(mvproducts, nconverged, converged, nev)

History shows whether the method has converged (when `nconverged` ≥ `nev`) and
how many matrix-vector products were necessary to do so.
"""
struct History
    mvproducts::Int
    nconverged::Int
    converged::Bool
    nev::Int
end

function _partialschur(
    A,
    arnoldi::ArnoldiWorkspace{T},
    mindim::Int,
    maxdim::Int,
    nev::Int,
    tol::Ttol,
    restarts::Int,
    which::Target,
    active::Int = 1, # when > 1, assumes we already have a converged schur form
) where {T,Ttol<:Real}
    # Unpack for convenience
    H = arnoldi.H
    V = arnoldi.V
    V_tmp = arnoldi.V_tmp
    Q = arnoldi.Q

    # We only need to store one eigenvector of the Hessenberg matrix.
    x = zeros(complex(T), maxdim)

    # And we store the reflector to transform H back to Hessenberg separately
    # TODO: can be in-place in H now.
    G = Reflector{T}(maxdim)

    # Approximate residual norms for all Ritz values, and Ritz values
    ritz = RitzValues{T}(maxdim)
    isconverged = IsConverged(ritz, tol)
    ordering = get_order(which)
    groups = zeros(Int, maxdim)

    # V[:,1:active-1   ] ⸺ Locked vectors / approximately invariant subspace
    # V[:,active:maxdim] ⸺ Active part of decomposition.
    # V[:,active:k     ] ⸺ Smallest size of Arnoldi relation; k will increase over time.
    #                        We keep length(active:k) ≈ mindim, but we should also retain
    #                        space to add new directions to the Krylov subspace, so 
    #                        length(k+1:maxdim) should not be too small either.
    k = mindim
    effective_nev = nev

    # Bookkeeping for number of mv-products
    prods = length(active:mindim)

    # Initialize an Arnoldi relation of size `mindim`
    iterate_arnoldi!(A, arnoldi, active:mindim)

    for iter = 1:restarts

        # Expand Krylov subspace dimension from `k` to `maxdim`.
        iterate_arnoldi!(A, arnoldi, k+1:maxdim)

        # Bookkeeping
        prods += length(k+1:maxdim)

        # Q accumulates the changes of basis via relfectors; initially it's just I.
        copyto!(Q, I)

        # Construct Schur decomposition of H[active:maxdim,active:maxdim] in-place
        local_schurfact!(view(H, OneTo(maxdim), :), active, maxdim, Q)

        # Update the Ritz values
        copyto!(ritz.ord, OneTo(maxdim))
        copy_eigenvalues!(ritz.λs, H)
        copy_residuals!(ritz.rs, H, Q, H[maxdim+1, maxdim], x, active:maxdim)

        # Create a permutation that sorts Ritz values from most wanted to least wanted
        sort!(ritz.ord, 1, maxdim, QuickSort, OrderPerm(ritz.λs, ordering))

        # Compute the Frobenius norm of H for the stopping criterion
        isconverged.H_frob_norm[] = norm(H)

        ### PARTITIONING OF SCHUR FORM IN [LOCKED | RETAINED | PURGED]

        # We keep at most `nev` or `nev+1` eigenvalues, depending on the split being halfway a
        # conjugate pair.
        effective_nev = include_conjugate_pair(T, ritz, nev)

        nlock = 0
        @inbounds for i = 1:effective_nev
            if isconverged(ritz.ord[i])
                groups[ritz.ord[i]] = 1
                nlock += 1
            else
                groups[ritz.ord[i]] = 2
            end
        end

        # Determine the new length `k` of the truncated Krylov subspace:
        # 1. The dimension of the active part should be roughly `mindim`; so `k` will be 
        #    larger than `mindim` when converged Ritz vectors have been locked.
        # 2. But `k` can't be so large that the expansion would barely give new information;
        #    hence we meet in the middle: `k` is at most halfway `mindim` and `maxdim`.
        # 3. Further we don't want to split on a conjugate pair.
        ideal_size = min(nlock + mindim, (mindim + maxdim) ÷ 2)
        k = effective_nev
        i = effective_nev + 1

        @inbounds while i ≤ maxdim
            is_pair = include_conjugate_pair(T, ritz, i) == i + 1
            num = is_pair ? 2 : 1
            if k < ideal_size && !isconverged(ritz.ord[i])
                # Keep the best, unconverged Ritz pairs.
                group = 2
                k += num
            else
                # Purge converged, unwanted ones, as well as the worst unconverged values.
                group = 3
            end
            if is_pair
                groups[ritz.ord[i]] = group
                groups[ritz.ord[i+1]] = group
                i += 2
            else
                groups[ritz.ord[i]] = group
                i += 1
            end
        end

        # Typically eigenvalues converge in the desired order: closest to the target first and
        # furthest last. This is not guaranteed though, especially with repeated eigenvalues close
        # to the target: they may start to converge only much later. In the worst case, the number
        # of values we've locked + the number of new Ritz values closer to the target exceed the
        # number of eigenvalues we're looking for. In that case, we unlock / purge the furthest
        # currently locked eigenvalues in favor of the new ones. Here we determine the index
        # `purge` of the first vector to be purged, or `active` if no purging is necessary. Notice
        # that `purge` can be any index in 1:active-1, because locked vectors are merely
        # partitioned as locked vectors, they are never sorted until full convergence.
        purge = 1
        @inbounds while purge < active && groups[purge] == 1
            purge += 1
        end

        partition_schur_three_way!(H, Q, groups)

        # Restore the Hessenberg matrix via Householder reflections.
        # Note that we restore the new active part only -- Q[end, 1:nlock] is small enough
        # by convergence criterion.
        restore_arnoldi!(H, nlock + 1, k, Q, G)

        # Finally do the change of basis to get the length `k` Arnoldi relation.
        @views mul!(V_tmp[:, purge:k], V[:, purge:maxdim], Q[purge:maxdim, purge:k])
        @views copyto!(V[:, purge:k], V_tmp[:, purge:k])
        @views copyto!(V[:, k+1], V[:, maxdim+1])

        # The active part 
        active = nlock + 1

        active > nev && break
    end

    nconverged = active - 1

    @views Vconverged = V[:, 1:nconverged]
    @views Hconverged = H[1:nconverged, 1:nconverged]

    # Sort the converged eigenvalues like the user wants them
    sortschur!(H, copyto!(Q, I), nconverged, ordering)

    # Change of basis
    @views mul!(V_tmp[:, 1:nconverged], Vconverged, Q[1:nconverged, 1:nconverged])
    @views copyto!(Vconverged, V_tmp[:, 1:nconverged])

    # Copy the eigenvalues just one more time
    copy_eigenvalues!(ritz.λs, H, OneTo(nconverged))

    history = History(prods, nconverged, nconverged ≥ nev, nev)
    schur = PartialSchur(Vconverged, Hconverged, ritz.λs[1:nconverged])

    return schur, history
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

    # Be very scared whenever eigenvalues are too close!
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
    sortschur!(R, Q, to::Int, ordering::Ordering)

Reorder the elements on the diagonal of R as indicates by `ordering`.
Eigenvalues are computed from R.
"""
function sortschur!(R, Q, to::Int, ordering::Ordering)
    # Some convoluted insertion sort algorithm
    # `next` is the index of the eigenvalue to be put in position
    # `curr` is the eigenvalue as it is being moved to the front of R
    # `prev` is the index of the block prior to `curr`.

    to ≤ 1 && return

    next_idx = 1

    while next_idx ≤ to
        # Points to the start of the block we're swapping
        curr_idx = next_idx

        # Get the current eigenvalue and remember whether we're on a block
        # so that we skip either 1 or 2 steps.
        curr_size = is_start_of_11_block(R, curr_idx) ? 1 : 2
        curr_λ = eigenvalue(R, curr_idx)

        # Insertion part of the sort
        while curr_idx > 1

            # Identify previous block size (1x1 or 2x2)
            prev_size = is_end_of_11_block(R, curr_idx - 1) ? 1 : 2
            prev_idx = curr_idx - prev_size
            prev_λ = eigenvalue(R, prev_idx)

            # Maybe we're done.
            lt(ordering, curr_λ, prev_λ) === false && break

            # Maybe swap.
            swap!(R, prev_idx, prev_size == 1, curr_size == 1, Q)
            curr_idx -= prev_size
        end

        next_idx += curr_size
    end
end

"""
    include_conjugate_pair(T, ritz, i) → {i, i + 1}

Returns i or i + 1 depending on whether Ritz value i and i + 1 form a conjugate pair
together
"""
@inline function include_conjugate_pair(::Type{<:Real}, ritz::RitzValues, i)
    i >= length(ritz.ord) && return i
    @inbounds λ1 = ritz.λs[ritz.ord[i+0]]
    @inbounds λ2 = ritz.λs[ritz.ord[i+1]]
    return imag(λ1) != 0 && λ1' == λ2 ? i + 1 : i
end

@inline include_conjugate_pair(::Type{T}, ritz::RitzValues, i) where {T} = i

"""
    update_residual_norms!(rs, H, Q, hₖ₊₁ₖ) → rs

Computes the Ritz residuals ‖Ax - λx‖₂ = |yₖ| * |hₖ₊₁ₖ| for each eigenvalue
"""
function copy_residuals!(
    rs::AbstractVector{T},
    H,
    Q,
    hₖ₊₁ₖ,
    x::AbstractVector,
    range::AbstractRange,
) where {T<:Real}
    fill!(rs, 0)
    m = size(H, 2)
    @inbounds for i in range
        fill!(x, zero(T))
        len = collect_eigen!(x, H, i)
        tmp = zero(complex(T))
        for j = 1:len
            tmp += Q[m, j] * x[j]
        end
        rs[i] = abs(tmp * hₖ₊₁ₖ)
    end

    rs
end
