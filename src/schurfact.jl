using LinearAlgebra: givensAlgorithm
using Base: @propagate_inbounds

import LinearAlgebra: lmul!, rmul!
import Base: Matrix

@propagate_inbounds is_offdiagonal_small(
    H::AbstractMatrix{T},
    i::Int,
    tol = eps(real(T)),
) where {T} = abs(H[i+1, i]) ≤ tol * (abs(H[i, i]) + abs(H[i+1, i+1]))


abstract type SmallRotation end

"""
Given's rotation acting on rows i:i+1
"""
struct Rotation2{Tc,Ts} <: SmallRotation
    c::Tc
    s::Ts
    i::Int
end

"""
Two Given's rotations acting on rows i:i+2. This could also be implemented as one Householder
reflector!
"""
struct Rotation3{Tc,Ts} <: SmallRotation
    c₁::Tc
    s₁::Ts
    c₂::Tc
    s₂::Ts
    i::Int
end

# Some utility to materialize a rotation to a matrix.
function Matrix(r::Rotation2{Tc,Ts}, n::Int) where {Tc,Ts}
    r.i < n || throw(ArgumentError("Matrix should have order $(r.i+1) or larger"))
    G = Matrix{promote_type(Tc, Ts)}(I, n, n)
    G[r.i+0, r.i+0] = r.c
    G[r.i+1, r.i+0] = -conj(r.s)
    G[r.i+0, r.i+1] = r.s
    G[r.i+1, r.i+1] = r.c
    return G
end

function Matrix(r::Rotation3{Tc,Ts}, n::Int) where {Tc,Ts}
    G₁ = Matrix(Rotation2(r.c₁, r.s₁, r.i + 1), n)
    G₂ = Matrix(Rotation2(r.c₂, r.s₂, r.i), n)
    return G₂ * G₁
end

"""
Get a rotation that maps [p₁, p₂] to a multiple of [1, 0]
"""
function get_rotation(p₁, p₂, i::Int)
    c, s, nrm = givensAlgorithm(p₁, p₂)
    Rotation2(c, s, i), nrm
end

"""
Get a rotation that maps [p₁, p₂, p₃] to a multiple of [1, 0, 0]
"""
function get_rotation(p₁, p₂, p₃, i::Int)
    c₁, s₁, nrm₁ = givensAlgorithm(p₂, p₃)
    c₂, s₂, nrm₂ = givensAlgorithm(p₁, nrm₁)
    Rotation3(c₁, s₁, c₂, s₂, i), nrm₂
end

"""
Passed into the Schur factorization function if you do not wish to have the Schur vectors.
"""
struct NotWanted end

@inline lmul!(G::SmallRotation, A::AbstractMatrix) = lmul!(G, A, 1, size(A, 2))
@inline rmul!(A::AbstractMatrix, G::SmallRotation) = rmul!(A, G, 1, size(A, 1))

lmul!(::SmallRotation, ::NotWanted, args...) = nothing
rmul!(::NotWanted, ::SmallRotation, args...) = nothing

@inline function lmul!(G::Rotation3, A::AbstractMatrix, from::Int, to::Int)
    @inbounds for j = from:to
        a₁ = A[G.i+0, j]
        a₂ = A[G.i+1, j]
        a₃ = A[G.i+2, j]

        a₂′ = G.c₁ * a₂ + G.s₁ * a₃
        a₃′ = -G.s₁' * a₂ + G.c₁ * a₃

        a₁′′ = G.c₂ * a₁ + G.s₂ * a₂′
        a₂′′ = -G.s₂' * a₁ + G.c₂ * a₂′

        A[G.i+0, j] = a₁′′
        A[G.i+1, j] = a₂′′
        A[G.i+2, j] = a₃′
    end

    A
end

@inline function rmul!(A::AbstractMatrix, G::Rotation3, from::Int, to::Int)
    @inbounds for j = from:to
        a₁ = A[j, G.i+0]
        a₂ = A[j, G.i+1]
        a₃ = A[j, G.i+2]

        a₂′ = a₂ * G.c₁ + a₃ * G.s₁'
        a₃′ = a₂ * -G.s₁ + a₃ * G.c₁

        a₁′′ = a₁ * G.c₂ + a₂′ * G.s₂'
        a₂′′ = a₁ * -G.s₂ + a₂′ * G.c₂

        A[j, G.i+0] = a₁′′
        A[j, G.i+1] = a₂′′
        A[j, G.i+2] = a₃′
    end
    A
end

@inline function lmul!(G::Rotation2, A::AbstractMatrix, from::Int, to::Int)
    @inbounds for j = from:to
        a₁ = A[G.i+0, j]
        a₂ = A[G.i+1, j]

        a₁′ = G.c * a₁ + G.s * a₂
        a₂′ = -G.s' * a₁ + G.c * a₂

        A[G.i+0, j] = a₁′
        A[G.i+1, j] = a₂′
    end

    A
end

@inline function rmul!(A::AbstractMatrix, G::Rotation2, from::Int, to::Int)
    @inbounds for j = from:to
        a₁ = A[j, G.i+0]
        a₂ = A[j, G.i+1]

        a₁′ = a₁ * G.c + a₂ * G.s'
        a₂′ = a₁ * -G.s + a₂ * G.c

        A[j, G.i+0] = a₁′
        A[j, G.i+1] = a₂′
    end
    A
end

function double_shift_schur!(
    H::AbstractMatrix{Tv},
    from::Int,
    to::Int,
    trace::Tv,
    determinant::Tv,
    Q = NotWanted(),
) where {Tv<:Real}
    m, n = size(H)

    # Compute the nonzero entries of p = (H - μ₋I)(H - μ₊I)e₁.
    # Because of the Hessenberg structure we only need H[min:min+2,min:min+1] to form p.
    @inbounds H₁₁ = H[from+0, from+0]
    @inbounds H₂₁ = H[from+1, from+0]

    @inbounds H₁₂ = H[from+0, from+1]
    @inbounds H₂₂ = H[from+1, from+1]
    @inbounds H₃₂ = H[from+2, from+1]

    # Todo: avoid under/overflow.
    p₁ = H₁₁ * H₁₁ + H₁₂ * H₂₁ - trace * H₁₁ + determinant
    p₂ = H₂₁ * (H₁₁ + H₂₂ - trace)
    p₃ = H₃₂ * H₂₁

    # Map that column to a mulitiple of e₁ via two Given's rotations
    G₁, nrm = get_rotation(p₁, p₂, p₃, from)

    # Apply the Given's rotations
    lmul!(G₁, H, from, n)
    rmul!(H, G₁, 1, min(from + 3, m))
    rmul!(Q, G₁)

    # Bulge chasing. First step of the for-loop below looks like:
    #  from           to
    #     ↓           ↓
    #     x x x x x x x     x x x x x x x     x + + + x x x
    # i → x x x x x x x     + + + + + + +     x + + + x x x 
    #     x x x x x x x     o + + + + + +       + + + x x x
    #     x x x x x x x  ⇒  o + + + + + +  ⇒    + + + x x x
    #       |   x x x x           x x x x       + + + x x x
    #       |     x x x             x x x             x x x
    #       |       x x               x x               x x
    #       ↑
    #       i
    #
    # Last iterations looks like:
    #  from           to
    #     ↓           ↓
    #     x x x x x x x     x x x x x x x     x x x x + + +
    #     x x x x x x x     x x x x x x x     x x x x + + +
    #       x x x x x x       x x x x x x       x x x + + +
    #         x x x x x  ⇒    x x x x x x  ⇒      x x + + +
    # i → ----- x x x x           + + + +           x + + +
    #           x x x x           o + + +             + + +
    #           x x x x           o + + +             + + +
    #             ↑
    #             i

    @inbounds for i = from+1:to-2
        p₁ = H[i+0, i-1]
        p₂ = H[i+1, i-1]
        p₃ = H[i+2, i-1]

        G, nrm = get_rotation(p₁, p₂, p₃, i)

        # First column is done by hand
        H[i+0, i-1] = nrm
        H[i+1, i-1] = zero(Tv)
        H[i+2, i-1] = zero(Tv)

        # Rotate remaining columns
        lmul!(G, H, i, n)

        # Create a new bulge
        rmul!(H, G, 1, min(i + 3, m))
        rmul!(Q, G)
    end

    # Last bulge is just one Given's rotation
    #     from          to
    #       ↓           ↓
    #from → x x x x x x x    x x x x x x x    x x x x x + +  
    #       x x x x x x x    x x x x x x x    x x x x x + +  
    #         x x x x x x      x x x x x x      x x x x + +  
    #           x x x x x  ⇒     x x x x x  ⇒     x x x + +  
    #             x x x x          x x x x          x x + +  
    #               x x x            + + +            x + +  
    # to  → ------- x x x            o + +              + +


    @inbounds Gₙ, nrm = get_rotation(H[to-1, to-2], H[to, to-2], to - 1)
    @inbounds H[to-1, to-2] = nrm
    @inbounds H[to, to-2] = zero(Tv)

    lmul!(Gₙ, H, to - 1, n)
    rmul!(H, Gₙ, 1, to)
    rmul!(Q, Gₙ)

    H
end

function single_shift_schur!(
    H::AbstractMatrix{Tv},
    from::Int,
    to::Int,
    μ::Number,
    Q = NotWanted(),
) where {Tv<:Number}
    m, n = size(H)

    # Compute the nonzero entries of p = (H - μI)e₁.
    @inbounds H₁₁ = H[from+0, from+0]
    @inbounds H₂₁ = H[from+1, from+0]

    p₁ = H₁₁ - μ
    p₂ = H₂₁

    # Map that column to a mulitiple of e₁ via two Given's rotations
    G₁, nrm = get_rotation(p₁, p₂, from)

    # Apply the Given's rotations
    lmul!(G₁, H, from, n)
    rmul!(H, G₁, 1, min(from + 2, m))
    rmul!(Q, G₁)

    # Bulge chasing. First step of the for-loop below looks like:
    #  from           to
    #     ↓           ↓
    #     x x x x x x x     x x x x x x x     x + + x x x x
    # i → x x x x x x x     + + + + + + +     x + + x x x x 
    #     x x x x x x x     o + + + + + +       + + x x x x
    #         x x x x x  ⇒      x x x x x  ⇒    + + x x x x
    #       |   x x x x           x x x x           x x x x
    #       |     x x x             x x x             x x x
    #       |       x x               x x               x x
    #       ↑
    #       i
    #
    # Last iterations looks like:
    #  from           to
    #     ↓           ↓
    #     x x x x x x x     x x x x x x x     x x x x x + +
    #     x x x x x x x     x x x x x x x     x x x x x + +
    #       x x x x x x       x x x x x x       x x x x + +
    #         x x x x x  ⇒      x x x x x  ⇒      x x x + +
    #           x x x x           x x x x           x x + +
    # i → ------- x x x             + + +             x + +
    #             x x x             0 + +               + +
    #               ↑
    #               i

    @inbounds for i = from+1:to-1
        p₁ = H[i+0, i-1]
        p₂ = H[i+1, i-1]

        G, nrm = get_rotation(p₁, p₂, i)

        # First column is done by hand
        H[i+0, i-1] = nrm
        H[i+1, i-1] = zero(Tv)

        # Rotate remaining columns
        lmul!(G, H, i, n)

        # Create a new bulge
        rmul!(H, G, 1, min(i + 2, m))
        rmul!(Q, G)
    end

    H
end

"""
Returns a tuple (is_real, c, s) where is_real is true iff the matrix H = [H₁₁ H₁₂; H₂₁ H₂₂] has
real eigenvalues. The (c, s) values form the most stable Given's rotation that makes G * H * G'
upper triangular (i.e. zeros out the bottom left entry).
"""
function upper_triangular_2x2(H₁₁::T, H₁₂::T, H₂₁::T, H₂₂::T) where {T<:Real}
    # Early exit in trivial cases.
    (iszero(H₂₁) || (iszero(H₁₁ - H₂₂) && sign(H₁₂) != sign(H₂₁))) &&
        return false, one(T), zero(T)
    iszero(H₁₂) && return true, zero(T), one(T)

    # The characteristic polynomial is `λ² - tr(H)λ + det(H) = 0`
    # => λ = (tr(H) ± √(tr(H)² - 4det(H))) / 2.
    # Conjugate pair iff tr(H)² - 4det(H) < 0
    # => (H₁₁ + H₂₂)² - 4(H₁₁H₂₂ - H₁₂H₂₁) < 0
    # => ((H₁₁ - H₂₂) / 2)² + H₁₂H₂₁ < 0.
    p = (H₁₁ - H₂₂) / 2
    bcmax = max(abs(H₁₂), abs(H₂₁))
    bcmis = min(abs(H₁₂), abs(H₂₁)) * sign(H₁₂) * sign(H₂₁)
    scale = max(abs(p), bcmax)
    z = (p / scale) * p + (bcmax / scale) * bcmis

    # Return when complex. LAPACK also deals with 0 < z < 4eps(T), but I don't find it worth it.
    # Note that the < is important, cause H = [1 -1/4; 1 2] for example is not upper triangular
    # and has zero discriminant.
    z < 0 && return false, one(T), zero(T)

    # The rotation is basically just a "perfect" Wilkinson shift, so compute from either
    # (H₁₁ - λ₁, H₁₂) or (H₁₁ - λ₂, H₁₂). We pick the option that would avoid catastrophic
    # cancellation by choosing equal signs. A small rewrite:
    # H₁₁ - λ with λ = (tr(H)/2 ± √((tr(H)/2)² - det(H))) means
    # H₁₁ - λ = (H₁₁ - H₂₂)/2 ± √(((H₁₁ - H₂₂)/2)² - H₂₁H₁₂)
    H₁₁_min_λ = p + copysign(sqrt(scale) * sqrt(z), p)
    nrm = hypot(H₂₁, H₁₁_min_λ)  # could use givensAlgorithm, but hypot is likely better regardless.
    return true, H₁₁_min_λ / nrm, H₂₁ / nrm
end

"""
Returns a tuple (is_single, λ) where is_single is true iff the matrix H = [H₁₁ H₁₂; H₂₁ H₂₂] has
real eigenvalues. In that case λ is a Wilkinson shift: the eigenvalue closest to H₂₂.
"""
function use_single_shift(H₁₁::T, H₁₂::T, H₂₁::T, H₂₂::T) where {T}
    # TODO: Merge the scaling tricks with the above.
    # Scaling to avoid losing precision in the case where we have nearly
    # repeated eigenvalues.
    scale = abs(H₁₁) + abs(H₁₂) + abs(H₂₁) + abs(H₂₂)
    H₁₁ /= scale
    H₁₂ /= scale
    H₂₁ /= scale
    H₂₂ /= scale

    # Trace and discriminant of small eigenvalue problem. Again,
    # λ = tr(H) / 2 ± √(tr(H)² / 4 - det(H))
    #   = (H₁₁ + H₂₂) / 2 ± √((H₁₁ - H₂₂)/2)² - H₁₂H₂₁) written in a funny way:
    t = (H₁₁ + H₂₂) / 2
    d = (H₁₁ - t) * (H₂₂ - t) - H₁₂ * H₂₁

    # Conjugate pair: need to do a double shift.
    d > zero(T) && return false, zero(T)

    # The shift is picked as the closest eigenvalue of the 2x2 block near H[to,to]
    sqrt_discr = sqrt(abs(d))
    λ₁ = t + sqrt_discr
    λ₂ = t - sqrt_discr
    λ = abs(H₂₂ - λ₁) < abs(H₂₂ - λ₂) ? λ₁ : λ₂
    return true, λ * scale
end

###
### Real arithmetic
###
function local_schurfact!(
    H::AbstractMatrix{T},
    start::Int,
    to::Int,
    Q = NotWanted(),
    tol = eps(T),
    maxiter = 100 * size(H, 1),
) where {T<:Real}
    # iteration count
    iter = 0

    @inbounds while to > start
        iter += 1
        iter > maxiter && throw("QR algorithm did not converge")

        # Indexing
        # `to` points to the column where the off-diagonal value was last zero.
        # while `from` points to the smallest index such that there is no small off-diagonal
        # value in columns from:end-1. Sometimes `from` is just 1. Cartoon of a split 
        # with from != 1:
        # 
        #  + + | | | | + +
        #  + + | | | | + +
        #    o X X X X = =
        #      X X X X = =
        #      . X X X = =
        #      .   X X = =
        #      .     o + +
        #      .     . + +
        #      ^     ^
        #   from   to
        # The X's form the unreduced Hessenberg matrix we are applying QR iterations to,
        # the | and = values get updated by Given's rotations, while the + values remain
        # untouched! The o's are zeros -- or numerically considered zeros.

        # We keep `from` one column past the zero off-diagonal value, so we check whether
        # the `from - 1` column has a small off-diagonal value.
        from = to
        while from > start
            if is_offdiagonal_small(H, from - 1, tol)
                H[from, from-1] = zero(T)
                break
            end
            from -= 1
        end

        if from == to
            # A single eigenvalue has converged
            to -= 1
            continue
        end

        # We can safely work with the bottom 2×2 block C := H[to-1:to,to-1:to] now.
        C₁₁, C₁₂ = H[to-1, to-1], H[to-1, to]
        C₂₁, C₂₂ = H[to, to-1], H[to, to]

        # A 2x2 block is always considered converged. Complex conjugates are left as a 2x2 block.
        # Real eigenvalues are "manually" upper triangularized.
        if from + 1 == to
            # In case of real eigenvalues, it should in principle be enough to do a single
            # Wilkinson shift: that would be a perfect shift, and upper triangularizes the 2x2
            # block. But it can also completely destroy accuracy. So, we do this single shift with
            # more accurate arithmetic with the rotation computed above.
            is_real, cs, sn = upper_triangular_2x2(C₁₁, C₁₂, C₂₁, C₂₂)

            if is_real
                G = Rotation2(cs, sn, from)
                lmul!(G, H, from, size(H, 2))
                rmul!(H, G, 1, to)
                rmul!(Q, G)
                H[to, to-1] = zero(T)
            end

            to -= 2
            continue
        end

        # Real eigenvalues: single wilkinson shift. Conjugate pair: Francis double shift.
        is_single, μ = use_single_shift(C₁₁, C₁₂, C₂₁, C₂₂)

        if is_single
            single_shift_schur!(H, from, to, μ, Q)
        else
            # A double shift is done by computing the first column
            # of (H - μ₊I)(H - μ₋I) where μ₊ and μ₋ are the eigenvalues of C. That's identical to
            # (H² - (μ₊ + μ₋)H + μ₊μ₋), and since μ₊₋ = (tr(C) ± √(tr(C)² - 4det(C))) / 2.
            # So, identical to H² - tr(C)H + det(C)I.
            trace = C₁₁ + C₂₂
            determinant = C₁₁ * C₂₂ - C₁₂ * C₂₁
            double_shift_schur!(H, from, to, trace, determinant, Q)
        end
    end

    return true
end

###
### Generic implementation
###
function local_schurfact!(
    H::AbstractMatrix{T},
    start::Int,
    to::Int,
    Q = NotWanted(),
    tol = eps(real(T)),
    maxiter = 100 * size(H, 1),
) where {T}
    # iteration count
    iter = 0

    @inbounds while true
        iter += 1
        iter > maxiter && return false

        # Indexing, see the real arithmetic version!

        from = to
        while from > start && !is_offdiagonal_small(H, from - 1, tol)
            from -= 1
        end

        if from == to
            # This just means H[to, to-1] == 0, so one eigenvalue converged at the end
            H[from, from-1] = zero(T)
            to -= 1
        else
            # Compute Wilkinson shift
            H₁₁, H₁₂ = H[to-1, to-1], H[to-1, to]
            H₂₁, H₂₂ = H[to, to-1], H[to, to]
            d = H₁₁ * H₂₂ - H₂₁ * H₁₂
            t = H₁₁ + H₂₂
            sqr = sqrt(t * t - 4d)
            λ₁ = (t + sqr) / 2
            λ₂ = (t - sqr) / 2
            λ = abs(H₂₂ - λ₁) < abs(H₂₂ - λ₂) ? λ₁ : λ₂

            # Run a bulge chase
            single_shift_schur!(H, from, to, λ, Q)
        end

        # Converged!
        to ≤ start && break
    end

    return true
end

local_schurfact!(
    H::AbstractMatrix{T},
    Q = NotWanted(),
    tol = eps(real(T)),
    maxiter = 100 * size(H, 1),
) where {T} = local_schurfact!(H, 1, size(H, 2), Q, tol, maxiter)
