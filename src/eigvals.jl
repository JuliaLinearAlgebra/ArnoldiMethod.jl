using Printf
using LinearAlgebra: givensAlgorithm
import Base: @propagate_inbounds

@propagate_inbounds is_offdiagonal_small(H, i, tol) = abs(H[i+1,i]) < tol*(abs(H[i,i]) + abs(H[i+1,i+1]))

"""
Computes the eigenvalues of the matrix A. Assumes that A is in Schur form.
"""
function eigvalues(A::AbstractMatrix{T}; tol = eps(real(T))) where {T}
    n = size(A, 1)
    λs = Vector{complex(T)}(undef, n)
    i = 1

    @inbounds while i < n
        if is_offdiagonal_small(A, i, tol)
            λs[i] = A[i, i]
            i += 1
        else
            d = A[i,i] * A[i+1,i+1] - A[i,i+1] * A[i+1,i]
            x = 0.5*(A[i,i] + A[i+1,i+1])
            y = sqrt(complex(x*x - d))
            λs[i] = x + y
            λs[i + 1] = x - y
            i += 2
        end
    end

    if i == n 
        @inbounds λs[i] = A[n, n] 
    end

    return λs
end

local_schurfact!(A, Q) = local_schurfact!(A, Q, 1, size(A, 1))

function local_schurfact!(H::AbstractMatrix{T}, Q::AbstractMatrix{T}, start, stop; tol = eps(real(T)), maxiter = 100*size(H, 1)) where {T<:Real}
    to = stop

    # iteration count
    iter = 0

    @inbounds while true
        iter += 1

        iter > maxiter && return false

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
        while from > start && !is_offdiagonal_small(H, from - 1, tol)
            from -= 1
        end

        if from == to
            # This just means H[to, to-1] == 0, so one eigenvalue converged at the end
            to -= 1
        else
            # Now we are sure we can work with a 2x2 block H[to-1:to,to-1:to]
            # We check if this block has a conjugate eigenpair, which might mean we have
            # converged w.r.t. this block if from + 1 == to. 
            # Otherwise, if from + 1 < to, we do either a single or double shift, based on
            # whether the H[to-1:to,to-1:to] part has real eigenvalues or a conjugate pair.

            H₁₁, H₁₂ = H[to-1,to-1], H[to-1,to]
            H₂₁, H₂₂ = H[to  ,to-1], H[to  ,to]

            # Matrix determinant and trace
            d = H₁₁ * H₂₂ - H₂₁ * H₁₂
            t = H₁₁ + H₂₂
            discriminant = t * t - 4d

            if discriminant ≥ zero(T)
                # Real eigenvalues.
                # Note that if from + 1 == to in this case, then just one additional
                # iteration is necessary, since the Wilkinson shift will do an exact shift.

                # Determine the Wilkinson shift -- the closest eigenvalue of the 2x2 block
                # near H[to,to]
                sqr = sqrt(discriminant)
                λ₁ = (t + sqr) / 2
                λ₂ = (t - sqr) / 2
                λ = abs(H₂₂ - λ₁) < abs(H₂₂ - λ₂) ? λ₁ : λ₂
                # Run a bulge chase
                single_shift_schur!(H, Q, λ, from, to)
            else
                # Conjugate pair
                if from + 1 == to
                    # A conjugate pair has converged apparently!
                    to -= 2
                else
                    # Otherwise we do a double shift!
                    sqr = sqrt(complex(discriminant))
                    λ = (t + sqr) / 2
                    double_shift_schur!(H, from, to, λ, Q)
                end
            end
        end

        # Converged!
        to ≤ start && break
    end

    return true
end

function local_schurfact!(H::AbstractMatrix{T}, Q::AbstractMatrix{T}, start, stop; tol = eps(real(T)), maxiter = 100*size(H, 1)) where {T}
    to = stop

    # iteration count
    iter = 0

    @inbounds while true
        iter += 1

        # Don't like that this throws :|
        # iter > maxiter && throw(ArgumentError("iteration limit $maxiter reached"))
        iter > maxiter && return false

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
        while from > start && !is_offdiagonal_small(H, from - 1, tol)
            from -= 1
        end

        if from == to
            # This just means H[to, to-1] == 0, so one eigenvalue converged at the end
            to -= 1
        else
            # Now we are sure we can work with a 2x2 block H[to-1:to,to-1:to]
            # We check if this block has a conjugate eigenpair, which might mean we have
            # converged w.r.t. this block if from + 1 == to. 
            # Otherwise, if from + 1 < to, we do either a single or double shift, based on
            # whether the H[to-1:to,to-1:to] part has real eigenvalues or a conjugate pair.

            H₁₁, H₁₂ = H[to-1,to-1], H[to-1,to]
            H₂₁, H₂₂ = H[to  ,to-1], H[to  ,to]

            # Matrix determinant and trace
            d = H₁₁ * H₂₂ - H₂₁ * H₁₂
            t = H₁₁ + H₂₂
            discriminant = t * t - 4d

            # Note that if from + 1 == to in this case, then just one additional
            # iteration is necessary, since the Wilkinson shift will do an exact shift.

            # Determine the Wilkinson shift -- the closest eigenvalue of the 2x2 block
            # near H[to,to]
            sqr = sqrt(discriminant)
            λ₁ = (t + sqr) / 2
            λ₂ = (t - sqr) / 2
            λ = abs(H₂₂ - λ₁) < abs(H₂₂ - λ₂) ? λ₁ : λ₂
            # Run a bulge chase
            single_shift_schur!(H, Q, λ, from, to)
        end

        # Converged!
        to ≤ start && break
    end

    return true
end

function single_shift_schur!(HH::StridedMatrix, Q::AbstractMatrix, shift::Number, istart::Integer, iend::Integer)
    m = size(HH, 1)
    H11 = HH[istart, istart]
    H21 = HH[istart + 1, istart]
    if m > istart + 1
        Htmp = HH[istart + 2, istart]
        HH[istart + 2, istart] = 0
    end
    c, s = givensAlgorithm(H11 - shift, H21)
    G = Givens(c, s, istart)
    lmul!(G, HH)
    rmul!(HH, G)
    rmul!(Q, G)
    for i = istart:iend - 2
        c, s = givensAlgorithm(HH[i + 1, i], HH[i + 2, i])
        G = Givens(c, s, i + 1)
        lmul!(G, HH)
        HH[i + 2, i] = Htmp
        if i < iend - 2
            Htmp = HH[i + 3, i + 1]
            HH[i + 3, i + 1] = 0
        end
        rmul!(HH, G)
        rmul!(Q, G)
    end
    return HH
end

function double_shift_schur!(H::AbstractMatrix{Tv}, min, max, μ::Complex, Q::AbstractMatrix) where {Tv<:Real}
    # Compute the three nonzero entries of (H - μ₂)(H - μ₁)e₁.
    @inbounds p₁ = abs2(μ) - 2 * real(μ) * H[min,min] + H[min,min] * H[min,min] + H[min,min+1] * H[min+1,min]
    @inbounds p₂ = -2.0 * real(μ) * H[min+1,min] + H[min+1,min] * H[min,min] + H[min+1,min+1] * H[min+1,min]
    @inbounds p₃ = H[min+2,min+1] * H[min+1,min]

    # Map that column to a mulitiple of e₁ via three Given's rotations
    c₁, s₁, nrm = givensAlgorithm(p₂, p₃)
    c₂, s₂,     = givensAlgorithm(p₁, nrm)
    G₁ = Givens(c₁, s₁, min+1)
    G₂ = Givens(c₂, s₂, min)

    # Apply the Given's rotations
    lmul!(G₁, H)
    lmul!(G₂, H)
    rmul!(H, G₁)
    rmul!(H, G₂)

    # Update Q
    rmul!(Q, G₁)
    rmul!(Q, G₂)

    # Bulge chasing. First step of the for-loop below looks like:
    #   min           max
    #     ↓           ↓
    #     x x x x x x x     x x x x x x x     x + + + x x x
    # i → x x x x x x x     + + + + + + +     x + + + x x x 
    #     x x x x x x x     o + + + + + +       + + + x x x
    #     x x x x x x x  ⇒  o + + + + + +  ⇒   + + + x x x
    #       |   x x x x           x x x x       + + + x x x
    #       |     x x x             x x x             x x x
    #       |       x x               x x               x x
    #       ↑
    #       i
    #
    # Last iterations looks like:
    #   min           max
    #     ↓           ↓
    #     x x x x x x x     x x x x x x x     x x x x + + +
    #     x x x x x x x     x x x x x x x     x x x x + + +
    #       x x x x x x       x x x x x x       x x x + + +
    #         x x x x x  ⇒    x x x x x x  ⇒     x x + + +
    # i → ----- x x x x           + + + +           x + + +
    #           x x x x           o + + +             + + +
    #           x x x x           o + + +             + + +
    #             ↑
    #             i

    @inbounds for i = min + 1 : max - 2
        c₁, s₁, nrm = givensAlgorithm(H[i+1,i-1], H[i+2,i-1])
        c₂, s₂,     = givensAlgorithm(H[i,i-1], nrm)
        G₁ = Givens(c₁, s₁, i+1)
        G₂ = Givens(c₂, s₂, i)

        # Restore to Hessenberg
        lmul!(G₁, H)
        lmul!(G₂, H)

        # Introduce zeros below Hessenberg part
        H[i+1,i-1] = zero(Tv)
        H[i+2,i-1] = zero(Tv)

        # Create a new bulge
        rmul!(H, G₁)
        rmul!(H, G₂)

        # Update Q
        rmul!(Q, G₁)
        rmul!(Q, G₂)
    end

    # Last bulge is just one Given's rotation
    #     min           max
    #       ↓           ↓
    # min → x x x x x x x    x x x x x x x    x x x x x + +  
    #       x x x x x x x    x x x x x x x    x x x x x + +  
    #         x x x x x x      x x x x x x      x x x x + +  
    #           x x x x x  ⇒     x x x x x  ⇒     x x x + +  
    #             x x x x          x x x x          x x + +  
    #               x x x            + + +            x + +  
    # max → ------- x x x            o + +              + +


    @inbounds c, s, = givensAlgorithm(H[max-1,max-2], H[max,max-2])
    G = Givens(c, s, max-1)
    lmul!(G, H)
    @inbounds H[max,max-2] = zero(Tv)
    rmul!(H, G)
    rmul!(Q, G)

    H
end
