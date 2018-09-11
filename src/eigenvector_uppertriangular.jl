"""
    shifted_backward_sub!(x, R, λ, k) → x

Solve the problem (R[1:k,1:k] - λI) \\ x[1:k] in-place.
"""
function shifted_backward_sub!(x, R::AbstractMatrix{Tm}, λ, k) where {Tm<:Real}
    # Real arithmetic with quasi-upper triangular R
    @inbounds while k > 0
        if k > 1 && R[k,k-1] != zero(Tm)
            # Solve 2x2 problem
            R11, R12 = R[k-1,k-1] - λ, R[k-1,k]
            R21, R22 = R[k  ,k-1]    , R[k  ,k] - λ
            det = R11 * R22 - R21 * R12
            a1 = ( R22 * x[k-1] - R12 * x[k-0]) / det
            a2 = (-R21 * x[k-1] + R11 * x[k-0]) / det
            x[k-1] = a1
            x[k-0] = a2

            # Backward substitute
            for i = 1 : k-2
                x[i] -= R[i,k-1] * x[k-1] + R[i,k-0] * x[k-0]
            end
            k -= 2
        else
            # Solve 1x1 "problem"
            x[k] /= R[k,k] - λ
    
            # Backward substitute
            for i = 1 : k - 1
                x[i] -= R[i,k] * x[k]
            end

            k -= 1
        end
    end
    
    x
end

function shifted_backward_sub!(x, R::AbstractMatrix, λ, k)
    # Generic implementation, upper triangular R
    @inbounds while k > 0
        # Solve 1x1 "problem"
        x[k] /= R[k,k] - λ

        # Backward substitute
        for i = 1 : k - 1
            x[i] -= R[i,k] * x[k]
        end

        k -= 1
    end
    
    x
end

"""
    collect_eigen!(x, R, j) → k

Store the `j`th eigenvector of an upper triangular matrix `R` in `x`.
In the end `norm(x[1:k]) = 1` This function leaves x[k+1:end] untouched!
"""
function collect_eigen!(x::AbstractVector{Tv}, R::AbstractMatrix{Tm}, j::Integer) where {Tm<:Real,Tv<:Number}
    n = size(R, 2)

    @inbounds begin
        # If it's a conjugate pair with the next index, then just increment j.
        if j < n && R[j+1,j] != zero(Tv)
            j += 1
        end

        # Initialize the rhs and do the first backward substitution
        # Then do the rest of the shifted backward substitution in another function,
        # cause λ is either real or complex.
        if j > 1 && R[j,j-1] != 0
            # Complex arithmetic
            R11, R21 = R[j-1,j-1], R[j-0,j-1]
            R12, R22 = R[j-1,j-0], R[j-0,j-0]
            det = R11 * R22 - R21 * R12
            tr = R11 + R22
            λ = (tr + sqrt(complex(tr * tr - 4det))) / 2
            x[j-1] = -R12 / (R11 - λ)
            x[j-0] = one(Tv)
            for i = 1 : j-2
                x[i] = -R[i,j-1] * x[j-1] - R[i,j]
            end

            shifted_backward_sub!(x, R, λ, j-2)
        else
            # Real arithmetic
            λ = R[j,j]
            x[j] = one(Tv)
            for i = 1 : j-1
                x[i] = -R[i,j]
            end
            shifted_backward_sub!(x, R, λ, j-1)
        end

        # Normalize
        nrm = zero(real(Tv))
        for k = 1:j
            nrm += abs2(x[k])
        end
        scale = inv(√nrm)
        for k = 1:j
            x[k] *= scale
        end
    end

    return j
end

function collect_eigen!(x::AbstractVector{Tv}, R::AbstractMatrix, j::Integer) where {Tv}
    n = size(R, 2)

    @inbounds begin
        λ = R[j,j]
        x[j] = one(Tv)
        for i = 1 : j-1
            x[i] = -R[i,j]
        end

        shifted_backward_sub!(x, R, λ, j-1)

        # Normalize
        nrm = zero(real(Tv))
        for k = 1:j
            nrm += abs2(x[k])
        end
        scale = inv(√nrm)
        for k = 1:j
            x[k] *= scale
        end
    end

    return j
end