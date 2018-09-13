"""
    copy_eigenvalues!(λs, A) → λs

Puts the eigenvalues of a quasi-upper triangular matrix A in the λs vector.
"""
function copy_eigenvalues!(λs, A::AbstractMatrix{T}, range = OneTo(size(A, 2)), tol = eps(real(T))) where {T}
    i = first(range)

    @inbounds while i < last(range)
        if is_offdiagonal_small(A, i, tol)
            λs[i] = A[i, i]
            i += 1
        else
            # Conjugate pair
            d = A[i,i] * A[i+1,i+1] - A[i,i+1] * A[i+1,i]
            x = (A[i,i] + A[i+1,i+1]) / 2
            y = sqrt(complex(x*x - d))
            λs[i + 0] = x + y
            λs[i + 1] = x - y
            i += 2
        end
    end

    @inbounds if i == last(range)
        λs[i] = A[i, i] 
    end

    return λs
end

"""
    eigenvalue(R, i) → λ
    
Get the `i`th eigenvalue of R. NOTE: assumes `i` points to the start of
a block.
"""
function eigenvalue(R, i)
    n = minimum(size(R))

    @inbounds begin
        if i == n || iszero(R[i+1,i])
            return complex(R[i,i])
        else
            d = R[i,i] * R[i+1,i+1] - R[i,i+1] * R[i+1,i]
            x = (R[i,i] + R[i+1,i+1]) / 2
            y = sqrt(complex(x*x - d))
            return x + y
        end
    end
end

"""
    eigenvalues(A::AbstractMatrix{T}) → Vector{complex(T)}

Computes the eigenvalues of the matrix A. Assumes that A is quasi-upper triangular.
The eigenvalues are returned in complex arithmetic, even if their imaginary
part is 0.
"""
eigenvalues(A::AbstractMatrix{T}, tol = eps(real(T))) where {T} =
    copy_eigenvalues!(Vector{complex(T)}(undef, size(A, 2)), A, OneTo(size(A, 2)), tol)

"""
    partialeigen(P::PartialSchur) → (Vector{<:Union{Real,Complex}}, Matrix{<:Union{Real,Complex}})

Transforms a partial Schur decomposition into an eigendecomposition.

!!! note

    For real-symmetric and Hermitian matrices the Schur vectors coincide with 
    the eigenvectors, and hence it is not necessary to call this function in 
    that case.

The method still relies on LAPACK to compute the eigenvectors of the (quasi)
upper triangular matrix `R` from the partial Schur decomposition.

!!! note

    This method is currently type unstable for real matrices, since we have not
    yet decided how to deal with complex conjugate pairs of eigenvalues. E.g.
    if almost all eigenvalues are real, but there are just a few conjugate 
    pairs, should all eigenvectors be complex-valued?
"""
function partialeigen(P::PartialSchur)
    vals, vecs = eigen(P.R)
    return vals, P.Q*vecs
end
