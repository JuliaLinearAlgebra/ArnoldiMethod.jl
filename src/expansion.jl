
"""
Allocate some space for an Arnoldi factorization of A where the Krylov subspace has
dimension `max` at most.
"""
function initialize(::Type{T}, n::Int, max::Int = 30) where {T}
    V = Matrix{T}(n, max + 1)
    H = zeros(T, max + 1, max)

    v1 = view(V, :, 1)
    rand!(v1)
    v1 ./= norm(v1)

    Arnoldi{T,typeof(V),typeof(H)}(V, H)
end

"""
Perform Arnoldi iterations.
"""
function iterate_arnoldi!(A::AbstractMatrix{T}, arnoldi::Arnoldi{T}, range::UnitRange{Int}, Δh::Vector{T}) where {T}
    V, H = arnoldi.V, arnoldi.H
    
    @inbounds @views for j = range
        v = V[:, j + 1]
        A_mul_B!(v, A, V[:, j])

        # Orthogonalize
        Ac_mul_B!(H[1:j,j], V[:,1:j], v)
        LinAlg.BLAS.gemv!('N', -one(T), V[:,1:j], H[1:j,j], one(T), v)
        
        Ac_mul_B!(Δh[1:j], V[:,1:j], v)
        LinAlg.BLAS.gemv!('N', -one(T), V[:,1:j], Δh[1:j], one(T), v)
        H[1:j,j] .+= Δh[1:j]

        # Normalize
        H[j + 1, j] = norm(v)
        v ./= H[j + 1, j]
    end

    return arnoldi
end