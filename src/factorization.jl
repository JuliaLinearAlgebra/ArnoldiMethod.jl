# using IRAM: Hessenberg, qr!, mul!, ListOfRotations

struct Arnoldi{T}
    V::StridedMatrix{T}
    H::StridedMatrix{T}
end

"""
Allocate some space for an Arnoldi factorization of A where the Krylov subspace has
dimension `max` at most.
"""
function initialize(::Type{T}, n::Int, max = 30) where {T}
    V = Matrix{T}(n, max + 1)
    H = zeros(T, max + 1, max)

    v1 = view(V, :, 1)
    rand!(v1)
    v1 ./= norm(v1)

    Arnoldi(V, H)
end


"""
Perform Arnoldi iterations.
"""
function iterate_arnoldi!(A::AbstractMatrix{T}, arnoldi::Arnoldi{T}, range::UnitRange{Int}) where {T}
    for j = range
        v_next = view(arnoldi.V, :, j + 1)
        v_curr = view(arnoldi.V, :, j)
        A_mul_B!(v_next, A, v_curr)

        # For now just use modified Gram-Schmidt, but switch to repeated classical soon
        for i = 1 : j
            v_prev = view(arnoldi.V, :, i)
            arnoldi.H[i, j] = dot(v_prev, v_next)
            v_next .-= arnoldi.H[i, j] .* v_prev
        end

        arnoldi.H[j + 1, j] = norm(v_next)
        v_next ./= arnoldi.H[j + 1, j]
    end

    return arnoldi
end

function shifted_qr_step!(H::AbstractMatrix{T}, θ, rotations::ListOfRotations) where {T}
    m = size(H,1) - 1

    @inbounds for i = 1:m # subtract λ from diagonal
        H[i,i] -= θ
    end

    qr!(Hessenberg(view(H, 1:m, 1:m)), rotations)

    @inbounds H[m,m] = H[m+1,m]
    @inbounds H[m+1,m] = zero(T)
    
    mul!(UpperTriangularMatrix(H), rotations) # RQ step

    @inbounds for i = 1:m # add λ back to diagonal
        H[i,i] += θ
    end
end

"""
Shrink the dimension of Krylov subspace from `max` to `min` using shifted QR,
where the Schur vectors corresponding to smallest eigenvalues are removed.
"""
function implicit_restart!(arnoldi::Arnoldi{T}, min = 5, max = 30) where {T}
    h = arnoldi.H[max + 1, max]

    λs = sort!(eigvals(view(arnoldi.H, 1 : max, 1 : max)), by = abs, rev = true)
    Q = eye(Complex128, max)
    H = copy(arnoldi.H)

    rotations = ListOfRotations(T, max)

    for m = max : -1 : min + 1

        shifted_qr_step!(view(H, 1 : m + 1, 1 : m), λs[m], rotations)
        mul!(view(Q, 1 : max, 1 : m), rotations)

    end

    # Remove the last columns and rows of H
    H = H[1 : min+1, 1 : min]

    # Update the H[end, end] value
    H[min + 1, min] = h

    # Update the Krylov basis
    V_new = [arnoldi.V[:, 1 : max] * Q[:, 1 : min] arnoldi.V[:, max + 1]]

    # Copy to the Arnoldi factorization
    copy!(view(arnoldi.H, 1 : min + 1, 1 : min), H)
    copy!(view(arnoldi.V, :, 1 : min + 1), V_new)

    return arnoldi
end

"""
Run IRAM for a couple restarts
"""
function restarted_arnoldi(A::AbstractMatrix{T}, min = 5, max = 30, restarts = 4) where {T}
    n = size(A, 1)

    arnoldi = initialize(T, n, max)
    iterate_arnoldi!(A, arnoldi, 1 : max)

    for i = 1 : restarts
        implicit_restart!(arnoldi, min, max)
        iterate_arnoldi!(A, arnoldi, min + 1 : max)
    end

    λs, xs = eig(view(arnoldi.H, 1 : max, 1 : max))

    return λs, view(arnoldi.V, :, 1 : max) * xs
end

"""
Run IRAM until the eigenpair is a good enough approximation
"""
function restarted_arnoldi_2(A::AbstractMatrix{T}, min = 5, max = 30, criterion = 1e-5) where {T}
    n = size(A, 1)

    arnoldi = initialize(T, n, max)
    iterate_arnoldi!(A, arnoldi, 1 : max)

    λs, xs = eig(view(arnoldi.H, 1 : max, 1 : max))

    while abs(arnoldi.H[max+1,max]) * abs(xs[max,1]) > criterion #doesn't seem to be working right, probably because eigenvectors are not sorted
        implicit_restart!(arnoldi, min, max)
        iterate_arnoldi!(A, arnoldi, min + 1 : max)
        λs, xs = eig(view(arnoldi.H, 1 : max, 1 : max)) #and this seems excessive
    end

    return λs[1], view(arnoldi.V, :, 1 : max) * xs[:,1] #returns one random eigenpair of A
end