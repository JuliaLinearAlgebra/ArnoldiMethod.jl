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
    V, H = arnoldi.V, arnoldi.H

    @inbounds @views for j = range
        v = V[:, j + 1]
        A_mul_B!(v, A, V[:, j])

        # Orthogonalize
        for i = 1 : j
            H[i, j] = dot(V[:, i], v)
            v .-= H[i, j] .* V[:, i]
        end

        # Normalize
        H[j + 1, j] = norm(v)
        v ./= H[j + 1, j]
    end

    return arnoldi
end

"""
Assume H is an (m + 1) x m unreduced Hessenberg matrix, the active part in the Arnoli
decomposition.
"""
function shifted_qr_step!(H::AbstractMatrix{T}, μ, rotations::ListOfRotations) where {T}
    m = size(H, 1) - 1

    # Apply the shift
    @inbounds for i = 1:m
        H[i,i] -= μ
    end

    # QR step; compute & apply Given's rotations from the left
    qr!(Hessenberg(view(H, 1:m, 1:m)), rotations)

    # Do the last Given's rotation by hand
    @inbounds H[m,m] = H[m+1,m]
    @inbounds H[m+1,m] = zero(T)
    
    # RQ step; apply Given's rotations from the right
    mul!(UpperTriangularMatrix(H), rotations)

    # Undo the shift
    @inbounds for i = 1:m
        H[i,i] += μ
    end
end

"""
Shrink the dimension of Krylov subspace from `max` to `min` using shifted QR,
where the Schur vectors corresponding to smallest eigenvalues are removed.
"""
function implicit_restart!(arnoldi::Arnoldi{T}, min = 5, max = 30, active = 1) where {T}
    V, H = arnoldi.V, arnoldi.H
    λs = sort!(eigvals(view(H, active:max, active:max)), by = abs)
    Q = eye(T, max)
    rotations = ListOfRotations(T, max)

    idx = 1

    @views for m = max : -1 : min + 1

        # Pick a shift
        μ = λs[idx]

        # Apply a shifted QR step to the H[active:m, active:m]
        shifted_qr_step!(H[active:m+1, active:m], μ, rotations)

        # Apply the rotations to the G block
        mul!(H[1:active-1, active:m], rotations)
        
        # Apply the rotations to Q
        mul!(Q[1:max, active:m], rotations)

        idx += 1
    end

    # Update the Krylov basis
    V_new = view(V, :, active:max) * view(Q, active:max, active:min)

    # Copy to the Arnoldi factorization
    copy!(view(V, :, active:min), V_new)
    copy!(view(V, :, min + 1), view(V, :, max + 1))

    return arnoldi
end


"""
Find the largest index `i` such that `H[i, i-1] ≈ 0`. This means that `i:max` constitutes the
active part in the Arnoldi decomp, while `V[:, 1:i-1]` forms a basis for an invariant 
subspace of A. Sets H[i, i-1] := 0.
"""
function detect_convergence!(H::AbstractMatrix{T}, tolerance) where {T}
    @inbounds for i = size(H, 2) : -1 : 2
        if abs(H[i, i-1]) ≤ tolerance
            H[i, i-1] = zero(T)
            return i
        end
    end

    # No convergence :(
    return 1
end

"""
Run IRAM until the eigenpair is a good enough approximation or until max_restarts has been reached
"""
function restarted_arnoldi(A::AbstractMatrix{T}, min = 5, max = 30, tolerance = 1e-5, max_restarts = 10) where {T}
    n = size(A, 1)

    arnoldi = initialize(T, n, max)
    iterate_arnoldi!(A, arnoldi, 1 : min)

    active = 1

    for restarts = 1 : max_restarts
        iterate_arnoldi!(A, arnoldi, min + 1 : max)
        implicit_restart!(arnoldi, min, max, active)

        new_active = detect_convergence!(view(arnoldi.H, 1:min+1, 1:min), tolerance)

        # Bring the new locked part oF H into upper triangular form
        if new_active > active 
            schur_form = schur(view(arnoldi.H, active : new_active - 1, active : new_active - 1))
            arnoldi.H[active : new_active - 1, active : new_active - 1] = schur_form[1]

            V_locked = view(arnoldi.V, :, active : new_active - 1)
            H_right = view(arnoldi.H, active : new_active - 1, new_active : min)

            A_mul_B!(V_locked, copy(V_locked), schur_form[2])
            Ac_mul_B!(H_right, schur_form[2], copy(H_right))
            
            if active > 1 
                H_above = view(arnoldi.H, 1 : active - 1, active : new_active - 1)
                A_mul_B!(H_above, copy(H_above), schur_form[2])
            end
            
            active = new_active
        end
        @show active
    end

    # At the end, bring the rest of H into upper triangular form as well
    schur_form = schur(view(arnoldi.H, active : min, active : min))
    arnoldi.H[active : min, active : min] = schur_form[1]

    V_locked = view(arnoldi.V, :, active : min)

    A_mul_B!(V_locked, copy(V_locked), schur_form[2])
            
    if active > 1 
        H_above = view(arnoldi.H, 1 : active - 1, active : min)
        A_mul_B!(H_above, copy(H_above), schur_form[2])
    end

    return arnoldi
end