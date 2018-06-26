"""
Run IRAM until the eigenvectors are approximated to the prescribed tolerance or until 
`max_restarts` has been reached.
"""
function restarted_arnoldi(A::AbstractMatrix{T}, min = 5, max = 30, converged = min, ε = 1e-5, max_restarts = 10) where {T}
    n = size(A, 1)

    arnoldi = initialize(T, n, max)
    h = Vector{T}(max)
    iterate_arnoldi!(A, arnoldi, 1 : min, h)

    # min′ is the effective starting point -- may be min - 1 when the last two removed guys
    # are a complex conjugate
    min′ = min

    active = 1
    V_new = Matrix{T}(n, min)
    V_prealloc = Matrix{T}(n, min)
    for restarts = 1 : max_restarts
        iterate_arnoldi!(A, arnoldi, min′ + 1 : max, h)
        min′ = implicit_restart!(arnoldi, min, max, active, V_new)
        new_active = detect_convergence!(view(arnoldi.H, 1:min′+1, 1:min′), ε)
        # @show vecnorm(A * arnoldi.V[:, 1:min′] - arnoldi.V[:, 1:min′+1] * arnoldi.H[1:min′+1, 1:min′] )
        # Bring the new locked part oF H into upper triangular form
        if new_active > active + 1

            H22 = view(arnoldi.H, active : new_active - 1, active : new_active - 1)
            schur_form = schurfact(H22)
            H22 .= schur_form.T

            V_locked = view(V_prealloc, :, active : new_active - 1)
            V_locked = view(arnoldi.V, :, active : new_active - 1)
            H_right = view(arnoldi.H, active : new_active - 1, new_active : min′)

            A_mul_B!(V_locked, copy(V_locked), schur_form.Z)
            Ac_mul_B!(H_right, schur_form.Z, copy(H_right))
            
            if active > 1
                H_above = view(arnoldi.H, 1 : active - 1, active : new_active - 1)
                A_mul_B!(H_above, copy(H_above), schur_form.Z)
            end
        end

        active = new_active

        @show active

        if active >= converged
            break 
        end
    end

    return PartialSchur(arnoldi.V, arnoldi.H, active - 1)
end
