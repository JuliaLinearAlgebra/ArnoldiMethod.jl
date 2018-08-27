"""
Find the largest index `i` such that `H[i, i-1] ≈ 0`. This means that `i:max` 
constitutes the active part in the Arnoldi decomp, while `V[:, 1:i-1]` forms a
basis for an invariant subspace of A. Sets H[i, i-1] := 0.
"""
function detect_convergence!(H::AbstractMatrix{T}, tolerance) where {T}
    n = size(H, 2)

    @inbounds for i = n : -1 : 2
        if abs(H[i, i-1]) ≤ tolerance
            H[i, i-1] = zero(T)
            return i
        end
    end

    # No convergence :(
    return 1
end

