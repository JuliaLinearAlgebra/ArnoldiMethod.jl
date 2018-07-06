using Base.LinAlg: Givens, Rotation, givensAlgorithm

import Base: A_mul_B!, A_mul_Bc!, Ac_mul_B, A_mul_Bc, A_ldiv_B!, ctranspose, full, getindex, size
import Base.LinAlg: QRPackedQ

"""
Computes the eigenvalues of the matrix A. Assumes that A is in Schur form.
"""
function eigvalues(A::AbstractMatrix{T}; tol = eps(real(T))) where {T}
        n = size(A, 1)
        λs = Vector{complex(T)}(n)
        i = 1

        while i < n
            if abs(A[i + 1, i]) < tol*(abs(A[i + 1, i + 1]) + abs( A[i, i]))
                λs[i] =  A[i, i]
                i += 1
            else
                d =  A[i, i]*A[i + 1, i + 1] - A[i, i + 1]*A[i + 1, i]
                x = 0.5*(A[i, i] + A[i + 1, i + 1])
                y = sqrt(complex(x*x - d))
                λs[i] = x + y
                λs[i + 1] = x - y
                i += 2
            end
        end

        if i == n 
            λs[i] = A[n, n] 
        end

        return λs
    end

    # H = triu(rand(5,5), -1)
    # tau = Rotation(Base.LinAlg.Givens{Float64}[])
    # singleShiftQR!(HH::StridedMatrix, τ::Rotation, shift::Number, istart::Integer, iend::Integer)

    # function wilkinson(Hmm, t, d)
    #     λ1 = (t + sqrt(t*t - 4d))/2
    #     λ2 = (t - sqrt(t*t - 4d))/2
    #     return ifelse(abs(Hmm - λ1) < abs(Hmm - λ2), λ1, λ2)
    # end

    function schurfact!(H::AbstractMatrix{T}, active, max; tol = eps(real(T)), debug = false, maxiter = 100*size(H, 1)) where {T<:Real}
        n = size(H, 1)

        istart = active
        iend = max
        # HH = view(H, active:max, active:max)

        # istart = 1
        # iend = max - active + 1
        # HH = view(H, active:max, active:max)
        # HH_copy = copy(HH)
        Q = eye(T, max)

        # iteration count
        i = 0

        @inbounds while true
            i += 1
            if i > maxiter
                throw(ArgumentError("iteration limit $maxiter reached"))
            end

            # Determine if the matrix splits. Find lowest positioned subdiagonal "zero"
            for istart = iend - 1:-1:1
                if abs(H[istart + 1, istart]) < tol*(abs(H[istart, istart]) + abs(H[istart + 1, istart + 1]))
                    istart += 1
                    debug && @printf("Split! Subdiagonal element is: %10.3e and istart now %6d\n", H[istart, istart - 1], istart)
                    break
                elseif istart > 1 && abs(H[istart, istart - 1]) < tol*(abs(H[istart - 1, istart - 1]) + abs(H[istart, istart]))
                    debug && @printf("Split! Next subdiagonal element is: %10.3e and istart now %6d\n", H[istart, istart - 1], istart)
                    break
                end
            end

            # if block size is one we deflate
            if istart >= iend
                debug && @printf("Bottom deflation! Block size is one. New iend is %6d\n", iend - 1)
                iend -= 1

            # and the same for a 2x2 block
            elseif istart + 1 == iend
                debug && @printf("Bottom deflation! Block size is two. New iend is %6d\n", iend - 2)
                iend -= 2

            # run a QR iteration
            # shift method is specified with shiftmethod kw argument
            else
                Hmm = H[iend, iend]
                Hm1m1 = H[iend - 1, iend - 1]
                d = Hm1m1*Hmm - H[iend, iend - 1]*H[iend - 1, iend]
                t = Hm1m1 + Hmm
                t = iszero(t) ? eps(one(t)) : t # introduce a small pertubation for zero shifts
                debug && @printf("block start is: %6d, block end is: %6d, d: %10.3e, t: %10.3e\n", istart, iend, d, t)

                debug && @printf("Double shift with Wilkinson shift! Subdiagonal is: %10.3e, last subdiagonal is: %10.3e\n", H[iend, iend - 1], H[iend - 1, iend - 2])
                
                # Determine if conjugate eigenvalues
                if t*t - 4d > zero(T)
                    # Wilkinson shift
                    λ1 = (t + sqrt(t*t - 4d))/2
                    λ2 = (t - sqrt(t*t - 4d))/2
                    λ = ifelse(abs(Hmm - λ1) < abs(Hmm - λ2), λ1, λ2)
                    # Run a bulge chase
                    singleShiftQR!(H, Q, λ, istart, iend)
                else
                    # Wilkinson shift
                    λ1 = (t + sqrt(complex(t*t - 4d)))/2
                    λ2 = (t - sqrt(complex(t*t - 4d)))/2
                    λ = ifelse(abs(Hmm - λ1) < abs(Hmm - λ2), λ1, λ2)
                    # Run a bulge chase
                    doubleShiftQR!(H, Q, t, d, istart, iend) #Not done
                    # double_shift!(H, istart, iend, λ, Q)
                end
            end
            if iend <= 2 break end #Wrong
        end

        return H, Q
    end

    function schurfact!(H::AbstractMatrix{T}, active, max; tol = eps(real(T)), debug = false, maxiter = 100*size(H, 1)) where {T}
        n = size(H, 1)

        istart = active
        iend = max
        # HH = view(H, active:max, active:max)

        # istart = 1
        # iend = max - active + 1
        # HH = view(H, active:max, active:max)
        # HH_copy = copy(HH)
        Q = eye(T, max)

        # iteration count
        i = 0

        @inbounds while true
            i += 1
            if i > maxiter
                throw(ArgumentError("iteration limit $maxiter reached"))
            end

            # Determine if the matrix splits. Find lowest positioned subdiagonal "zero"
            for istart = iend - 1:-1:1
                if abs(H[istart + 1, istart]) < tol*(abs(H[istart, istart]) + abs(H[istart + 1, istart + 1]))
                    istart += 1
                    debug && @printf("Split! Subdiagonal element is: %10.3e and istart now %6d\n", H[istart, istart - 1], istart)
                    break
                elseif istart > 1 && abs(H[istart, istart - 1]) < tol*(abs(H[istart - 1, istart - 1]) + abs(H[istart, istart]))
                    debug && @printf("Split! Next subdiagonal element is: %10.3e and istart now %6d\n", H[istart, istart - 1], istart)
                    break
                end
            end

            # if block size is one we deflate
            if istart >= iend
                debug && @printf("Bottom deflation! Block size is one. New iend is %6d\n", iend - 1)
                iend -= 1

            # and the same for a 2x2 block
            elseif istart + 1 == iend
                debug && @printf("Bottom deflation! Block size is two. New iend is %6d\n", iend - 2)
                iend -= 2

            # run a QR iteration
            # shift method is specified with shiftmethod kw argument
            else
                Hmm = H[iend, iend]
                Hm1m1 = H[iend - 1, iend - 1]
                d = Hm1m1*Hmm - H[iend, iend - 1]*H[iend - 1, iend]
                t = Hm1m1 + Hmm
                t = iszero(t) ? eps(one(t)) : t # introduce a small pertubation for zero shifts
                debug && @printf("block start is: %6d, block end is: %6d, d: %10.3e, t: %10.3e\n", istart, iend, d, t)

                debug && @printf("Double shift with Wilkinson shift! Subdiagonal is: %10.3e, last subdiagonal is: %10.3e\n", H[iend, iend - 1], H[iend - 1, iend - 2])
                
                # Wilkinson shift
                λ1 = (t + sqrt(t*t - 4d))/2
                λ2 = (t - sqrt(t*t - 4d))/2
                λ = ifelse(abs(Hmm - λ1) < abs(Hmm - λ2), λ1, λ2)
                # Run a bulge chase
                singleShiftQR!(H, Q, λ, istart, iend)
            end
            if iend <= 2 break end #Wrong
        end

        return H, Q
    end

    function singleShiftQR!(HH::StridedMatrix, Q::AbstractMatrix, shift::Number, istart::Integer, iend::Integer)
        m = size(HH, 1)
        H11 = HH[istart, istart]
        H21 = HH[istart + 1, istart]
        if m > istart + 1
            Htmp = HH[istart + 2, istart]
            HH[istart + 2, istart] = 0
        end
        c, s = givensAlgorithm(H11 - shift, H21)
        G = Givens(c, s, istart)
        mul!(G, HH)
        mul!(HH, G)
        mul!(Q, G)
        for i = istart:iend - 2
            c, s = givensAlgorithm(HH[i + 1, i], HH[i + 2, i])
            G = Givens(c, s, i + 1)
            mul!(G, HH)
            HH[i + 2, i] = Htmp
            if i < iend - 2
                Htmp = HH[i + 3, i + 1]
                HH[i + 3, i + 1] = 0
            end
            mul!(HH, G)
            mul!(Q, G)
        end
        return HH
    end

    function doubleShiftQR!(HH::StridedMatrix, Q::AbstractMatrix, shiftTrace::Number, shiftDeterminant::Number, istart::Integer, iend::Integer)
        m = size(HH, 1)
        H11 = HH[istart, istart]
        H21 = HH[istart + 1, istart]
        Htmp11 = HH[istart + 2, istart]
        HH[istart + 2, istart] = 0
        if istart + 3 <= m
            Htmp21 = HH[istart + 3, istart]
            HH[istart + 3, istart] = 0
            Htmp22 = HH[istart + 3, istart + 1]
            HH[istart + 3, istart + 1] = 0
        else
            # values doen't matter in this case but variables should be initialized
            Htmp21 = Htmp22 = Htmp11
        end
        c, s, nrm = givensAlgorithm(H21*(H11 + HH[istart + 1, istart + 1] - shiftTrace, H21*HH[istart + 2, istart + 1]))
        G1 = Givens(c1, s1, istart + 1)
        c, s, _ = givensAlgorithm(H11*H11 + HH[istart, istart + 1]*H21 - shiftTrace*H11 + shiftDeterminant, nrm)
        G2 = Givens(c2, s2, istart)

        vHH = view(HH, :, istart:m)
        mul!(G1, vHH)
        mul!(G2, vHH)
        vHH = view(HH, 1:min(istart + 3, m), :)
        mul!(vHH, G1)
        mul!(vHH, G2)
        mul!(Q, G1)
        mul!(Q, G2)

        #Not sure what to do with the rest
        for i = istart:iend - 2
            for j = 1:2
                if i + j + 1 > iend break end
                # G, _ = givens(H.H,i+1,i+j+1,i)
                G, _ = givens(HH[i + 1, i], HH[i + j + 1, i], i + 1, i + j + 1)
                mul!(G, view(HH, :, i:m))
                HH[i + j + 1, i] = Htmp11
                Htmp11 = Htmp21
                # if i + j + 2 <= iend
                    # Htmp21 = HH[i + j + 2, i + 1]
                    # HH[i + j + 2, i + 1] = 0
                # end
                if i + 4 <= iend
                    Htmp22 = HH[i + 4, i + j]
                    HH[i + 4, i + j] = 0
                end
                mul!(view(HH, 1:min(i + j + 2, iend), :), G)
                # A_mul_B!(G, τ)
            end
        end
        return HH
    end