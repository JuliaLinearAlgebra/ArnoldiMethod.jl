import Base: \
import LinearAlgebra: lu

using Base: OneTo
using LinearAlgebra: givensAlgorithm



"""
    rotate_right!(R, from, to, Q) → nothing

Rotates the diagonal elements between R[from:to,from:to]
such that the eigenvalue at R[to,to] → R[from,from]
and all the others move R[i,i] → R[i+1,i+1].

Assumes that `from` and `to` refer to the first index of the 2×2 block in the
case of complex conjugate pairs of eigenvalues.
"""
function rotate_right!(R, from::Integer, to::Integer, Q = NotWanted())
    i = to

    while i > from
        # Let's just do this check always; sometimes a block might split.
        curr_11 = is_start_of_11_block(R, i)
        prev_11 = is_end_of_11_block(R, i-1)
        j = prev_11 ? i - 1 : i - 2
        swap!(R, j, prev_11, curr_11, Q)
        i = j
    end

    nothing
end

"""
    rotate_left!(R, from, to, Q) → nothing

Rotates the diagonal elements between R[from:to,from:to]
such that the eigenvalue at R[from,from] → R[to,to]
and all the others move R[i,i] → R[i-1,i-1].

Assumes that `from` and `to` refer to the first index of the 2×2 block in the
case of complex conjugate pairs of eigenvalues.
"""
function rotate_left!(R, from::Integer, to::Integer, Q = NotWanted())
    i = from

    while true
        # Let's just do this check always; sometimes a block might split.
        curr_11 = is_start_of_11_block(R, i)
        j = curr_11 ? i + 1 : i + 2
        j > to && break
        next_11 = is_start_of_11_block(R, j)

        swap!(R, i, curr_11, next_11, Q)
        i = next_11 ? i + 1 : i + 2
    end

    nothing
end

struct CompletelyPivotedLU{T,N,TA<:SMatrix{N,N,T},TP}
    A::TA
    p::TP
    q::TP
    singular::Bool
end

struct CompletePivoting end

"""
    lu(A, CompletePivoting) → CompletelyPivotedLU(B, p, q)

Computes the LU factorization of A using complete pivoting.
"""
function lu(A::SMatrix{N,N,T}, ::Type{CompletePivoting}) where {N,T}
    # A, p and q should allocate, but escape analysis will eliminate this!
    if isbitstype(T)
        A = MMatrix(A)
    else
        A = SizedMatrix(A)
    end
    q = @MVector fill(N, N)
    p = @MVector fill(N, N)
    singular = false

    # Maybe I should consider doing this recursively.
    for k = OneTo(N - 1)
        # Find max value in sub-part.
        m, n, maxval = 1, 1, zero(real(T))
        for j = k:N, i = k:N
            if abs(A[i,j]) > maxval
                m, n, maxval = i, j, abs(A[i,j])
            end
        end

        # Store the p and q
        p[k] = m
        q[k] = n

        # Swap row and col
        for j = k:N
            A[k, j], A[m, j] = A[m, j], A[k, j]
        end

        for j = k:N
            A[j, k], A[j, n] = A[j, n], A[j, k]
        end

        Akk = A[k,k]

        # It has actually happened :(
        if iszero(Akk)
            singular = true
            break
        end

        for i = k+1:N
            A[i, k] /= Akk
        end

        for j = k+1:N
            Akj = A[k,j]

            for i = k+1:N
                A[i,j] -= A[i,k] * Akj
            end
        end
    end

    if iszero(A[N,N])
        singular = true
    end

    # Back to immutable land!
    return CompletelyPivotedLU(SMatrix(A), SVector(p), SVector(q), singular)
end

function (\)(LU::CompletelyPivotedLU{T,N}, b::SVector{N}) where {T,N}
    if isbitstype(T)
        x = MVector(b)
    else
        x = SizedVector(b)
    end

    # x ← L \ (P * b)
    for i = OneTo(N)
        x[i], x[LU.p[i]] = x[LU.p[i]], x[i]
        for j = i+1:N
            x[j] -= LU.A[j, i] * x[i]
        end
    end

    # x ← Q * (U \ x)
    for i = N:-1:1
        for j = N:-1:i+1
            x[i] -= LU.A[i, j] * x[j]
        end
        x[i] /= LU.A[i,i]
        x[i], x[LU.q[i]] = x[LU.q[i]], x[i]
    end

    # Go back to immutable land!
    SVector(x)
end

@inline sylvsystem(A::SMatrix{1,1,T}, B::SMatrix{2,2,T}) where {T} =
    @SMatrix [A[1,1]-B[1,1] -B[2,1]       ;
              -B[1,2]        A[1,1]-B[2,2]]

@inline sylvsystem(A::SMatrix{2,2,T}, B::SMatrix{1,1,T}) where {T} = 
    @SMatrix [A[1,1]-B[1,1] A[1,2]       ;
              A[2,1]        A[2,2]-B[1,1]]

@inline sylvsystem(A::SMatrix{2,2,T}, B::SMatrix{2,2,T}) where {T} =
    @SMatrix [A[1,1]-B[1,1] A[1,2]        -B[2,1]       T(0)         ;
              A[2,1]        A[2,2]-B[1,1] T(0)          -B[2,1]      ;
              -B[1,2]       T(0)          A[1,1]-B[2,2] A[1,2]       ;
              T(0)          -B[1,2]       A[2,1]        A[2,2]-B[2,2]]

"""
    sylv(A, B, C) → X, singular

Solve A * X - X * B = C for X, where A and B are 1×1 or 2×2 matrices.

It works by recasting the Sylvester equation to a linear system 
(I ⊗ A + Bᵀ ⊗ I) vec(X) = vec(C) of size 2 or 4, which is then solved by
Gaussian elimination with complete pivoting.

If the eigenvalues of A and B are equal, then `singular = true`.
"""
@inline function sylv(A::SMatrix{N,N,T}, B::SMatrix{M,M,T}, C::SMatrix{N,M,T}) where {T,N,M}
    fact = lu(sylvsystem(A, B), CompletePivoting)
    rhs = SVector{N*M,T}(C)
    SMatrix{N,M,T}(fact \ rhs), fact.singular
end

"""
    swap22_rotations(X) → c₁, s₁, c₂, s₂, c₃, s₃, c₄, s₄

Construct two sets of two Given's rotations that transform
```
  -x₁₁ -x₁₂
  -x₂₁ -x₂₂
   1    .
   .    1
```
to upper triangular form:
```
  x x  ⇒  x x  ⇒  * *  ⇒  x x  ⇒  x x
  x x  ⇒  * *  ⇒  . *  ⇒  . x  ⇒  . *
  1 .  ⇒  . *  ⇒  . x  ⇒  . *  ⇒  . .
  . 1  ⇒  . 1  ⇒  . 1  ⇒  . .  ⇒  . .
```
"""
function swap22_rotations(X::SMatrix{2,2,T}) where {T}
    # Upper triangulize first column of X
    c₁, s₁, nrm₁ = givensAlgorithm(-X[2,1], T(1))
    c₂, s₂, nrm₂ = givensAlgorithm(-X[1,1], nrm₁)

    # Apply the Givens rotations to the second column of X
    X22 = c₁ * -X[2,2] # + s₁ * 0
    X32 = -s₁' * -X[2,2] # + c₁ * 0
    # X12 = c₂ * -X[1,2] + s₂ * X22 # aint gonna need it!
    X22 = -s₂' * -X[1,2] + c₂ * X22

    # Upper triangularize the second column of X
    c₃, s₃, nrm₃ = givensAlgorithm(X32, T(1))
    c₄, s₄, nrm₄ = givensAlgorithm(X22, nrm₃)

    return c₁, s₁, c₂, s₂, c₃, s₃, c₄, s₄
end

"""
    swap12_rotations(X) → c₁, s₁, c₂, s₂

Construct two Given's rotations that transform
```
  -x₁₁ -x₁₂
   1    .
   .    1
```
to upper triangular form:
```
  x x  ⇒  * *  ⇒  x x
  1 .  ⇒  . *  ⇒  . *
  . 1  ⇒  . 1  ⇒  . .
```
"""
function swap12_rotations(X::SMatrix{1,2,T}) where {T}
    # Upper triangulize first column of X
    c₁, s₁, nrm₁₁ = givensAlgorithm(-X[1,1], T(1))

    # Apply the Givens rotations to the second column of X
    # X12 = c₁₁ * -X[1,2] # + s₁₁ * 0 # ain't gonna need it!
    X22 = -s₁' * -X[1,2] # + c₁₁ * 0

    # Upper triangularize the second column of X
    c₂, s₂, nrm₁₂ = givensAlgorithm(X22, T(1))

    return c₁, s₁, c₂, s₂
end


"""
    swap21_rotations(X) → c₁, s₁, c₂, s₂

Construct two Given's rotations that transform
```
  -x₁₁ 
  -x₂₁
   1
```
to upper triangular form:
```
  x  ⇒  x  ⇒  x
  x  ⇒  *  ⇒  .
  1  ⇒  .  ⇒  .
```
"""
function swap21_rotations(X::SMatrix{2,1,T}) where {T}
    c₁, s₁, nrm₁ = givensAlgorithm(-X[2,1], T(1))
    c₂, s₂, nrm₂ = givensAlgorithm(-X[1,1], nrm₁)
    return c₁, s₁, c₂, s₂
end

"""
    swap22!(R, i) → R

Swap a 2×2 block with a 2×2 block in R[i:i+4,i:i+4] via unitary transformations.
```
    i
    ↓
i → x x * *  ⇒  y y * *
    x x * *  ⇒  y y * *
    . . y y  ⇒  . . x x
    . . y y  ⇒  . . x x
```
Swapping means similarity transformation.
"""
function swap22!(R::AbstractMatrix{T}, i::Integer, Q = NotWanted()) where {T}
    m, n = size(R)

    @inbounds begin

        # Copy the upper triangular blocks + connection between them.
        A = @SMatrix [R[i+0,i+0] R[i+0,i+1]; R[i+1,i+0] R[i+1,i+1]]
        B = @SMatrix [R[i+2,i+2] R[i+2,i+3]; R[i+3,i+2] R[i+3,i+3]]
        C = @SMatrix [R[i+0,i+2] R[i+0,i+3]; R[i+1,i+2] R[i+1,i+3]]

        # A * X - X * B = C
        X, singular = sylv(A, B, C)

        # No need to swap if eigenvalues are indistinguishable
        singular && return R

        # Rotations that upper triangularize X
        c₁,s₁, c₂,s₂, c₃,s₃, c₄,s₄ = swap22_rotations(X)
        G₁ = Rotation3(c₁, s₁, c₂, s₂, i+0)
        G₂ = Rotation3(c₃, s₃, c₄, s₄, i+1)

        # Apply to R
        lmul!(G₁, R, i, n)
        rmul!(R, G₁, 1, i+3)
        lmul!(G₂, R, i, n)
        rmul!(R, G₂, 1, i+3)

        # Zero out things.
        R[i+2,i+0] = zero(T)
        R[i+3,i+0] = zero(T)
        R[i+2,i+1] = zero(T)
        R[i+3,i+1] = zero(T)

        # Accumulate
        rmul!(Q, G₁)
        rmul!(Q, G₂)
    end

    R
end

"""
    swap21!(R, i) → R

Swap a 1×1 block with a 2×2 block in R[i:i+3,i:i+3] via unitary transformations
```
    i
    ↓
i → x x *  ⇒  y * *
    x x *  ⇒  . x x
    . . y  ⇒  . x x
```
Swapping means similarity transformation.
"""
function swap21!(R::AbstractMatrix{T}, i::Integer, Q = NotWanted()) where {T}
    m, n = size(R)

    @inbounds begin

        # Copy the upper triangular blocks + connection between them.
        A = @SMatrix [R[i+0,i+0] R[i+0,i+1]; R[i+1,i+0] R[i+1,i+1]]
        B = @SMatrix [R[i+2,i+2]]
        C = @SMatrix [R[i+0,i+2]; R[i+1,i+2]]

        # A * X - X * B = C
        X, singular = sylv(A, B, C)

        # No need to swap if eigenvalues are indistinguishable
        singular && return R

        # Rotations that upper triangularize X
        c₁,s₁, c₂,s₂ = swap21_rotations(X)
        G₁ = Rotation3(c₁, s₁, c₂, s₂, i)

        # Apply rotations
        lmul!(G₁, R, i, n)
        rmul!(R, G₁, 1, i+2)

        # Zero out things.
        R[i+1,i+0] = zero(T)
        R[i+2,i+0] = zero(T)

        # Accumulate
        rmul!(Q, G₁)
    end
    
    R
end

"""
    swap12!(R, i) → R

Swap a 1×1 block with a 2×2 block in R[i:i+3,i:i+3] via unitary transformations
```
    i
    ↓
i → y * *  ⇒  x x *
    . x x  ⇒  x x *
    . x x  ⇒  . . y
```
Swapping means similarity transformation. 

TODO: we can optimize this swap to use just one Rotation3 rather than two
Rotation2's.
"""
function swap12!(R::AbstractMatrix{T}, i::Integer, Q = NotWanted()) where {T}
    m, n = size(R)

    @inbounds begin

        # Copy the upper triangular blocks + connection between them.
        A = @SMatrix [R[i+0,i+0]]
        B = @SMatrix [R[i+1,i+1] R[i+1,i+2]; R[i+2,i+1] R[i+2,i+2]]
        C = @SMatrix [R[i+0,i+1] R[i+0,i+2]]

        # A * X - X * B = C
        X, singular = sylv(A, B, C)

        # No need to swap if eigenvalues are indistinguishable
        singular && return R

        # Rotations that upper triangularize X
        c₁,s₁, c₂,s₂ = swap12_rotations(X)
        G₁ = Rotation2(c₁, s₁, i+0)
        G₂ = Rotation2(c₂, s₂, i+1)

        # Apply rotations
        lmul!(G₁, R, i, n)
        rmul!(R, G₁, 1, i+2)
        lmul!(G₂, R, i, n)
        rmul!(R, G₂, 1, i+2)

        # Zero out things.
        R[i+2,i+0] = zero(T)
        R[i+2,i+1] = zero(T)

        # Accumulate
        rmul!(Q, G₁)
        rmul!(Q, G₂)
    end
    
    R
end

"""
    swap11!(R, i) → R

Swap a 1×1 block with a 1×1 block in R[i:i+1,i:i+1] via unitary transformations
    i
    ↓
i → x *  ⇒  y *
    . y  ⇒  . x
"""
function swap11!(R::AbstractMatrix, i::Integer, Q = NotWanted())
    m, n = size(R)

    @inbounds begin
        R₁₁ = R[i+0,i+0]
        R₁₂ = R[i+0,i+1]
        R₂₂ = R[i+1,i+1]
        
        # Turns out the Sylvester equation is not so hard to solve in this case.
        G, = get_rotation(R₁₂, R₂₂ - R₁₁, i)
        
        # Miniscule optimization by not touching R[i:i+1,i:i+1]
        lmul!(G, R, i+2, n)
        rmul!(R, G, 1, i-1)
        R[i+0,i+0] = R₂₂
        R[i+1,i+1] = R₁₁

        # Accumulate
        rmul!(Q, G)
    end

    R
end

"""
    swap!(R, i, curr_11, next_11, Q) → nothing

Swap the two consecutive blocks of the Schur form starting at index i.
"""
function swap!(R::AbstractMatrix, i::Integer, curr_11::Bool, next_11::Bool, Q = NotWanted())
    if curr_11 
        if next_11
            swap11!(R, i, Q)
        else
            swap12!(R, i, Q)
        end
    else
        if next_11
            swap21!(R, i, Q)
        else
            swap22!(R, i, Q)
        end
    end
end

@inline is_start_of_11_block(R, i) = i == size(R, 2) || @inbounds(iszero(R[i+1,i]))
@inline is_end_of_11_block(R, i) = i == 1 || @inbounds(iszero(R[i,i-1]))
