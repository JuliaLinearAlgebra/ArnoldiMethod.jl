module IRAM

using LinearAlgebra

export partial_schur, LM, SR, LR, SI, LI, eigvalues, schur_to_eigen

struct Arnoldi{T,TV<:StridedMatrix{T},TH<:StridedMatrix{T}}
    V::TV
    H::TH

    function Arnoldi{T}(matrix_order::Int, krylov_dimension::Int) where {T}
        krylov_dimension <= matrix_order || throw(ArgumentError("Krylov dimension should be less than matrix order."))
        V = Matrix{T}(undef, matrix_order, krylov_dimension + 1)
        H = zeros(T, krylov_dimension + 1, krylov_dimension)    
        return new{T,typeof(V),typeof(H)}(V, H)
    end
end

struct PartialSchur{TQ,TR}
    Q::TQ
    R::TR
end

include("targets.jl")
include("partition.jl")
include("schurfact.jl")
include("expansion.jl")
include("implicit_restart.jl")
include("factorization.jl")
include("run.jl")
include("eigvals.jl")
include("eigenvector_uppertriangular.jl")


end