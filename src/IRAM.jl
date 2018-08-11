module IRAM

using LinearAlgebra

struct Arnoldi{T,TV<:StridedMatrix{T},TH<:StridedMatrix{T}}
    V::TV
    H::TH
end

struct PartialSchur{TQ,TR} 
    Q::TQ
    R::TR
end

include("targets.jl")
include("schurfact.jl")
include("expansion.jl")
include("implicit_restart.jl")
include("factorization.jl")
include("backward_substitution.jl")
include("run.jl")
include("eigvals.jl")


end