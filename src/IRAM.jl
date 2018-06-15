module IRAM

struct Arnoldi{T}
    V::StridedMatrix{T}
    H::StridedMatrix{T}
end

struct PartialSchur{TQ,TR} 
    Q::TQ
    R::TR
    k::Int
end

include("rotations.jl")
include("expansion.jl")
include("implicit_restart.jl")
include("factorization.jl")
include("run.jl")


end