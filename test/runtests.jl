using Base.Test

# include("factorization.jl")
include("hessenberg_qr.jl")
include("givens_rotation.jl")
include("shifted_qr.jl")
include("implicit_restart.jl")
include("locked_restart.jl")
include("single_shift.jl")
# include("restarted_arnoldi.jl") # locked_restart.jl does what this used to do