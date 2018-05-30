using BenchmarkTools

using IRAM: mul!, Givens, Hessenberg, shifted_qr_step!, ListOfRotations, qr!, restarted_arnoldi_2

function bencharnoldi()
    # n = 5

    # H = triu(rand(Complex128, n+1,n), -1)
    # H_new = copy(H)

    # rotations = ListOfRotations(eltype(H),n-1)

    A = rand(Complex128, 50,50)
    
    # λ = sort!(eigvals(A), by = abs, rev = true)
    # Base.showarray(STDOUT,[λs, λ],false)
    @benchmark restarted_arnoldi_2(B, 3, 20) setup = (B = copy($A))

end
