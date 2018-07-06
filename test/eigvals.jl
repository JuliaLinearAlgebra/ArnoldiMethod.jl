using Base.Test

using IRAM

@testset "schurfact!" begin
    T = Float64

    # Conjugate pair should converge in 1 iteration.
    let H = [1.0 2.0; -3.0 4.0], H_copy = copy(H)
        @test IRAM.schurfact!(H, 1, 2, maxiter = 1)
    end

    # Upper triangular should converge in at most 1 or 2 (?) iterations
    let H = [2.0 0.5; 0.0 3.0], H_copy = copy(H)
        @test IRAM.schurfact!(H, 1, 2, maxiter = 2)
    end

    # Distinct real eigenvalues should converge in 1 or 2 (?) iterations because exact shifts
    let H = [2.0 0.5; 0.2 3.0], H_copy = copy(H)
        @test IRAM.schurfact!(H, 1, 2, maxiter = 2)
    end
end