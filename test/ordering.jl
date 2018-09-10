using Test
using ArnoldiMethod: OrderBy, OrderReverse, OrderPerm
using Base.Order: lt

@testset "Stable permutation ordering" begin
    xs = [1+3im, 1-3im, 4]
    @testset "Forward with f = $f" for f in (real, abs)
        ord = OrderPerm(xs, OrderBy(f))
    
        # Test whether the order is stable!
        @test lt(ord, 1, 2)
        @test !lt(ord, 2, 1)
        @test lt(ord, 1, 3)
        @test lt(ord, 2, 3)
        @test sort!([1,2,3], QuickSort, ord) == [1, 2, 3]
    end

    @testset "Backward with f = $f" for f in (real, abs)
        ord = OrderPerm(xs, OrderReverse(OrderBy(f)))
        
        # Test whether the order is stable even backwards -- note that i
        @test lt(ord, 1, 2)
        @test !lt(ord, 2, 1)
        @test lt(ord, 3, 1)
        @test lt(ord, 3, 2)
        @test sort!([1,2,3], QuickSort, ord) == [3, 1, 2]
    end
end

