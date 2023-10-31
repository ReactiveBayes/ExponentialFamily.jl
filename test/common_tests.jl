@testitem "dot3arg" begin
    using LinearAlgebra, ForwardDiff
    using ExponentialFamily: dot3arg

    for n in 2:10
        x = rand(n)
        y = rand(n)
        A = rand(n, n)
        @test dot3arg(x, A, y) ≈ dot(x, A, y)
        @test all(ForwardDiff.hessian((x) -> dot3arg(x, A, x), x) .!== 0)
        @test all(ForwardDiff.hessian((x) -> dot3arg(x, A, x), y) .!== 0)
    end

end