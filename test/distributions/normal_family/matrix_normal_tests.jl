@testitem "MatrixNormal: common" begin
    include("../distributions_setuptests.jl")

    @test MatrixNormal <: Distribution
    @test MatrixNormal <: ContinuousDistribution

    @test value_support(MatrixNormal) === Continuous
    @test variate_form(MatrixNormal) === Matrixvariate

    M = rand(3, 4)
    U = diagm(ones(3))
    V = diagm(ones(4))
    U = U * U'
    V = V * V'
    
    @test MatrixNormal(M, U, V) isa MatrixNormal
    @test_throws ArgumentError MatrixNormal(M, U, diagm(ones(2)))
    @test_throws ArgumentError MatrixNormal(M, diagm(ones(2)), V)
    @test_throws LinearAlgebra.PosDefException MatrixNormal(M, U, -rand(4,4))
    @test_throws LinearAlgebra.PosDefException MatrixNormal(M, -rand(3,3), V)
end

@testitem "MatrixNormal: params" begin
    include("../distributions_setuptests.jl")

    M = rand(3, 4)
    U = rand(3, 3)
    V = rand(4, 4)
    U = U * U'
    V = V * V'
    
    d = MatrixNormal(M, U, V)
    
    @test size(d) == (3, 4)
    @test params(d) == (M, U, V)
    @test mean(d) == M
    @test BayesBase.cov(d) == (U, V)
    @test BayesBase.cov(d)[1] == U
    @test BayesBase.cov(d)[2] == V
    @test Base.precision(d)[1] ≈ inv(U)
    @test Base.precision(d)[2] ≈ inv(V)
end

@testitem "MatrixNormal: entropy" begin
    include("../distributions_setuptests.jl")

    M = zeros(3, 4)
    U = rand(3, 3)
    V = rand(4, 4)
    U = U * U'
    V = V * V'
    
    d = MatrixNormal(M, U, V)

    n,p = Base.size(d)
    expected_entropy = n*p/2*log(2π) + p/2*logdet(U) + n/2*logdet(V) + n*p/2
    
    @test entropy(d) ≈ expected_entropy atol=1e-10
end

@testitem "MatrixNormal: pdf and logpdf numerical values" begin
    include("../distributions_setuptests.jl")

    # Fixed parameters for reproducibility
    M = zeros(3, 4)
    U = Matrix(I(3))
    V = Matrix(I(4))
    
    d = MatrixNormal(M, U, V)
    
    X = zeros(3, 4)
    n, p = 3, 4
    expected_logpdf = -n*p/2 * log(2π)
    expected_pdf = exp(expected_logpdf)
    
    @test isapprox(logpdf(d, X), expected_logpdf; atol=1e-10)
    @test isapprox(pdf(d, X), expected_pdf; atol=1e-10)
end

@testitem "MatrixNormal: convert to MvNormal" begin
    include("../distributions_setuptests.jl")

    M = rand(2, 3)
    U = rand(2, 2)
    V = rand(3, 3)
    U = U * U'
    V = V * V'
    d = MatrixNormal(M, U, V)

    mv_normal = convert(MvNormalMeanCovariance, d)
    
    @test mean(mv_normal) ≈ vec(M)
    @test BayesBase.cov(mv_normal) ≈ kron(U,V)
end

@testitem "MatrixNormal: natural parameters" begin
    include("../distributions_setuptests.jl")

    M = rand(3, 4)
    U = rand(3, 3)
    V = rand(4, 4)
    U = U * U'
    V = V * V'
    
    d = MatrixNormal(M, U, V)

    ef = convert(ExponentialFamilyDistribution, d)
    η = getnaturalparameters(ef)

    η1 = reshape(η[1:12], (3,4))
    η2 = reshape(η[13:21], (3,3))
    η3 = reshape(η[22:37], (4,4))
    
    @test η1  ≈ M
    @test η2 ≈ inv(U)
    @test η3 ≈ inv(V)
end

@testitem "MatrixNormal: sufficient statistics" begin
    include("../distributions_setuptests.jl")

    M = zeros(3, 4)
    U = Matrix(I(3))
    V = Matrix(I(4))
    
    d = MatrixNormal(M, U, V)
    
    rng = StableRNG(42)
    X = rand(rng, d)
    
    suff_stats = getsufficientstatistics(MatrixNormal)
    @test length(suff_stats) == 2
    
    s1 = suff_stats[1](X)
    s2 = suff_stats[2](X)
    
    @test s1 ≈ vec(X)
    @test s2 ≈ vec(X) * vec(X)'
end

@testitem "MatrixNormal: vague prior" begin
    include("../distributions_setuptests.jl")

    d_vague = BayesBase.vague(MatrixNormal, (3, 4))
    
    @test isa(d_vague, MatrixNormal)
    @test size(d_vague) == (3, 4)
    @test mean(d_vague) ≈ zeros(3, 4)
    @test BayesBase.cov(d_vague)[1] ≈ Matrix(I(3))
    @test BayesBase.cov(d_vague)[2] ≈ Matrix(I(4))
end