@testitem "TensorDirichlet: common" begin
    include("distributions_setuptests.jl")

    @test TensorDirichlet <: Distribution
    @test TensorDirichlet <: ContinuousDistribution
    @test TensorDirichlet <: ContinuousTensorDistribution

    @test value_support(TensorDirichlet) === Continuous
    @test variate_form(TensorDirichlet) === ArrayLikeVariate
end

@testitem "TensorDirichlet: entropy" begin
    include("distributions_setuptests.jl")

    a = [1.0, 1.0, 1.0]
    b = [1.2, 3.3]
    c = [0.2, 3.4]
    d = [4.0, 5.0]
    e = [5.0, 11.0]
    f = [0.2, 0.6]
    g = [2.0, 1.1]

    tensorA = Array{Float64, 3}(undef, (3, 2, 3))
    for i in CartesianIndices(Base.tail(size(tensorA)))
        tensorA[:, i] = a
    end

    tensorB = Array{Float64, 3}(undef, (2, 2, 3))
    tensorB[:, 1, 1] = g
    tensorB[:, 1, 2] = b
    tensorB[:, 1, 3] = c
    tensorB[:, 2, 1] = d
    tensorB[:, 2, 2] = e
    tensorB[:, 2, 3] = f

    @test entropy(TensorDirichlet(tensorA)) ≈ -log(2) * 6
    @test entropy(TensorDirichlet(tensorB)) ≈ mapreduce(x -> entropy(Dirichlet(x)), +, [b, c, d, e, f, g])
end

@testitem "TensorDirichlet: mean(::typeof(log))" begin
    # mean log is now part of the closeForm package and the implementation of this should no more be part of this

end

@testitem "TensorDirichlet: var" begin
    include("distributions_setuptests.jl")

    a = [1.0, 1.0]
    b = [1.2, 3.3]
    c = [0.2, 3.4]
    d = [4.0, 5.0]

    tensorDiri = Array{Float64, 3}(undef, (2, 2, 2))

    tensorDiri[:, 1, 1] = a
    tensorDiri[:, 1, 2] = b
    tensorDiri[:, 2, 1] = c
    tensorDiri[:, 2, 2] = d

    varDiri = Array{Float64, 3}(undef, (2, 2, 2))

    varDiri[:, 1, 1] = var(Dirichlet(a))
    varDiri[:, 1, 2] = var(Dirichlet(b))
    varDiri[:, 2, 1] = var(Dirichlet(c))
    varDiri[:, 2, 2] = var(Dirichlet(d))
    @show var(TensorDirichlet(tensorDiri))
    @show varDiri
    @test var(TensorDirichlet(tensorDiri)) == varDiri
end

@testitem "TensorDirichlet: cov" begin
    include("distributions_setuptests.jl")

    a = [1.0, 1.0]
    b = [1.2, 3.3]
    c = [0.2, 3.4]
    d = [4.0, 5.0]

    tensorDiri = Array{Float64, 3}(undef, (2, 2, 2))

    tensorDiri[:, 1, 1] = a
    tensorDiri[:, 1, 2] = c
    tensorDiri[:, 2, 1] = b
    tensorDiri[:, 2, 2] = d

    covTensorDiri = Matrix{Matrix{Float64}}(undef, (2, 2))
    for i in eachindex(covTensorDiri)
        covTensorDiri[i] = Matrix{Float64}(undef, (2, 2))
    end

    covTensorDiri[1] = cov(Dirichlet(a))
    covTensorDiri[2] = cov(Dirichlet(b))
    covTensorDiri[3] = cov(Dirichlet(c))
    covTensorDiri[4] = cov(Dirichlet(d))

    @test cov(TensorDirichlet(tensorDiri)) == covTensorDiri
end

@testitem "TensorDirichlet: ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    for len in 3:5
        α = Array{Float64, 3}(undef, (len, len, len))
        for i in eachindex(CartesianIndices(Base.tail(size(α))))
            α[:, i] = rand(len) .+ 1
        end
        @testset let d = TensorDirichlet(α)
            ef = test_exponentialfamily_interface(d; test_basic_functions = true, option_assume_no_allocations = false)
            η1 = getnaturalparameters(ef)
            for x in [rand(1.0:2.0, len, len) for _ in 1:3]
                x = x ./ sum(x)
                @test @inferred(isbasemeasureconstant(ef)) === ConstantBaseMeasure()
                @test @inferred(basemeasure(ef, x)) === 1.0
                @test all(@inferred(sufficientstatistics(ef, x)) .≈ (map(log, x),))
                @test @inferred(logpartition(ef)) ≈ mapreduce(
                    d -> getlogpartition(NaturalParametersSpace(), Dirichlet)(convert(Vector, d)),
                    +,
                    eachcol(first(unpack_parameters(TensorDirichlet, η1)))
                )
            end
        end
    end

    inf_test = [Inf, 1.0]
    nan_test = [NaN, 2.0]
    negative_num_test = [-1.0, -1.2]
    negative_num_natural_param_test = [-0.1, -0.2]
    a = [1.0, 1.0]
    b = [1.2, 3.3]
    c = [0.2, 3.4]
    d = [4.0, 5.0]

    tensorDiri = Matrix{Vector{Float64}}(undef, (2, 2))
    for i in eachindex(tensorDiri)
        tensorDiri[i] = Vector{Float64}(undef, 2)
    end

    tensorDiri[1] = a
    tensorDiri[2] = b
    tensorDiri[3] = c
    tensorDiri[4] = d

    for space in (MeanParametersSpace(), NaturalParametersSpace())
        @test isproper(space, TensorDirichlet, tensorDiri)
        @test !isproper(space, TensorDirichlet, tensorDiri, Inf)
        tensorDiri[1] = nan_test
        @test !isproper(space, TensorDirichlet, tensorDiri)
        tensorDiri[1] = inf_test
        @test !isproper(space, TensorDirichlet, tensorDiri)
        tensorDiri[1] = negative_num_test
        @test !isproper(space, TensorDirichlet, tensorDiri)
        tensorDiri[1] = a
    end
    tensorDiri[1] = negative_num_natural_param_test
    @test !isproper(MeanParametersSpace(), TensorDirichlet, tensorDiri)
    @test isproper(NaturalParametersSpace(), TensorDirichlet, tensorDiri)

    @test_throws Exception convert(ExponentialFamilyDistribution, TensorDirichlet([Inf Inf; 2 3]))
end

@testitem "TensorDirichlet: prod with Distribution" begin
    include("distributions_setuptests.jl")

    a = [1.0, 1.0]
    b = [1.2, 3.3]
    c = [0.2, 3.4]
    d = [4.0, 5.0]
    e = [5.0, 11.0]
    f = [0.2, 0.6]
    g = [2.0, 1.1]

    D1 = Array{Float64, 2}(undef, (2, 3))
    D1[:, 1] = c
    D1[:, 2] = e
    D1[:, 3] = f

    D2 = Array{Float64, 2}(undef, (2, 3))
    D2[:, 1] = b
    D2[:, 2] = d
    D2[:, 3] = g

    D3 = Array{Float64, 2}(undef, (2, 3))
    D3[:, 1] = D3[:, 2] = D3[:, 3] = a

    d1 = TensorDirichlet(D1)
    d2 = TensorDirichlet(D2)
    d3 = TensorDirichlet(D3)
    @test @inferred(
        prod(PreserveTypeProd(Distribution), d1, d2) ≈ TensorDirichlet([0.3999999999999999 8.0 1.2000000000000002; 5.699999999999999 15.0 0.7000000000000002])
    )
    @test @inferred(prod(PreserveTypeProd(Distribution), d1, d3)) ≈ TensorDirichlet(
        [0.19999999999999996 5.0 0.19999999999999996; 3.4000000000000004 11.0 0.6000000000000001]
    )
    @test @inferred(prod(PreserveTypeProd(Distribution), d2, d3)) ≈ TensorDirichlet([1.2000000000000002 4.0 2.0; 3.3 5.0 1.1])
end

@testitem "TensorDirichlet: prod with ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    for len in 3:6
        αleft = Matrix{Vector{Float64}}(undef, (len, len))
        for i in eachindex(αleft)
            αleft[i] = Vector{Float64}(undef, len)
            αleft[i] = rand(len) .+ 1
        end

        αright = Matrix{Vector{Float64}}(undef, (len, len))
        for i in eachindex(αright)
            αright[i] = Vector{Float64}(undef, len)
            αright[i] = rand(len) .+ 1
        end

        # αleft = rand(len, len) .+ 1
        # αright = rand(len, len) .+ 1
        @show αleft
        @testset let (left, right) = (TensorDirichlet(αleft), TensorDirichlet(αright))
            @test_broken test_generic_simple_exponentialfamily_product(
                left,
                right,
                strategies = (
                    ClosedProd(),
                    GenericProd()
                )
            )
        end
    end
end

@testitem "TensorDirichlet: promote_variate_type" begin
    include("distributions_setuptests.jl")

    @test_throws MethodError promote_variate_type(Univariate, TensorDirichlet)

    @test promote_variate_type(Multivariate, Dirichlet) === Dirichlet
    @test promote_variate_type(ArrayLikeVariate, Dirichlet) === TensorDirichlet

    @test promote_variate_type(Multivariate, TensorDirichlet) === TensorDirichlet
end

@testitem "TensorDirichlet: prod with PreserveTypeProd{Distribution}" begin
    include("distributions_setuptests.jl")

    tensorDiri = Matrix{Vector{Float64}}(undef, (2, 2))
    for i in eachindex(tensorDiri)
        tensorDiri[i] = Vector{Float64}(undef, 2)
        tensorDiri[i] = ones(2) .* 2
    end

    result = Matrix{Vector{Float64}}(undef, (2, 2))
    for i in eachindex(tensorDiri)
        result[i] = Vector{Float64}(undef, 2)
        result[i] = ones(2) .* 3
    end

    @test_broken @inferred(prod(PreserveTypeProd(Distribution), TensorDirichlet(tensorDiri), TensorDirichlet(tensorDiri))) == TensorDirichlet(result)
end

@testitem "TensorDirichlet: rand" begin
    include("distributions_setuptests.jl")

    a = [1.0, 1.0]
    b = [1.2, 3.3]
    c = [0.2, 3.4]
    d = [4.0, 5.0]

    tensorDiri = Array{Float64, 3}(undef, (2, 2, 2))

    tensorDiri[:, 1, 1] = a
    tensorDiri[:, 1, 2] = b
    tensorDiri[:, 2, 1] = c
    tensorDiri[:, 2, 2] = d

    @test typeof(rand(TensorDirichlet(tensorDiri))) <: Array{Float64, 3}
    @test size(rand(TensorDirichlet(tensorDiri))) == (2, 2, 2)
    @test typeof(rand(TensorDirichlet(tensorDiri), 3)) <: AbstractVector{Array{Float64, 3}}
    @test size(rand(TensorDirichlet(tensorDiri), 3)) == (3,)
end

@testitem "TensorDirichlet: vague" begin
    include("distributions_setuptests.jl")

    dirichlet = vague(TensorDirichlet, 3)
    @test typeof(dirichlet.a) <: Array{Float64, 2}
    @test size(dirichlet.a) == (3, 3)

    @test typeof(vague(TensorDirichlet, (2, 2, 2, 3)).a) <: Array{Float64, 4}

    @test vague(TensorDirichlet, 3) == vague(TensorDirichlet, (3, 3))

    @test_throws MethodError vague(TensorDirichlet)
end

@testitem "TensorDirichlet: NaturalParametersSpace" begin
    include("distributions_setuptests.jl")

    a = [1.0, 1.0]
    b = [1.2, 3.3]
    c = [0.2, 3.4]
    d = [4.0, 5.0]

    tensorDiri = Matrix{Vector{Float64}}(undef, (2, 2))
    for i in eachindex(tensorDiri)
        tensorDiri[i] = Vector{Float64}(undef, 2)
    end

    tensorDiri[1] = a
    tensorDiri[2] = b
    tensorDiri[3] = c
    tensorDiri[4] = d

    logPartitionDirichlet = getlogpartition(NaturalParametersSpace(), Dirichlet)

    @test getlogpartition(NaturalParametersSpace(), TensorDirichlet)(tensorDiri) ==
          logPartitionDirichlet(a) + logPartitionDirichlet(b) + logPartitionDirichlet(c) + logPartitionDirichlet(d)

    gradLogPartition = Matrix{Vector{Float64}}(undef, (2, 2))
    for i in eachindex(gradLogPartition)
        gradLogPartition[i] = Vector{Float64}(undef, 2)
    end

    grad = getgradlogpartition(NaturalParametersSpace(), Dirichlet)

    gradLogPartition[1] = grad(a)
    gradLogPartition[2] = grad(b)
    gradLogPartition[3] = grad(c)
    gradLogPartition[4] = grad(d)

    @test getgradlogpartition(NaturalParametersSpace(), TensorDirichlet)(tensorDiri) == gradLogPartition

    info = getfisherinformation(NaturalParametersSpace(), Dirichlet)

    @test getfisherinformation(NaturalParametersSpace(), TensorDirichlet)(tensorDiri) == info(a) + info(b) + info(c) + info(d)
end

@testitem "TensorDirichlet: logpdf" begin
    include("distributions_setuptests.jl")

    for rank in (3, 5)
        for d in (2, 5, 10)
            for _ in 1:10
                alpha = rand([d for _ in 1:rank]...)

                distribution = TensorDirichlet(alpha)
                mat_of_dir = Dirichlet.(eachslice(alpha, dims = Tuple(2:rank)))

                sample = rand(distribution)
                sample ./= sum(sample, dims = 1)

                mat_logpdf = sum(logpdf.(mat_of_dir, eachslice(sample, dims = Tuple(2:rank))))
                @test logpdf(distribution, sample) ≈ mat_logpdf
            end
        end
    end
end