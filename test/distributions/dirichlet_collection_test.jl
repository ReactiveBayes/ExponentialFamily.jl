@testitem "DirichletCollection: common" begin
    include("distributions_setuptests.jl")

    @test DirichletCollection <: Distribution
    @test DirichletCollection <: ContinuousDistribution
    @test DirichletCollection{Float64, 3, Array{Float64, 3}} <: Distribution{ArrayLikeVariate{3}, Continuous}

    @test value_support(DirichletCollection) === Continuous
    @test variate_form(DirichletCollection{Float64, 2, Array{Float64, 2}}) === Matrixvariate
    for N in (2, 3, 4)
        @test variate_form(DirichletCollection{Float64, N, Array{Float64, N}}) === ArrayLikeVariate{N}
    end

    @test_throws "ArgumentError: All elements of the alpha tensor should be positive" DirichletCollection(zeros(3, 3, 3))
end

@testitem "DirichletCollection: entropy" begin
    include("distributions_setuptests.jl")

    # Specific value tests
    @test entropy(DirichletCollection([1.0 1.0; 1.0 1.0; 1.0 1.0])) ≈ -1.3862943611198906
    @test entropy(DirichletCollection([1.2 3.3; 4.0 5.0; 2.0 1.1])) ≈ -3.1139933152617787
    @test entropy(DirichletCollection([0.2 3.4; 5.0 11.0; 0.2 0.6])) ≈ -11.444984495104693

    # General tests
    for rank in (3, 5)
        for d in (2, 5, 10)
            for _ in 1:10
                alpha = rand([d for _ in 1:rank]...)

                distribution = DirichletCollection(alpha)
                mat_of_dir = Dirichlet.(eachslice(alpha, dims = Tuple(2:rank)))

                mat_entropy = sum(entropy.(mat_of_dir))
                @test entropy(distribution) ≈ mat_entropy
            end
        end
    end
end

@testitem "DirichletCollection: var" begin
    include("distributions_setuptests.jl")

    for rank in (3, 5)
        for d in (2, 5, 10)
            for _ in 1:10
                alpha = rand([d for _ in 1:rank]...)

                distribution = DirichletCollection(alpha)
                mat_of_dir = Dirichlet.(eachslice(alpha, dims = Tuple(2:rank)))

                temp = var.(mat_of_dir)
                mat_var = similar(alpha)
                for i in CartesianIndices(Base.tail(size(alpha)))
                    mat_var[:, i] = temp[i]
                end
                @test var(distribution) ≈ mat_var
            end
        end
    end
end

@testitem "DirichletCollection: mean" begin
    include("distributions_setuptests.jl")

    for rank in (3, 5)
        for d in (2, 5, 10)
            for _ in 1:10
                alpha = rand([d for _ in 1:rank]...)

                distribution = DirichletCollection(alpha)
                mat_of_dir = Dirichlet.(eachslice(alpha, dims = Tuple(2:rank)))

                temp = mean.(mat_of_dir)
                mat_mean = similar(alpha)
                for i in CartesianIndices(Base.tail(size(alpha)))
                    mat_mean[:, i] = temp[i]
                end
                @test mean(distribution) ≈ mat_mean
            end
        end
    end
end

@testitem "DirichletCollection: logmean" begin
    include("distributions_setuptests.jl")

    import Base.Broadcast: BroadcastFunction

    # Specific value tests
    @test mean(BroadcastFunction(log), DirichletCollection([1.0 1.0; 1.0 1.0; 1.0 1.0])) ≈ [
        -1.5000000000000002 -1.5000000000000002
        -1.5000000000000002 -1.5000000000000002
        -1.5000000000000002 -1.5000000000000002
    ]
    @test mean(BroadcastFunction(log), DirichletCollection([1.2 3.3; 4.0 5.0; 2.0 1.1])) ≈ [
        -2.1920720408623637 -1.1517536610071326
        -0.646914475838374 -0.680458481634953
        -1.480247809171707 -2.6103310904778305
    ]
    @test mean(BroadcastFunction(log), DirichletCollection([0.2 3.4; 5.0 11.0; 0.2 0.6])) ≈ [
        -6.879998107291004 -1.604778825293528
        -0.08484054226701443 -0.32259407259407213
        -6.879998107291004 -4.214965875553984
    ]

    # General tests
    for rank in (3, 5)
        for d in (2, 5, 10)
            for _ in 1:10
                alpha = rand([d for _ in 1:rank]...)

                distribution = DirichletCollection(alpha)
                mat_of_dir = Dirichlet.(eachslice(alpha, dims = Tuple(2:rank)))

                temp = mean.(Base.Broadcast.BroadcastFunction(log), mat_of_dir)
                mat_mean = similar(alpha)
                for i in CartesianIndices(Base.tail(size(alpha)))
                    mat_mean[:, i] = temp[i]
                end
                @test mean(Base.Broadcast.BroadcastFunction(log), distribution) ≈ mat_mean
            end
        end
    end
end

@testitem "DirichletCollection: std" begin
    include("distributions_setuptests.jl")

    for rank in (3, 5)
        for d in (2, 5, 10)
            for _ in 1:10
                alpha = rand([d for _ in 1:rank]...)

                distribution = DirichletCollection(alpha)
                mat_of_dir = Dirichlet.(eachslice(alpha, dims = Tuple(2:rank)))

                temp = std.(mat_of_dir)
                mat_std = similar(alpha)
                for i in CartesianIndices(Base.tail(size(alpha)))
                    mat_std[:, i] = temp[i]
                end
                @test std(distribution) ≈ mat_std
            end
        end
    end
end

@testitem "DirichletCollection: cov" begin
    include("distributions_setuptests.jl")

    for rank in (3, 5)
        for d in (2, 5, 10)
            for _ in 1:10
                alpha = rand([d for _ in 1:rank]...)

                distribution = DirichletCollection(alpha)
                mat_of_dir = Dirichlet.(eachslice(alpha, dims = Tuple(2:rank)))

                temp = cov.(mat_of_dir)
                old_shape = size(alpha)
                new_shape = (first(old_shape), first(old_shape), Base.tail(old_shape)...)
                mat_cov = ones(new_shape)
                for i in CartesianIndices(Base.tail(size(alpha)))
                    mat_cov[:, :, i] = temp[i]
                end
                @test cov(distribution) ≈ mat_cov
            end
        end
    end
end

@testitem "DirichletCollection: ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    for len in 3:5
        α = rand(len, len, len) .+ 1
        let d = DirichletCollection(α)
            ef = test_exponentialfamily_interface(d;
                option_assume_no_allocations = false,
                nsamples_for_gradlogpartition_properties = 20000)
            η1 = getnaturalparameters(ef)
            conditioner = getconditioner(ef)
            for x in [rand(1.0:2.0, len, len) for _ in 1:3]
                x = x ./ sum(x)
                @test @inferred(isbasemeasureconstant(ef)) === ConstantBaseMeasure()
                @test @inferred(basemeasure(ef, x)) === 1.0
                @test all(@inferred(sufficientstatistics(ef, x)) .≈ (map(log, x),))
                @test @inferred(logpartition(ef)) ≈ mapreduce(
                    d -> getlogpartition(NaturalParametersSpace(), Dirichlet)(convert(Vector, d)),
                    +,
                    eachcol(reshape(first(unpack_parameters(DirichletCollection, η1, conditioner)), len, len * len))
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

    dirichletCollection = Array{Float64, 3}(undef, (2, 2, 2))

    dirichletCollection[:, 1, 1] .= a
    dirichletCollection[:, 1, 2] .= b
    dirichletCollection[:, 2, 1] .= c
    dirichletCollection[:, 2, 2] .= d

    for space in (MeanParametersSpace(), NaturalParametersSpace())
        @test isproper(space, DirichletCollection, dirichletCollection)
        @test !isproper(space, DirichletCollection, Inf)
        dirichletCollection[:, 1, 1] .= nan_test
        @test !isproper(space, DirichletCollection, dirichletCollection)
        dirichletCollection[:, 1, 1] .= inf_test
        @test !isproper(space, DirichletCollection, dirichletCollection)
        dirichletCollection[:, 1, 1] .= negative_num_test
        @test !isproper(space, DirichletCollection, dirichletCollection)
        dirichletCollection[:, 1, 1] .= a
    end
    dirichletCollection[:, 1, 1] = negative_num_natural_param_test
    @test !isproper(MeanParametersSpace(), DirichletCollection, dirichletCollection)
    @test isproper(NaturalParametersSpace(), DirichletCollection, dirichletCollection)

    @test_throws Exception convert(ExponentialFamilyDistribution, DirichletCollection([Inf Inf; 2 3]))
end

@testitem "DirichletCollection: prod with Distribution" begin
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

    d1 = DirichletCollection(D1)
    d2 = DirichletCollection(D2)
    d3 = DirichletCollection(D3)

    # Test all product strategies
    for strategy in (GenericProd(), ClosedProd(), PreserveTypeProd(Distribution), PreserveTypeLeftProd(), PreserveTypeRightProd())
        @test @inferred(prod(strategy, d1, d2)) ≈
              DirichletCollection(D1 .+ D2 .- 1.0)
        @test @inferred(prod(strategy, d1, d3)) ≈ DirichletCollection(D1 .+ D3 .- 1.0)
        @test @inferred(prod(strategy, d2, d3)) ≈ DirichletCollection(D2 .+ D3 .- 1.0)
    end
end

@testitem "DirichletCollection: prod with PreserveTypeProd{Distribution}" begin
    include("distributions_setuptests.jl")

    for rank in (3, 5)
        for d in (2, 5, 10)
            for _ in 1:10
                alpha1 = rand([d for _ in 1:rank]...) .+ 1
                alpha2 = rand([d for _ in 1:rank]...) .+ 1
                distribution1 = DirichletCollection(alpha1)
                distribution2 = DirichletCollection(alpha2)

                mat_of_dir_1 = Dirichlet.(eachslice(alpha1, dims = Tuple(2:rank)))
                mat_of_dir_2 = Dirichlet.(eachslice(alpha2, dims = Tuple(2:rank)))
                dim = rank - 1

                prod_temp = Array{Dirichlet, dim}(undef, Base.tail(size(alpha1)))
                for i in CartesianIndices(Base.tail(size(alpha1)))
                    prod_temp[i] = prod(PreserveTypeProd(Distribution), mat_of_dir_1[i], mat_of_dir_2[i])
                end
                mat_prod = similar(alpha1)
                for i in CartesianIndices(Base.tail(size(alpha1)))
                    mat_prod[:, i] = prod_temp[i].alpha
                end
                @test @inferred(prod(PreserveTypeProd(Distribution), distribution1, distribution2)) ≈ DirichletCollection(mat_prod)
            end
        end
    end
end

@testitem "DirichletCollection: rand" begin
    include("distributions_setuptests.jl")
    using StableRNGs
    import Random: seed!
    rng = StableRNG(1234)

    # Specific dimension tests
    @test sum(rand(DirichletCollection(ones(3, 5))), dims = 1) ≈ ones(1, 5)
    @test sum(rand(DirichletCollection(ones(5, 3))), dims = 1) ≈ ones(1, 3)
    @test sum(rand(DirichletCollection(ones(5, 5))), dims = 1) ≈ ones(1, 5)

    # General tests
    for rank in (3, 5)
        for d in (2, 3, 4, 5)
            seed!(rng, 1234)
            alpha = rand([d for _ in 1:rank]...) .+ 2
            distribution = DirichletCollection(alpha)
            seed!(rng, 1234)
            sample = rand(rng, distribution)

            @test size(sample) == size(alpha)
            @test all(sum(sample; dims = 1) .≈ 1)

            mat_of_dir = Dirichlet.(eachslice(alpha, dims = Tuple(2:rank)))
            mat_sample = Array{Float64, rank}(undef, size(alpha))
            seed!(rng, 1234)
            for i in CartesianIndices(Base.tail(size(alpha)))
                mat_sample[:, i] = rand(rng, mat_of_dir[i])
            end

            @test sample ≈ mat_sample

            seed!(rng, 1234)
            sample = rand(rng, distribution, 10)
            @test size(sample) == (10,)
            @test(all(x -> all(sum(x; dims = 1) .≈ 1), sample))
            @test(all(x -> size(x) == size(alpha), sample))
        end
    end
end

@testitem "DirichletCollection: vague" begin
    include("distributions_setuptests.jl")

    @test_throws MethodError vague(DirichletCollection)
    @test_throws MethodError vague(DirichletCollection, 3)

    @test typeof(vague(DirichletCollection, (2, 2, 2, 3)).α) <: Array{Float64, 4}
    @test typeof(vague(DirichletCollection, (2, 2, 2, 3)).α0) <: Array{Float64, 4}
end

@testitem "DirichletCollection: logpdf" begin
    include("distributions_setuptests.jl")

    for rank in (3, 4, 5, 6)
        for d in (2, 3, 5, 10)
            for i in 1:10
                alpha = rand([d for _ in 1:rank]...)

                distribution = DirichletCollection(alpha)
                mat_of_dir = Dirichlet.(eachslice(alpha, dims = Tuple(2:rank)))

                sample = rand(distribution)
                sample ./= sum(sample, dims = 1)

                mat_logpdf = sum(logpdf.(mat_of_dir, eachslice(sample, dims = Tuple(2:rank))))
                @test logpdf(distribution, sample) ≈ mat_logpdf
                @test pdf(distribution, sample) ≈ prod(pdf.(mat_of_dir, eachslice(sample, dims = Tuple(2:rank))))
                sample = ones(size(sample))
                mat_logpdf = sum(logpdf.(mat_of_dir, eachslice(sample, dims = Tuple(2:rank))))
                @test logpdf(distribution, sample) ≈ mat_logpdf

                sample = rand(distribution, 10)
                lpdf = logpdf(distribution, sample)
                @test all(lpdf .≈ map(s -> sum(logpdf.(mat_of_dir, eachslice(s, dims = Tuple(2:rank)))), sample))
            end
        end
    end
end

@testitem "DirichletCollection: specific entropy values" begin
    include("distributions_setuptests.jl")

    @test entropy(DirichletCollection([1.0 1.0; 1.0 1.0; 1.0 1.0])) ≈ -1.3862943611198906
    @test entropy(DirichletCollection([1.2 3.3; 4.0 5.0; 2.0 1.1])) ≈ -3.1139933152617787
    @test entropy(DirichletCollection([0.2 3.4; 5.0 11.0; 0.2 0.6])) ≈ -11.444984495104693
end

@testitem "DirichletCollection: specific logmean values" begin
    include("distributions_setuptests.jl")

    import Base.Broadcast: BroadcastFunction

    @test mean(BroadcastFunction(log), DirichletCollection([1.0 1.0; 1.0 1.0; 1.0 1.0])) ≈ [
        -1.5000000000000002 -1.5000000000000002
        -1.5000000000000002 -1.5000000000000002
        -1.5000000000000002 -1.5000000000000002
    ]
    @test mean(BroadcastFunction(log), DirichletCollection([1.2 3.3; 4.0 5.0; 2.0 1.1])) ≈ [
        -2.1920720408623637 -1.1517536610071326
        -0.646914475838374 -0.680458481634953
        -1.480247809171707 -2.6103310904778305
    ]
    @test mean(BroadcastFunction(log), DirichletCollection([0.2 3.4; 5.0 11.0; 0.2 0.6])) ≈ [
        -6.879998107291004 -1.604778825293528
        -0.08484054226701443 -0.32259407259407213
        -6.879998107291004 -4.214965875553984
    ]
end

@testitem "DirichletCollection: specific rand dimension tests" begin
    include("distributions_setuptests.jl")

    @test sum(rand(DirichletCollection(ones(3, 5))), dims = 1) ≈ ones(1, 5)
    @test sum(rand(DirichletCollection(ones(5, 3))), dims = 1) ≈ ones(1, 3)
    @test sum(rand(DirichletCollection(ones(5, 5))), dims = 1) ≈ ones(1, 5)
end

@testitem "DirichletCollection: additional product strategies" begin
    include("distributions_setuptests.jl")

    d1 = DirichletCollection([0.2 3.4; 5.0 11.0; 0.2 0.6])
    d2 = DirichletCollection([1.2 3.3; 4.0 5.0; 2.0 1.1])
    d3 = DirichletCollection([1.0 1.0; 1.0 1.0; 1.0 1.0])

    for strategy in (GenericProd(), ClosedProd(), PreserveTypeProd(Distribution), PreserveTypeLeftProd(), PreserveTypeRightProd())
        @test @inferred(prod(strategy, d1, d2)) ≈
              DirichletCollection([0.3999999999999999 5.699999999999999; 8.0 15.0; 1.2000000000000002 0.7000000000000002])
        @test @inferred(prod(strategy, d1, d3)) ≈ DirichletCollection(
            [0.19999999999999996 3.4000000000000004; 5.0 11.0; 0.19999999999999996 0.6000000000000001]
        )
        @test @inferred(prod(strategy, d2, d3)) ≈ DirichletCollection([1.2000000000000002 3.3; 4.0 5.0; 2.0 1.1])
    end
end
