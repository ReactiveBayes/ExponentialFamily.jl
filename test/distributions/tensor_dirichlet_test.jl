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

    for rank in (3, 5)
        for d in (2, 5, 10)
            for _ in 1:10
                alpha = rand([d for _ in 1:rank]...)

                distribution = TensorDirichlet(alpha)
                mat_of_dir = Dirichlet.(eachslice(alpha, dims = Tuple(2:rank)))

                mat_entropy = sum(entropy.(mat_of_dir))
                @test entropy(distribution) ≈ mat_entropy
            end
        end
    end


end


@testitem "TensorDirichlet: var" begin
    include("distributions_setuptests.jl")

    for rank in (3, 5)
        for d in (2, 5, 10)
            for _ in 1:10
                alpha = rand([d for _ in 1:rank]...)

                distribution = TensorDirichlet(alpha)
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

@testitem "TensorDirichlet: mean" begin
    include("distributions_setuptests.jl")

    for rank in (3, 5)
        for d in (2, 5, 10)
            for _ in 1:10
                alpha = rand([d for _ in 1:rank]...)

                distribution = TensorDirichlet(alpha)
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

@testitem "TensorDirichlet: std" begin
    include("distributions_setuptests.jl")

    for rank in (3, 5)
        for d in (2, 5, 10)
            for _ in 1:10
                alpha = rand([d for _ in 1:rank]...)

                distribution = TensorDirichlet(alpha)
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

@testitem "TensorDirichlet: cov" begin
    include("distributions_setuptests.jl")

    for rank in (3, 5)
        for d in (2, 5, 10)
            for _ in 1:10
                alpha = rand([d for _ in 1:rank]...)

                distribution = TensorDirichlet(alpha)
                mat_of_dir = Dirichlet.(eachslice(alpha, dims = Tuple(2:rank)))

                temp = cov.(mat_of_dir)
                old_shape = size(alpha)
                new_shape = (first(old_shape),first(old_shape),Base.tail(old_shape)...)
                mat_cov = ones(new_shape)
                for i in CartesianIndices(Base.tail(size(alpha)))
                    mat_cov[:,:, i] = temp[i]
                end
                @test cov(distribution) ≈ mat_cov
            end
        end
    end
end

@testitem "TensorDirichlet: ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    for len in 3:5
        α = rand(len, len, len) .+ 1
        let d = TensorDirichlet(α)
            ef = test_exponentialfamily_interface(d; option_assume_no_allocations = false)
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

    tensorDiri = Array{Float64, 3}(undef, (2, 2, 2))

    tensorDiri[:, 1, 1] .= a
    tensorDiri[:, 1, 2] .= b
    tensorDiri[:, 2, 1] .= c
    tensorDiri[:, 2, 2] .= d

    for space in (MeanParametersSpace(), NaturalParametersSpace())
        @test isproper(space, TensorDirichlet, tensorDiri)
        @test !isproper(space, TensorDirichlet, tensorDiri, Inf)
        tensorDiri[:, 1, 1] .= nan_test
        @test !isproper(space, TensorDirichlet, tensorDiri)
        tensorDiri[:, 1, 1] .= inf_test
        @test !isproper(space, TensorDirichlet, tensorDiri)
        tensorDiri[:, 1, 1] .= negative_num_test
        @test !isproper(space, TensorDirichlet, tensorDiri)
        tensorDiri[:, 1, 1] .= a
    end
    tensorDiri[:, 1, 1] = negative_num_natural_param_test
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
        αleft = rand(len, len, len) .+ 1
        αright = rand(len, len, len) .+ 1

        # αleft = rand(len, len) .+ 1
        # αright = rand(len, len) .+ 1
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

    for rank in (3, 5)
        for d in (2, 5, 10)
            for _ in 1:10
                alpha1 = rand([d for _ in 1:rank]...) .+ 1
                alpha2 = rand([d for _ in 1:rank]...) .+ 1
                distribution1 = TensorDirichlet(alpha1)
                distribution2 = TensorDirichlet(alpha2)
            
                mat_of_dir_1 = Dirichlet.(eachslice(alpha1, dims = Tuple(2:rank)))
                mat_of_dir_2 = Dirichlet.(eachslice(alpha2, dims = Tuple(2:rank)))
                dim = rank-1

                prod_temp = Array{Dirichlet,dim}(undef, Base.tail(size(alpha1)))
                for i in CartesianIndices(Base.tail(size(alpha1)))
                    prod_temp[i] = prod(PreserveTypeProd(Distribution),mat_of_dir_1[i],mat_of_dir_2[i])
                end
                mat_prod = similar(alpha1)
                for i in CartesianIndices(Base.tail(size(alpha1)))
                    mat_prod[:,i] = prod_temp[i].alpha
                end
                @test @inferred(prod(PreserveTypeProd(Distribution),distribution1,distribution2)) ≈ TensorDirichlet(mat_prod)
            end
        end
    end
end

@testitem "TensorDirichlet: rand" begin
    include("distributions_setuptests.jl")

    for rank in (3, 5)
        for d in (2, 5, 10)
            for _ in 1:10
                alpha = rand([d for _ in 1:rank]...)

                distribution = TensorDirichlet(alpha)
                mat_of_dir = Dirichlet.(eachslice(alpha, dims = Tuple(2:rank)))

                temp = var.(mat_of_dir)
                mat_var = similar(alpha)
                for i in CartesianIndices(Base.tail(size(alpha)))
                    mat_var[:, i] = temp[i]
                end
                @test typeof(rand(distribution)) <: Array{Float64, rank}
                @test size(rand(distribution)) == size(alpha)
                @test typeof(rand(distribution, 3)) <: AbstractVector{Array{Float64, rank}}
                @test size(rand(distribution, 3)) == (3,)
            end
        end
    end
    
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

    logPartitionDirichlet = getlogpartition(NaturalParametersSpace(), Dirichlet)
    logPartitionTensor    = getlogpartition(NaturalParametersSpace(), TensorDirichlet)
    grad                  = getgradlogpartition(NaturalParametersSpace(), Dirichlet)
    gradTensor            = getgradlogpartition(NaturalParametersSpace(), TensorDirichlet)
    info                  = getfisherinformation(NaturalParametersSpace(), Dirichlet)
    infoTensor            = getfisherinformation(NaturalParametersSpace(), TensorDirichlet)

    for rank in (3, 5)
        for d in (2, 5, 10)
            for _ in 1:10
                
                alpha = rand([d for _ in 1:rank]...)
                distribution = TensorDirichlet(alpha)
                (naturalParam,) = unpack_parameters(TensorDirichlet, alpha)
                
                mat_logPartition = sum(logPartitionDirichlet.(eachslice(alpha, dims = Tuple(2:rank))))
                mat_grad = grad.(eachslice(alpha, dims = Tuple(2:rank)))
                mat_info = sum(info.(eachslice(alpha, dims = Tuple(2:rank))))

                @test logPartitionTensor(naturalParam) ≈ mat_logPartition
                @test gradTensor(naturalParam) ≈ mat_grad
                @test infoTensor(naturalParam) ≈ mat_info
            end
        end
    end
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