@testitem "TensorDirichlet: common" begin
    include("distributions_setuptests.jl")

    @test TensorDirichlet <: Distribution
    @test TensorDirichlet <: ContinuousDistribution
    @test TensorDirichlet <: ContinuousTensorDistribution

    @test value_support(TensorDirichlet) === Continuous
    @test variate_form(TensorDirichlet) === Tensorvariate
end

@testitem "TensorDirichlet: vague" begin
    include("distributions_setuptests.jl")

    @test_throws MethodError vague(TensorDirichlet)

    d1 = vague(TensorDirichlet, 3)
    d1= vague(TensorDirichlet, 3, 5) # 3 possible outcome for input of dimensions 5 with each dimension having 3 possible value.  

    @test typeof(d1) <: TensorDirichlet
    @test mean(d1) == ones(3, 3, 3, 3, 3) ./ sum(ones(3, 3, 3, 3 ,3); dims = 1)

    d2 = vague(TensorDirichlet, 5, 3)

    @test typeof(d2) <: TensorDirichlet
    @test mean(d2) == ones(5, 5, 5) ./ sum(ones(5, 5, 5); dims = 1)

    @test vague(TensorDirichlet, 3, 3, 3) == vague(TensorDirichlet, (3, 3, 3))
    @test vague(TensorDirichlet, 4, 4, 4) == vague(TensorDirichlet, (4, 4, 4))
    @test vague(TensorDirichlet, 3, 4, 5) == vague(TensorDirichlet, (3, 4, 5))
    @test vague(TensorDirichlet, 4, 3, 2) == vague(TensorDirichlet, (4, 3, 2))
    @test vague(TensorDirichlet, 4, 3, 2, 5) == vague(TensorDirichlet, (4, 3, 2, 5))

    d3 = vague(TensorDirichlet, 3, 4, 5, 5)

    @test typeof(d3) <: TensorDirichlet
    @test mean(d3) == ones(3, 4, 5, 5) ./ sum(ones(3, 4, 5, 5); dims = 1)
end

@testitem "TensorDirichlet: entropy" begin
    include("distributions_setuptests.jl")

    a = [1.0 1.0; 1.0 1.0; 1.0 1.0]
    b = [1.2 3.3; 4.0 5.0; 2.0 1.1]
    c = [0.2 3.4; 5.0 11.0; 0.2 0.6]

    @test entropy(TensorDirichlet(cat(a,b, dims=3))) ≈ -1.3862943611198906  + (-3.1139933152617787)
    @test entropy(TensorDirichlet(cat(b,c, dims=3))) ≈ -3.1139933152617787 + (-11.444984495104693)
    @test entropy(TensorDirichlet(cat(c,a, dims=3))) ≈ -11.444984495104693 + (-1.386294361119890)
    @test entropy(TensorDirichlet(cat(a,b,c, dims=4))) ≈ -1.3862943611198906 -3.1139933152617787 -11.444984495104693
end

@testitem "TensorDirichlet: mean(::typeof(log))" begin
    include("distributions_setuptests.jl")

    import Base.Broadcast: BroadcastFunction

    a = [1.0 1.0; 1.0 1.0; 1.0 1.0]
    b = [1.2 3.3; 4.0 5.0; 2.0 1.1]
    c = [0.2 3.4; 5.0 11.0; 0.2 0.6]

    d = cat(a,b,c, dims=3)
    d2 = cat(a,b,c, dims=3)

    A = [
        -1.5000000000000002 -1.5000000000000002
        -1.5000000000000002 -1.5000000000000002
        -1.5000000000000002 -1.5000000000000002
    ]
    B = [
        -2.1920720408623637 -1.1517536610071326
        -0.646914475838374 -0.680458481634953
        -1.480247809171707 -2.6103310904778305
    ]
    C = [
        -6.879998107291004 -1.604778825293528
        -0.08484054226701443 -0.32259407259407213
        -6.879998107291004 -4.214965875553984
    ]
    D = cat(A,B,C, dims=3)
    D2 = cat(A,B,C, dims=3)

    @test mean(BroadcastFunction(log), TensorDirichlet(cat(a,b, dims=3))) ≈ cat(A,B, dims=3)
    @test mean(BroadcastFunction(log), TensorDirichlet(cat(b,c, dims=3))) ≈ cat(B,C, dims=3)
    @test mean(BroadcastFunction(log), TensorDirichlet((cat(c,a, dims=3)))) ≈ cat(C,A, dims=3)
    @test mean(BroadcastFunction(log), TensorDirichlet((cat(c,a, dims=3)))) ≈ cat(C,A, dims=3)
    @test mean(BroadcastFunction(log), TensorDirichlet((cat(d,d2,dims=4)))) ≈ cat(D,D2, dims=4)

end

@testitem "TensorDirichlet: ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    for len in 3:5
        α = rand(1.0:2.0, len, len)
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

    inf_test = [Inf Inf; Inf 1.0]
    nan_test = [NaN 2.0; 3.0 1.0]
    correctInput = [2.0, 3.0]
    negativeNumber_test = [-1.0, -1.2]

    for space in (MeanParametersSpace(), NaturalParametersSpace())
        @test !isproper(space, TensorDirichlet, cat(inf_test,correctInput,dims=3), Inf)
        @test !isproper(space, TensorDirichlet, cat(nan_test,correctInput,dims=3),)
        @test !isproper(space, TensorDirichlet, cat(negativeNumber_test,correctInput,dims=3),)
        @test !isproper(space, TensorDirichlet, cat(correctInput,correctInput,dims=3), 2.0)
        @test isproper(space, TensorDirichlet, cat(correctInput,correctInput,dims=3),)
    end

    @test_throws Exception convert(ExponentialFamilyDistribution, TensorDirichlet([Inf Inf; 2 3]))
end

@testitem "TensorDirichlet: prod with Distribution" begin
    include("distributions_setuptests.jl")

    d1 = TensorDirichlet([0.2 3.4; 5.0 11.0; 0.2 0.6])
    d2 = TensorDirichlet([1.2 3.3; 4.0 5.0; 2.0 1.1])
    d3 = TensorDirichlet([1.0 1.0; 1.0 1.0; 1.0 1.0])
    for strategy in (GenericProd(), ClosedProd(), PreserveTypeProd(Distribution), PreserveTypeLeftProd(), PreserveTypeRightProd())
        @test @inferred(prod(strategy, d1, d2)) ≈
              TensorDirichlet([0.3999999999999999 5.699999999999999; 8.0 15.0; 1.2000000000000002 0.7000000000000002])
        @test @inferred(prod(strategy, d1, d3)) ≈ TensorDirichlet(
            [0.19999999999999996 3.4000000000000004; 5.0 11.0; 0.19999999999999996 0.6000000000000001]
        )
        @test @inferred(prod(strategy, d2, d3)) ≈ TensorDirichlet([1.2000000000000002 3.3; 4.0 5.0; 2.0 1.1])
    end
end

@testitem "TensorDirichlet: prod with ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    for len in 3:6
        αleft = rand(len, len) .+ 1
        αright = rand(len, len) .+ 1
        @testset let (left, right) = (TensorDirichlet(αleft), TensorDirichlet(αright))
            @test test_generic_simple_exponentialfamily_product(
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
    @test promote_variate_type(Tensorvariate, Dirichlet) === TensorDirichlet

    @test promote_variate_type(Multivariate, TensorDirichlet) === Dirichlet
    @test promote_variate_type(Tensorvariate, TensorDirichlet) === TensorDirichlet
end

@testitem "TensorDirichlet: rand" begin
    include("distributions_setuptests.jl")

    @test_throws DimensionMismatch sum(rand(TensorDirichlet(ones(3, 5))), dims = 1) ≈ [1.0;; 1.0;; 1.0]

    @test sum(rand(TensorDirichlet(ones(3, 5, 2))), dims = 1) ≈ ones(1, 5, 2)
    @test sum(rand(TensorDirichlet(ones(5, 3, 2))), dims = 1) ≈ ones(1, 3, 2)
    @test sum(rand(TensorDirichlet(ones(5, 5, 2))), dims = 1) ≈ ones(1, 5, 2)
end
