
# VonMisesFisher comes from Distributions.jl and most of the things should be covered there
# Here we test some extra ExponentialFamily.jl specific functionality

@testitem "VonMisesFisher: vague" begin
    include("distributions_setuptests.jl")

    d = vague(VonMisesFisher, 3)

    @test typeof(d) <: VonMisesFisher
    @test mean(d) == zeros(3)
    @test params(d) == (zeros(3), 1.0e-12)
end

@testitem "VonMisesFisher: ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    for len in 3:5, b in (0.5)
        a_unnormalized = rand(len)
        a = a_unnormalized ./ norm(a_unnormalized)
        @testset let d = VonMisesFisher(a, b)
            ef = test_exponentialfamily_interface(d; option_assume_no_allocations = false, test_fisherinformation_against_jacobian = false,
                test_fisherinformation_properties = false
            )

            run_test_fisherinformation_against_jacobian(d; assume_no_allocations = false, mappings = (
                NaturalParametersSpace() => MeanParametersSpace(),
                # MeanParametersSpace() => NaturalParametersSpace(), # here is the problem for discussion, the test is broken
            ))

            for x in rand(d)
                @test @inferred(isbasemeasureconstant(ef)) === ConstantBaseMeasure()
                @test @inferred(basemeasure(ef, x)) === (1 / twoπ)^(length(x) * (1 / 2))
                @test all(@inferred(sufficientstatistics(ef, x)) .≈ (x,))
                @test @inferred(logpartition(ef)) ≈ log(besseli((len / 2) - 1, b)) - ((len / 2) - 1) * log(b)
            end
        end
    end
end

@testitem "VonMisesFisher: prod" begin
    include("distributions_setuptests.jl")

    for strategy in (ClosedProd(), PreserveTypeLeftProd(), PreserveTypeRightProd(), PreserveTypeProd(Distribution))
        @test prod(strategy, VonMisesFisher([sin(30), cos(30)], 3.0), VonMisesFisher([sin(45), cos(45)], 4.0)) ≈
              Base.convert(
            Distribution,
            prod(strategy, convert(ExponentialFamilyDistribution, VonMisesFisher([sin(30), cos(30)], 3.0)),
                convert(ExponentialFamilyDistribution, VonMisesFisher([sin(45), cos(45)], 4.0)))
        )
        @test prod(strategy, VonMisesFisher([sin(15), cos(15)], 5.0), VonMisesFisher([cos(20), sin(20)], 2.0)) ≈
              Base.convert(
            Distribution,
            prod(strategy, convert(ExponentialFamilyDistribution, VonMisesFisher([sin(15), cos(15)], 5.0)),
                convert(ExponentialFamilyDistribution, VonMisesFisher([cos(20), sin(20)], 2.0)))
        )
    end
end

@testitem "VonMisesFisher: prod with ExponentialFamilyDistribution" begin
    include("distributions_setuptests.jl")

    for μleft in eachcol(10rand(4, 4)), μright in eachcol(10rand(4, 4)), σleft in (2, 3), σright in (2, 3)
        @testset let (left, right) = (VonMisesFisher(μleft / norm(μleft), σleft), VonMisesFisher(μright / norm(μright), σright))
            @test test_generic_simple_exponentialfamily_product(
                left,
                right,
                strategies = (
                    ClosedProd(),
                    GenericProd(),
                    PreserveTypeProd(ExponentialFamilyDistribution),
                    PreserveTypeProd(ExponentialFamilyDistribution{VonMisesFisher})
                )
            )
        end
    end
end
