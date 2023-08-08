module LaplaceTest

using ExponentialFamily, Distributions
using Test, ForwardDiff, Random, StatsFuns, StableRNGs

import ExponentialFamily:
    ExponentialFamilyDistribution, getnaturalparameters, compute_logscale, logpartition, basemeasure, insupport,
    sufficientstatistics, fisherinformation, pack_parameters, unpack_parameters, isbasemeasureconstant,
    ConstantBaseMeasure, MeanToNatural, NaturalToMean, NaturalParametersSpace, default_prod_rule

@testset "Laplace" begin

    # Laplace comes from Distributions.jl and most of the things should be covered there
    # Here we test some extra ExponentialFamily.jl specific functionality

    @testset "vague" begin
        d = vague(Laplace)

        @test typeof(d) <: Laplace
        @test mean(d) === 0.0
        @test params(d) === (0.0, 1e12)
    end

    @testset "prod with Distribution" begin 
        @test default_prod_rule(Laplace, Laplace) === PreserveTypeProd(Laplace)

        @test @inferred(prod(PreserveTypeProd(Laplace), Laplace(0.0, 0.5), Laplace(0.0, 0.5))) ≈ Laplace(0.0, 0.25)
        @test @inferred(prod(PreserveTypeProd(Laplace), Laplace(1.0, 1.0), Laplace(1.0, 1.0))) ≈ Laplace(1.0, 0.5)
        @test @inferred(prod(PreserveTypeProd(Laplace), Laplace(2.0, 3.0), Laplace(2.0, 7.0))) ≈ Laplace(2.0, 2.1)

        # GenericProd should always check the default strategy and fallback if available
        @test @inferred(prod(GenericProd(), Laplace(0.0, 0.5), Laplace(0.0, 0.5))) ≈ Laplace(0.0, 0.25)
        @test @inferred(prod(GenericProd(), Laplace(1.0, 1.0), Laplace(1.0, 1.0))) ≈ Laplace(1.0, 0.5)
        @test @inferred(prod(GenericProd(), Laplace(2.0, 3.0), Laplace(2.0, 7.0))) ≈ Laplace(2.0, 2.1)

        # Different location parameters cannot be compute a closed prod with the same type
        @test_throws Exception prod(PreserveTypeProd(Laplace), Laplace(0.0, 0.5), Laplace(0.01, 0.5)) 
        @test_throws Exception prod(PreserveTypeProd(Laplace), Laplace(1.0, 0.5), Laplace(-1.0, 0.5)) 
    end

    @testset "prod with ExponentialFamilyDistribution" begin 
        @test default_prod_rule(ExponentialFamilyDistribution{Laplace}, ExponentialFamilyDistribution{Laplace}) === PreserveTypeProd(ExponentialFamilyDistribution{Laplace})

        for location in (0.0, 1.0), sleft in 0.1:0.1:0.9, sright in 0.1:0.1:0.9
            efleft = @inferred(convert(ExponentialFamilyDistribution, Laplace(location, sleft)))
            efright = @inferred(convert(ExponentialFamilyDistribution, Laplace(location, sright)))

            ηleft = @inferred(getnaturalparameters(efleft))
            ηright = @inferred(getnaturalparameters(efright))

            for strategy in (PreserveTypeProd(ExponentialFamilyDistribution{Laplace}), GenericProd())
                @test @inferred(prod(strategy, efleft, efright)) == ExponentialFamilyDistribution(Laplace, ηleft + ηright, location)
            end

            @test @inferred(prod!(similar(efleft), efleft, efright)) ==
              ExponentialFamilyDistribution(Laplace, ηleft + ηright, location)

            let _similar = similar(efleft)
                @test @allocated(prod!(_similar, efleft, efright)) === 0
            end

            @test @inferred(prod(PreserveTypeProd(Laplace), efleft, efright)) ≈
                prod(PreserveTypeProd(Laplace), Laplace(location, sleft), Laplace(location, sright))
        end

        # Different location parameters cannot be compute a closed prod with the same type
        @test_throws Exception prod(
            PreserveTypeProd(ExponentialFamilyDistribution{Laplace}), 
            convert(ExponentialFamilyDistribution, Laplace(0.0, 0.5)), 
            convert(ExponentialFamilyDistribution, Laplace(0.01, 0.5))
        ) 
        @test_throws Exception prod(
            PreserveTypeProd(ExponentialFamilyDistribution{Laplace}), 
            convert(ExponentialFamilyDistribution, Laplace(1.0, 0.5)), 
            convert(ExponentialFamilyDistribution, Laplace(-1.0, 0.5))
        ) 
    end

    # @testset "prod" begin
    #     for i in 1:100
    #         μleft = 100 * randn()
    #         μright = 100 * randn()
    #         σleft = 100 * rand()
    #         σright = 100 * rand()
    #         l_left = Laplace(μleft, σleft)
    #         l_right = Laplace(μleft, σright)
    #         l_right2 = Laplace(μright, σright)
    #         ef_left = convert(ExponentialFamilyDistribution, l_left)
    #         ef_right = convert(ExponentialFamilyDistribution, l_right)
    #         ef_right2 = convert(ExponentialFamilyDistribution, l_right2)
    #         (η_left, conditioner_left) = (getnaturalparameters(ef_left), getconditioner(ef_left))
    #         (η_right, conditioner_right) = (getnaturalparameters(ef_right), getconditioner(ef_right))
    #         (η_right2, conditioner_right2) = (getnaturalparameters(ef_right2), getconditioner(ef_right2))
    #         @test prod(ef_left, ef_right) ==
    #               ExponentialFamilyDistribution(Laplace, η_left + η_right, conditioner_left)
    #         @test prod(ClosedProd(), l_left, l_right) ≈ convert(Distribution, prod(ef_left, ef_right))

    #         basemeasure = (x) -> 1.0
    #         sufficientstatistics = (x) -> [abs(x - conditioner_left), abs(x - conditioner_right2)]
    #         sorted_conditioner = sort([conditioner_left, conditioner_right2])
    #         function logpartition(η)
    #             A1 = exp(η[1] * conditioner_left + η[2] * conditioner_right2)
    #             A2 = exp(-η[1] * conditioner_left + η[2] * conditioner_right2)
    #             A3 = exp(-η[1] * conditioner_left - η[2] * conditioner_right2)
    #             B1 = (exp(sorted_conditioner[2] * (-η[1] - η[2])) - 1.0) / (-η[1] - η[2])
    #             B2 =
    #                 (exp(sorted_conditioner[1] * (η[1] - η[2])) - exp(sorted_conditioner[2] * (η[1] - η[2]))) /
    #                 (η[1] - η[2])
    #             B3 = (1.0 - exp(sorted_conditioner[1] * (η[1] + η[2]))) / (η[1] + η[2])

    #             return log(A1 * B1 + A2 * B2 + A3 * B3)
    #         end
    #         naturalparameters = vcat(η_left, η_right2)
    #         supp = support(l_left)
    #         dist_prod = prod(ClosedProd(), l_left, l_right2)
    #         ef_prod = prod(ef_left, ef_right2)
    #         @test getnaturalparameters(dist_prod) == naturalparameters
    #         @test getsupport(dist_prod) == supp
    #         @test getbasemeasure(dist_prod)(1.0) == basemeasure(1.0)
    #         @test getsufficientstatistics(dist_prod)(1.0) ==
    #               sufficientstatistics(1.0)

    #         @test getnaturalparameters(ef_prod) == naturalparameters
    #         @test getsupport(ef_prod) == supp
    #         @test getbasemeasure(ef_prod)(1.0) == basemeasure(1.0)
    #         @test getsufficientstatistics(ef_prod)(1.0) ==
    #               sufficientstatistics(1.0)
    #     end
    # end

    # @testset "natural parameters related" begin
    #     @testset "convert" begin
    #         for i in 1:10
    #             @test convert(Distribution, ExponentialFamilyDistribution(Laplace, [-i], 2.0)) ==
    #                   Laplace(2.0, 1 / i)

    #             @test convert(ExponentialFamilyDistribution, Laplace(sqrt(i), i)) ==
    #                   ExponentialFamilyDistribution(Laplace, [-1 / i], sqrt(i))
    #         end
    #     end

    #     @testset "logpartition" begin
    #         @test logpartition(ExponentialFamilyDistribution(Laplace, [-1.0], 1.0)) ≈ log(2)
    #         @test logpartition(ExponentialFamilyDistribution(Laplace, [-2.0], 1.0)) ≈ log(1)
    #     end

    #     @testset "logpdf" begin
    #         for i in 1:10
    #             @test logpdf(ExponentialFamilyDistribution(Laplace, [-i], 0.0), 0.01) ≈
    #                   logpdf(Laplace(0.0, 1 / i), 0.01)
    #             @test logpdf(ExponentialFamilyDistribution(Laplace, [-i], 1.0), 0.5) ≈
    #                   logpdf(Laplace(1.0, 1 / i), 0.5)
    #         end
    #     end

    #     @testset "isproper" begin
    #         for i in 1:10
    #             @test isproper(ExponentialFamilyDistribution(Laplace, [-i], 1.0)) === true
    #             @test isproper(ExponentialFamilyDistribution(Laplace, [i], 2.0)) === false
    #         end
    #     end

    #     @testset "basemeasure" begin
    #         for (i) in (1:10)
    #             @test basemeasure(ExponentialFamilyDistribution(Laplace, [-i], 1.0), i^2) == 1.0
    #         end
    #     end

    #     @testset "fisher information" begin
    #         for λ in 1:10, u in 1.0:0.5:5.0
    #             dist = Laplace(u, λ)
    #             ef = convert(ExponentialFamilyDistribution, dist)
    #             η = getnaturalparameters(ef)
    #             transformation(η) = [u, -inv(η[1])]
    #             f_logpartition = (η) -> logpartition(ExponentialFamilyDistribution(Laplace, η, getconditioner(ef)))
    #             autograd_information = (η) -> ForwardDiff.hessian(f_logpartition, η)
    #             @test first(fisherinformation(ef)) ≈ first(autograd_information(η)) atol = 1e-8
    #             J = ForwardDiff.jacobian(transformation, η)
    #             @test first(J' * fisherinformation(dist) * J) ≈ first(fisherinformation(ef)) atol = 1e-8
    #         end
    #     end
    # end

    # @testset "ExponentialFamilyDistribution mean,var" begin
    #     for λ in 1:10, u in 1.0:0.5:5.0
    #         dist = Laplace(u, λ)
    #         ef = convert(ExponentialFamilyDistribution, dist)
    #         @test mean(dist) ≈ mean(ef) atol = 1e-8
    #         @test var(dist) ≈ var(ef) atol = 1e-8
    #     end
    # end
end

end
