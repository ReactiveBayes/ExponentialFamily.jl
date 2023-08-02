module KnownExponentialFamilyDistributionTest

using ExponentialFamily, Test, StatsFuns

import Distributions: RealInterval
import ExponentialFamily: basemeasure, sufficientstatistics, logpartition, insupport
import ExponentialFamily: getnaturalparameters, getbasemeasure, getsufficientstatistics, getlogpartition, getsupport
import ExponentialFamily: ExponentialFamilyDistributionAttributes

# import ExponentialFamily:
#     ExponentialFamilyDistribution, getnaturalparameters, getconditioner, reconstructargument!, as_vec,
#     pack_naturalparameters, unpack_naturalparameters, insupport
# import Distributions: pdf, logpdf, cdf

## ===========================================================================
## Tests fixtures
const ArbitraryExponentialFamilyAttributes = ExponentialFamilyDistributionAttributes((x) -> 1 / sum(x), [ (x) -> x, (x) -> log.(x) ], (x) -> 1 / sum(x),  RealInterval(0, Inf)) 

# Defines its own 
# - `basemeasure`
# - `sufficientstatistics`
# - `logpartition`
# - `support`
struct ArbitraryDistributionFromExponentialFamily end

ExponentialFamily.getbasemeasure(::Type{ArbitraryDistributionFromExponentialFamily}) = (η) -> 1 / sum(η)
ExponentialFamily.getsufficientstatistics(::Type{ArbitraryDistributionFromExponentialFamily}) = [ (η) -> η, (η) -> log.(η) ]
ExponentialFamily.getlogpartition(::Type{ArbitraryDistributionFromExponentialFamily}) = (η) -> 1 / sum(η)
ExponentialFamily.getsupport(::Type{ArbitraryDistributionFromExponentialFamily}) = RealInterval(0, Inf)

## ===========================================================================
## Tests

@testset "ExponentialFamilyDistributionAttributes" begin 

    # See the `ArbitraryExponentialFamilyAttributes` defined in the fixtures (above)
    @testset let attributes = ArbitraryExponentialFamilyAttributes
        @test getbasemeasure(attributes)([ 2.0 ]) ≈ 0.5
        @test getsufficientstatistics(attributes)[1]([ 2.0 ]) ≈ [ 2.0 ]
        @test getsufficientstatistics(attributes)[2]([ 2.0 ]) ≈ [ log(2.0) ]
        @test getlogpartition(attributes)([ 2.0 ]) ≈ 0.5
        @test getsupport(attributes) == RealInterval(0, Inf)
        @test insupport(attributes, 1.0)
        @test !insupport(attributes, -1.0)
    end

end

@testset "ExponentialFamilyDistribution" begin 

    # See the `ArbitraryDistributionFromExponentialFamily` defined in the fixtures (above)
    @testset let member = ExponentialFamilyDistribution(ArbitraryDistributionFromExponentialFamily, [ 2.0 ])
        η = getnaturalparameters(member)

        @test basemeasure(member) ≈ 0.5
        @test getbasemeasure(member)(η) ≈ 0.5
        @test getbasemeasure(member)([ 4.0 ]) ≈ 0.25

        @test sufficientstatistics(member) ≈ [ [ 2.0 ], [ log(2.0) ] ]
        @test map(f -> f(η), getsufficientstatistics(member)) ≈ [ [ 2.0 ], [ log(2.0) ] ]
        @test map(f -> f([ 4.0 ]), getsufficientstatistics(member)) ≈ [ [ 4.0 ], [ log(4.0) ] ]

        @test logpartition(member) ≈ 0.5
        @test getlogpartition(member)(η) ≈ 0.5
        @test getlogpartition(member)([ 4.0 ]) ≈ 0.25

        @test getsupport(member) == RealInterval(0, Inf)
        @test insupport(member, 1.0)
        @test !insupport(member, -1.0)
    end

end

# @testset "ExponentialFamilyDistribution" begin
#     ef1 = ExponentialFamilyDistribution(Bernoulli, [0.9])
#     ef2 = ExponentialFamilyDistribution(Bernoulli, [0.2])
#     @test getnaturalparameters(ef1) == [0.9]
#     @test_throws AssertionError ExponentialFamilyDistribution(Bernoulli, [0.9, 0.1])

#     @test getnaturalparameters(ef1) + getnaturalparameters(ef2) == [1.1]
#     @test getnaturalparameters(ef1) - getnaturalparameters(ef2) == [0.7]
#     (logprobability1,) = unpack_naturalparameters(ef1)
#     @test Base.convert(Bernoulli, ef1) == Bernoulli(exp(logprobability1) / (1 + exp(logprobability1)))
#     @test Base.convert(ExponentialFamilyDistribution, Bernoulli(0.9)) ==
#           ExponentialFamilyDistribution(Bernoulli, [logit(0.9)])

#     @test_throws AssertionError ExponentialFamilyDistribution(Categorical, log.([0.9, 0.1]), 2.0)
#     f = x -> x^3
#     @test_throws AssertionError ExponentialFamilyDistribution(Categorical, log.([0.9, 0.1]), f)

#     @test insupport(ef1, 1) == true
#     @test insupport(ef1, 0) == true
# end

# @testset "pdf,cdf" begin
#     ef1 = ExponentialFamilyDistribution(Bernoulli, [0.9])

#     @test logpdf(ef1, 1) ≈ logpdf(Base.convert(Bernoulli, ef1), 1)
#     @test pdf(ef1, 1) ≈ pdf(Base.convert(Bernoulli, ef1), 1)
#     @test cdf(ef1, 1) == cdf(Base.convert(Bernoulli, ef1), 1)

#     @test logpdf(ef1, 0) ≈ logpdf(Base.convert(Bernoulli, ef1), 0)
#     @test pdf(ef1, 0) ≈ pdf(Base.convert(Bernoulli, ef1), 0)
#     @test cdf(ef1, 0) == cdf(Base.convert(Bernoulli, ef1), 0)

#     @test cdf(ef1, 0.1) == cdf(Base.convert(Bernoulli, ef1), 0.1)
# end

# @testset "reconstruct arguments" begin
#     # Test case 1: reconstruct 2D array
#     A = reshape(1:6, 2, 3)
#     A_flat = as_vec(A)
#     A_recon = similar(A)
#     reconstructargument!(A_recon, A_recon, A_flat)
#     @test A == A_recon

#     # Test case 2: reconstruct 3D array
#     B = reshape(1:24, 2, 3, 4)
#     B_flat = as_vec(B)
#     B_recon = similar(B)
#     reconstructargument!(B_recon, B_recon, B_flat)
#     @test B == B_recon

#     # Test case 3: reconstruct scalar array
#     C = [1]
#     C_flat = as_vec(C)
#     C_recon = similar(C)
#     reconstructargument!(C_recon, C_recon, C_flat)
#     @test C == C_recon

#     # Test case 4: reconstruct array with different element types
#     D = [1.0, [2, 3 + 2im], [4 5; 6 1]]
#     D_flat = vcat(D[1], D[2], as_vec(D[3]))
#     D_recon = deepcopy(D)
#     reconstructargument!(D_recon, D_recon, D_flat)
#     @test D == D_recon

#     E = [rand(2, 3), rand(2, 3), rand(2)]
#     E_flat = vcat(as_vec(E[1]), as_vec(E[2]), E[3])
#     E_recon = deepcopy(E)
#     reconstructargument!(E_recon, E_recon, E_flat)
#     @test E == E_recon

#     # Test case 6: reconstruct empty array
#     F = Array{Int}(undef, 0, 3)
#     F_flat = as_vec(F)
#     F_recon = similar(F)
#     reconstructargument!(F_recon, F_recon, F_flat)
#     @test F == F_recon

#     # Test case 7: η and ηef dimensions mismatch
#     G  = [0, 0]
#     G₁ = [0, 0, 2]
#     G̃ = [1, 2, 3, 4]
#     @test_throws AssertionError reconstructargument!(G, G₁, G̃)

#     # Test case 8: ηvec does not have enough elements
#     E = [0, 0, 0]
#     Ẽ = [2, 1, 2, 3]
#     @test_throws AssertionError reconstructargument!(E, E, Ẽ)
# end

end
