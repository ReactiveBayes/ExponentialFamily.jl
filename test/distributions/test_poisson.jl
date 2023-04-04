module PoissonTest

using Test
using ExponentialFamily
using Random
using Distributions

import SpecialFunctions: logfactorial
import ExponentialFamily: xtlog, KnownExponentialFamilyDistribution, getnaturalparameters, basemeasure

@testset "Poisson" begin
    @testset "Constructors" begin
        @test Poisson(1) == Poisson(1)
        @test vague(Poisson) == Poisson(1e12)
    end

end
end