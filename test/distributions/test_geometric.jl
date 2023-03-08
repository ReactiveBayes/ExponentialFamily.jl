module GeometricTest

using Test
using ExponentialFamily
using Random
using Distributions

@testset "Geometric" begin
    @testset "Geometric vague" begin
        d = Geometric(0.6)

        @test Geometric() == Geometric(0.5)
        @test vague(Geometric) == Geometric(1e-12)
        @test succprob(d) == 0.6
        @test failprob(d) == 0.4
        @test probvec(d) == (0.4, 0.6)
    end

    @testset "Geometric prod" begin
        @test prod(ProdAnalytical(),Geometric(0.5),Geometric(0.6)) == Geometric(0.8)
    end

    @testset "Geometric natural parameter" begin
        @testset "natural parameter" begin
            @test naturalparams(Geometric(0.6)) ≈ log(0.4)
            @test lognormalizer(GeometricNaturalParameters(log(0.4))) == -log(0.6)
        end

        @testset "Geometric logpdf" begin
            @test logpdf(GeometricNaturalParameters(log(0.4)), 2) == logpdf(Geometric(0.6), 2)
        end

        @testset "convert" begin
            @test convert(Geometric, GeometricNaturalParameters(log(0.4))) == Geometric(0.6)
        end

        @testset "basic operations" begin
            @test GeometricNaturalParameters(log(0.5)) + GeometricNaturalParameters(log(0.4)) == GeometricNaturalParameters(log(0.5) + log(0.4))
        end

        @testset "proper" begin
            for i=1:5
                @test isproper(GeometricNaturalParameters(-i)) == true 
                @test isproper(GeometricNaturalParameters(i)) == false
            end 
        end
    end
    
end