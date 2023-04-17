module MvNormalWishartTests

using Distributions, HCubature
using Test
using StatsFuns: logmvgamma
using LinearAlgebra
import ExponentialFamily: MvNormalWishart, KnownExponentialFamilyDistribution,params,dof,invscatter
import ExponentialFamily: scale, dim, getnaturalparameters, tiny, logpartition, cholinv, MvNormalMeanPrecision

using Distributions

function normal_wishart_pdf(x::Vector{Float64},
                            lambda::Matrix{Float64},
                            mu::Vector{Float64},
                            kappa::Float64,
                            nu::Float64,
                            Ψ::Matrix{Float64})  
    return pdf(MvNormalMeanPrecision(mu, kappa*lambda),x)*pdf(Wishart(nu,Ψ), lambda)
end

@testset "MvNormalWishart" begin
    @testset "common" begin
        m = rand(2)
        dist = MvNormalWishart(m, [1.0 0.0; 0.0 1.0], 0.1, 3.0)
        @test params(dist) == (m, [1.0 0.0; 0.0 1.0],  0.1, 3.0)
        @test dof(dist) == 3.0 
        @test invscatter(dist) == [1.0 0.0; 0.0 1.0]
        @test scale(dist) == 0.1
        @test dim(dist) == 2
    end

    @testset "conversions" begin
        for i=1:10, j=2:6
            m = rand(j)
            κ = rand()
            Ψ = diagm(rand(j))
            ν = 2*j+1
            dist = MvNormalWishart(m, Ψ, κ, ν)
            ef = convert(KnownExponentialFamilyDistribution, dist)

            @test getnaturalparameters(ef) ≈ [κ*m, -(1/2)*(inv(Ψ) + κ*m*m'), -κ/2, (ν-j)/2 ]
            @test invscatter(convert(Distribution, ef)) ≈ cholinv(Ψ)
            @test dof(convert(Distribution, ef)) == 2*j + 1 
        end
    end

    @testset "exponential family functions" begin
   
        for i=1:10, j=1:5,κ=0.01:1.0:5.0
            m = rand(j) 
            Ψ = m*m'+ I
            dist = MvNormalWishart(m, Ψ, κ, j+1)
            ef = convert(KnownExponentialFamilyDistribution, dist)
            pdf(dist,[m, Ψ]) ≈ normal_wishart_pdf(m,Ψ,m,κ,float(j+1) ,Ψ )
            logpdf(dist, [m,Ψ]) ≈ log(normal_wishart_pdf(m,Ψ,m,κ,float(j+1) ,Ψ ))
            
        end
    end

end

end