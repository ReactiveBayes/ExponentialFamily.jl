@testitem "Multinomial: probvec" begin
    include("distributions_setuptests.jl")

    @test probvec(Multinomial(5, [1 / 3, 1 / 3, 1 / 3])) == [1 / 3, 1 / 3, 1 / 3]
    @test probvec(Multinomial(3, [0.2, 0.2, 0.4, 0.1, 0.1])) == [0.2, 0.2, 0.4, 0.1, 0.1]
    @test probvec(Multinomial(2, [0.5, 0.5])) == [0.5, 0.5]
end

@testitem "Multinomial: vague" begin
    include("distributions_setuptests.jl")

    @test_throws MethodError vague(Multinomial)
    @test_throws MethodError vague(Multinomial, 4)

    vague_dist1 = vague(Multinomial, 5, 4)
    @test typeof(vague_dist1) <: Multinomial
    @test probvec(vague_dist1) == [1 / 4, 1 / 4, 1 / 4, 1 / 4]

    vague_dist2 = vague(Multinomial, 3, 5)
    @test typeof(vague_dist2) <: Multinomial
    @test probvec(vague_dist2) == [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]
end

@testitem "Multinomial: test_EF_interface" begin
    include("distributions_setuptests.jl")

    ef = vague(Multinomial, 5, 4)

    test_exponentialfamily_interface(ef)
end