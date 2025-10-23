import AllocCheck

type_for_alloccheck(x::Type{T}) where {T} = Type{T}
type_for_alloccheck(x::Any) = typeof(x)

macro test_no_allocations(expr::Expr)
    @assert expr.head === :call "`@test_no_allocations` macro must be used with a simple function call, got $(expr)"

    fn_symbol = expr.args[1]

    @assert isa(fn_symbol, Symbol) "`@test_no_allocations` macro must be used with a simple function call, got $(expr)"

    args = expr.args[2:end]
    args_types = map(arg -> :(type_for_alloccheck($arg)), args)
    result_symbol = gensym(:result)

    return esc(quote
        @test length(AllocCheck.check_allocs($fn_symbol, ($(args_types...),))) == 0
    end)
end