import AllocCheck

type_for_alloccheck(x::Type{T}) where {T} = Type{T}
type_for_alloccheck(x::Any) = typeof(x)

macro test_no_allocations(expr::Expr)
    @assert expr.head === :call "`@test_no_allocations` macro must be used with a simple function call, got $(expr)"

    fn_symbol = expr.args[1]

    @assert isa(fn_symbol, Symbol) "`@test_no_allocations` macro must be used with a simple function call, got $(expr)"

    alloc_symbol = gensym(:allocations)

    args = expr.args[2:end]
    args_types = map(arg -> :(type_for_alloccheck($arg)), args)
    result_symbol = gensym(:result)

    return esc(quote
        # precompile inside of the enclosed let block
        let 
            $expr
        end
        $alloc_symbol = @allocated($expr)
        # We first check with the standard `@allocated` to see if the function does not allocate
        # sometimes it reports spurious allocations, also depends on Julia version and OS
        # If the standard `@allocated` reports allocations, we double check with AllocCheck.jl just in case
        # The actual different between both is that one checks run-time allocations, 
        # and the other tries to statically prove that the function does not allocate
        if iszero($alloc_symbol)
            @test true
        else
            @test length(AllocCheck.check_allocs($fn_symbol, ($(args_types...),))) == 0
        end
    end)
end