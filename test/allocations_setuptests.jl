import AllocCheck

type_for_alloccheck(x::Type{T}) where {T} = Type{T}
type_for_alloccheck(x::Any) = typeof(x)

macro test_no_allocations(expr::Expr)
    @assert expr.head === :call "`@test_no_allocations` macro must be used with a simple function call, got $(expr)"

    fn_symbol = expr.args[1]

    @assert isa(fn_symbol, Symbol) "`@test_no_allocations` macro must be used with a simple function call, got $(expr)"

    expr_lambda = gensym(:expr_lambda)
    alloc_symbol = gensym(:allocations)

    args = expr.args[2:end]
    args_types = map(arg -> :(type_for_alloccheck($arg)), args)
    result_symbol = gensym(:result)

    gcstats0_symbol = gensym(:gstats0)
    gcstats1_symbol = gensym(:gstats1)
    gcstats_diff_symbol = gensym(:gstats_diff)
    gcstats_allocd_symbol = gensym(:gcstats_allocd)

    return esc(quote
        # precompile inside of the enclosed let block
        $expr_lambda = () -> $expr
        let
            Core.donotdelete($expr)
            Core.donotdelete(($expr_lambda)())
        end
        # We first check with the standard `@allocated` to see if the function does not allocate
        # sometimes it reports spurious allocations, also depends on Julia version and OS
        $alloc_symbol = min(@allocated($expr), @allocated(($expr_lambda)()))
        
        # Then we try to use internal Base.GC_Diff, since this is what popular 
        # benchmarking packages are using
        $gcstats_allocd_symbol = begin 
            $gcstats0_symbol = Base.gc_num()
            Core.donotdelete($expr)
            $gcstats1_symbol = Base.gc_num()
            $gcstats_diff_symbol = Base.GC_Diff($gcstats1_symbol, $gcstats0_symbol)
            ($gcstats_diff_symbol).allocd # This is internal to GC_Diff
        end

        if iszero($alloc_symbol)
            @test true
        elseif iszero($gcstats_allocd_symbol)
            @test true
        else
            # If the standard `@allocated` and GC_Diff reports allocations, we double check with AllocCheck.jl just in case
            # The actual different between both is that one checks run-time allocations, 
            # and the other tries to statically prove that the function does not allocate
            @test length(AllocCheck.check_allocs($fn_symbol, ($(args_types...),))) == 0
        end
    end)
end
