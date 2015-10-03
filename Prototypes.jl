
module Prototypes

export @protofun


# the following macro generates "function defaults" which throw an error message
# with a little extra information in case a certain function has not been
# implemented. This is in some way duplicating default Julia behaviour, but it
# has the advantage that we can write documention for non-existing functions and
# it also emphasizes that this specific function is part of the abstract atoms
# interface.

##### This is an old version of the macro, for Julia 0.3
# macro protofun(fname, argtypes...)
#     docstr = argtypes[end]
#     argtypes = argtypes[1:end-1]
#     str = "@doc doc\"$(docstr)\"->\nfunction $(fname)("
#     for n = 1:length(argtypes)
#         str *= string("arg", n, argtypes[n], ", ")
#     end
#     str = str[1:end-2]
#     str *= ") \n"
#     str *= "    error(string(\"AtomsInterface: `$(string(fname))(\", "
#     for n = 1:length(argtypes)
#         str *= string("\"::\", typeof(arg", n, ")")
#         if n < length(argtypes)
#             str *= ", \", \", "
#         end
#     end
#     str *= ", \")' has no implementation.\"))\nend"
#     eval(parse(str))            
# end

macro protofun(fsig::Expr)
    @assert fsig.head == :call 
    fname = fsig.args[1] 
    argnames = Any[] 
    for idx in 2:length(fsig.args) 
        arg = fsig.args[idx] 
        if isa(arg, Expr) && arg.head == :kw 
            arg = arg.args[1] 
        end 
        if isa(arg, Symbol) 
            push!(argnames, arg) 
        elseif isa(arg, Expr) && arg.head == :(::) 
            if length(arg.args) != 2 
                @gensym s 
                insert!(arg.args, 1, s) 
            end 
            push!(argnames, arg.args[1]) 
        end 
    end 
    body = quote 
        error(string("AtomsInterface: ", $fname, 
                     ($([:(typeof($(esc(arg)))) for arg in argnames]...),),
                     " ) has no implementation.") ) 
    end 
    Expr(:function, esc(fsig), body)
end

end
