
module AtomsInterface


export AbstractAtoms,
       AbstractAtomsX,
       get_positions,
       set_positions!,
       ddim,
       rdim


import Base.length, Base.getindex, Base.setindex!


# Lexicon.update! is something that many users will have installed, 
try
    import Lexicon.update!
catch
    nothing
end

# the following macro generates "function defaults" which throw an error
# message with a little extra information in case a certain function
# has not been implemented. This is in some way duplicating default Julia
# behaviour, but it has the advantage that we can write documention for
# non-existing functions and it also emphasizes that this specific function
# is part of the abstract atoms interface.

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


#######################################################################
#     AbstractAtoms
#######################################################################

"""`AbstractAtoms`: the abstract supertype for storing atomistic
configurations. A basic implementation might simply store a list of positions.
"""
abstract AbstractAtoms


# the following are all dummy method definitions that just throw an error if a
# method hasn't been implemented

"Return number of atoms"
@protofun length(::AbstractAtoms)

"Return position(s) of atom(s) `i`"
@protofun getindex(::AbstractAtoms, i)

"Set position(s) of atom(s) `i`"
@protofun setindex!(::AbstractAtoms, x, i)

"Return reference to positions of all atoms as a `d x N` array."
@protofun get_positions(::AbstractAtoms)

"Set positions of all atoms as a `d x N` array."
@protofun set_positions!(::AbstractAtoms, ::Any)

"Return domain dimension."
@protofun ddim(::AbstractAtoms)

"Return range dimension."
@protofun rdim(::AbstractAtoms)





#######################################################################
#     AbstractAtomsX
# a still abstract implementation of much of the functionality
# of AbstractAtoms, assuming that atom positions are just standard
# vectors and are stored in a field .X
#######################################################################


"""`AbstractAtomsX`: implementations of this abstract sub-type
*must* have a field `.X`, with d rows and nX columns, where nX is the
number of atoms and d the space dimension in which the atoms live, normally
d = 3.  With this guaranteed, much of the functionality of the 
`AbstractAtoms` interface can already be implemented.
""" 
abstract AbstractAtomsX

# note: this is not part of the interface, but I am finding it very useful
# maybe it could be included in the general interface as well?
"Tell the 'listeners' that the atom configuration has changed."
update!(atm::AbstractAtomsX) = nothing

# implementations of the standard interface
length(atm::AbstractAtomsX) = size(atm.X, 2)
getindex(atm::AbstractAtomsX, i) = atm.X[:, i]
setindex!(atm::AbstractAtomsX, x, i) = begin atm.X[:, i] = x; update!(atm); end
get_positions(atm::AbstractAtomsX) = atm.X
set_positions!(atm::AbstractAtomsX, X) = begin atm.X = X; update!(atm) end
ddim(atm::AbstractAtomsX) = size(atm.X, 1)
rdim(atm::AbstractAtomsX) = size(atm.X, 1)



#######################################################################
#  NEIGHBOURLIST
#######################################################################

"Abstract supertype of neighbourlists."
abstract AbstractNeighbourList


# each atoms object should probably have a neighbourlist attached even
# if it is just a trivial "all atoms are neighbours" thing
"Return the neighbourlist attached to the atoms object."
@protofun get_neiglist(::AbstractAtoms)

"attach a neighbourlist to an atoms object"
@protofun set_neiglist!(::AbstractAtoms, ::AbstractNeighbourList)

"tells the neighbourlist that the atoms object has changed"
update!(::AbstractAtoms, ::AbstractNeighbourList) = nothing

"""`get_neigs(n, a; rcut==-1) -> Ineig, r`: returns a list of
indices of neighbours of `n` and distances `r`. if rcut == -1 (default), then
the cut-off  radius originally supplied to the neighbourlist is used.
If rcut > nlist.rcut, then an error is thrown.

(Note that `r` is computed anyhow, so it is passed by default; it can be
discarded if not needed.)
"""
get_neigs(n, atm::AbstractAtoms; rcut=-1) =
    get_neigs(n, get_nlist(atm), atm; rcut=-1)

# DISCUSS: maybe it would be useful to also return the list of
#          direction-vectors? It was probably computes anyhow

"""Returns `(Ineig, r)` where `Ineig is an integer vector with indices 
of neighbour atoms of `n` and r the a Float vector with their distances."""
@protofun get_neigs(n::Integer, neigs::AbstractNeighbourList,
                    atm::AbstractAtoms; rcut=-1)



#######################################################################
#  CONSTRAINTS
#######################################################################

# constraints implement boundary conditions, or other types
# of constraints; the details of this interface are still a bit fuzzy for me

"""Abstract supertype for constraints; these are objects that implement boundary
conditions."""
abstract AbstractConstraints

"Return constraints attached to the atoms object (if one exists)"
@protofun get_constraints(::AbstractAtoms)

"Attach a constraints object to the atoms object"
@protofun set_constraints!(::AbstractAtoms, ::AbstractConstraints)


"""Returns a bare `Vector{T <: FloatingPoint}` object containing the degrees of
freedom describing the state of the simulation. This function should be
overloaded for concrete implementions of `AbstractAtoms` and
`AbstractConstraints`.  

Alternative wrapper function 
    get_dofs(atm::AbstractAtoms) = get_dofs(atm, get_constraints(atm))
"""
@protofun get_dofs(a::AbstractAtoms, c::AbstractConstraints)
get_dofs(atm::AbstractAtoms) = get_dofs(atm, get_constraints(atm))

"""Takes a \"dual\" array (rdim x lenght) and applies the dual constraints
to obtain effective forces acting on dofs. Returns a vector of the same
length as dofs."""
@protofun forces_to_dofs(Matrix{T <: FloatingPoint}, con::AbstractConstraints)


#######################################################################
#     CALCULATOR
#######################################################################


"""`AbstractCalculator`: the abstract supertype of calculators. These
store model information, and are linked to the implementation of energy, forces,
and so forth.  """
abstract AbstractCalculator

# The following two functions are, in my view, superfluous for the
# general interface since - typically - a calculator need not be attached
# to a specific atoms object. However, it does turn out to be convenient to
# have the calculator attached to the atoms object; see multiple convenience
# wrapper functions below.
"Return calculator attached to the atoms object (if one exists)"
@protofun get_calculator(::AbstractAtoms)

"Attach a calculator to the atoms object"
@protofun set_calculator(::AbstractAtoms)


"Returns the cut-off radius of the potential."
@protofun rcut(::AbstractCalculator)

## ==============================
## get_E and has_E   : total energy

"""Return the total energy of a configuration of atoms `a`, using the calculator
`c`.  Alternatively can call `get_E(a) = get_E(a, get_calculator(a))` """
@protofun get_E(a::AbstractAtoms, c::AbstractCalculator)
get_E(a::AbstractAtoms) = get_E(a, get_calculator(a))

"Returns `true` if the calculator `c` can compute total energies."
@protofun has_E(c::AbstractCalculator)
has_E(a::AbstractAtoms) = has_E(get_calculator(a))

## ==============================
## get_Es and has_Es   : site energy

"""`get_Es(idx, a::AbstractAtoms, c::AbstractCalculator)`: 
Returns an `Vector{Float64}` of site energies of a configuration of
atoms `a`, using the calculator `c`. If idx==[] (default), then *all*
site energies are returned, otherwise those corresponding to the list
of indices idx.
"""
@protofun get_Es(idx, a::AbstractAtoms, c::AbstractCalculator)
get_Es(idx, a::AbstractAtoms) = get_Es(idx, a, get_calculator(a))

"Returns `true` if the calculator `c` can compute site energies."
@protofun has_Es(c::AbstractCalculator)
has_Es(a::AbstractAtoms) = has_Es(get_calculator(a))


"""same calling convention as get_Es.

Returns a tuple `(dEs, Ineigs)`, where `dEs` is d x nneigs and
`Ineigs` is the list of neighbours for which the forces have been computed
"""
@protofun get_dEs(idx, a::AbstractAtoms, c::AbstractCalculator)
get_dEs(idx, a::AbstractAtoms) = get_dEs(idx, a, get_calculator(calc))


# ==========================================================
# get_dE
# (every calculator needs this, so there is no has_dE())

"""Returns the  gradient of the total energy in the format `rdim x length`.
Alternatively, one can call the simplified form
    get_dE(a::AbstractAtoms) = get_dE(a, get_calculator(a))
provided that a.calc is avilable."""
@protofun get_dE(a::AbstractAtoms, c::AbstractCalculator)
get_dE(a::AbstractAtoms) = get_dE(a, get_calculator(a))


"Return gradient of total energy taken w.r.t. dofs, i.e., as a long vector. "
get_dE_dofs(a::AbstractAtoms, calc::AbstractCalculator, con::AbstractConstraints) =
    forces_to_dofs(get_dE(a, calc), con)
get_dE_dofs(a::AbstractAtoms) = get_dE(a, get_calculator(a),
                                       get_constraints(a) )

end
