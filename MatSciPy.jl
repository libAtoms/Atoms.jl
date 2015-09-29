"""
## module MatSciPy

### Summary

Julia wrappers for the [matscipy](https://github.com/libAtoms/matscipy) library.
At the moment, only a neighbourlist is implemented. 

* `MatSciPy.neighbour_list` : the raw neighbour_list wrapper
* `MatSciPy.NeighbourList` : some boiler plate code, including an iterator

`neighbourlist` currently uses the Python interface and therefore requires
ASE. 

"""
module MatSciPy

using ASE, PyCall, Potentials
@pyimport matscipy.neighbours as matscipy_neighbours

export update!, Sites, NeighbourList


"""
`neighbour_list(atoms::ASEAtoms, quantities::AbstractString, cutoff::Float64) 
    -> tuple`

The elements of the tuple depend on the content of `quantities`. E.g., 
```{julia}
    i, j, D, d = neighbours(at, "ijDd", 5.0)
```
will return a vector `i` of indices, a vector of neighbour indices `j`, 
the distance vectors in `D` and the scalar distances in `d`.

**Warning: ** to minimise overhead, this does *not* convert the relative
distance vectors `D` from the ASE N x 3 to the Atoms.jl 3 x N convention!
"""
function neighbour_list(atoms::ASEAtoms, quantities::AbstractString,
                        cutoff::Float64)
    results = matscipy_neighbours.neighbour_list(quantities,
                                                 atoms.po, # pyobject(atoms),
                                                 cutoff)
    results = collect(results) # tuple -> array so we can change in place
    # translate from 0- to 1-based indices
    for (idx, quantity) in enumerate(quantities)
        if quantity == 'i' || quantity == 'j'
            results[idx] += 1
        end
    end
   return tuple(results...)
end


"""
A basic wrapper around the `neighbour_list` builder. 

Initialise an empty neighbourlist using
```
nlist = NeighbourList(cutoff; quantities="ijDd", skin=0.5)
```
where `quantities` is a string which my contain

* 'i' : first atom index
* 'j' : second atom index
* 'd' : absolute distance
* 'D' : distance vector
* 'S' : shift vector (number of cell boundaries crossed by the bond
              between atom i and j)

By convention, the string *must* start with `"ij"`. `Skin` is the *buffer* which
is added to the cut-off radius.

The second constructor is
```
nlist = NeighbourList(cutoff, at::ASEAtoms; kwargs...)
```
and automatically calls `build!(nlist, at)` to initialise the neighbourlist.
"""
type NeighbourList
    cutoff::Float64
    quantities::ASCIIString   # remember which quantities the user wants
    skin::Float64
    # X::Matrix{Float64}   # positions at which the list was built
    # i::Vector{Int}       # i,j,D,d are the quantities that are computed
    # j::Vector{Int} 
    # D::Matrix{Float64}
    # d::Vector{Float64}
    # S::Matrix{Int}
    Q::Dict{Char, Any}
end

# eye-candy access to the quantities
getitem(nlist::NeighbourList, idx::Char) = nlist.Q[idx]
setitem!(nlist::NeighbourList, val, idx::Char) = begin nlist.Q[idx] = val end

# empty constructor
function NeighbourList(cutoff::Float64; quantities="ijdD", skin=0.5)
    if quantities[1:2] != "ij"
        error("NeighbourList: quantities must start with \"ij\"")
    end
    NeighbourList(cutoff, quantities, skin, Dict{Char, Any}())
end

# cosntruct from ASEAtoms
function NeighbourList(cutoff::Float64, at::ASEAtoms; kwargs...)
    nlist = NeighbourList(cutoff; kwargs...)
    return build!(nlist, at)
end

# build the neighbourlist
"""`build!(nlist::NeighbourList, at::ASEAtoms) -> NeighbourList

Forces a rebuild of the neighbourlist.
"""
function build!(nlist::NeighbourList, at::ASEAtoms)
    # by deleting the "X" key (if it exists) we force update! to
    # rebuild
    delete!(nlist.Q, 'X')
    return update!(nlist, at)
end

# build the neighbourlist
function update!(nlist::NeighbourList, at::ASEAtoms)
    ####### TODO: use reference for better performance? ######
    Xnew = positions(at)
    # if nlist.Q contains an X array, then we should check whether
    # the new X array has moved enough to rebuild
    if haskey(nlist.Q, 'X')
        ##### TODO: this is AWFUL; need to make this a periodic distance! #####
        if norm(Xnew[:]-X[:], Inf) < nlist.skin / 2
            return nlist
        end
    end

    # we decided that we need to rebuild
    nlist.Q['X'] = Xnew
    qlist = neighbour_list(at, nlist.quantities, nlist.cutoff+nlist.skin)
    for (c, q) in zip(nlist.quantities, qlist)
        nlist.Q[c] = q
    end

    return nlist
end



# implementation of an iterator


"""`Sites`: helper to define an iterator over sites. Usage:
```{julia}
for n, ... in Sites(nlist)

end
```

Equivalently, one can just call
```{julia}
for n, ... in nlist
```
"""
type Sites
    neiglist::NeighbourList
end

"""`type AtomIteratorState` : iterator state for iterating over
sites via the `MatSciPy.NeighbourList`; see `?Sites`.
"""
type AtomIteratorState
    neiglist::NeighbourList
    n::Int         # site index
    m::Int         # index on where in neiglist we are
end

import Base.start
start(s::Sites) = AtomIteratorState(s.neiglist, 0, 0)

import Base.done
done(::Sites, state::AtomIteratorState) =
    (size(state.neiglist.Q['X'],2) == state.n)

import Base.next
function next(::Sites, state::AtomIteratorState)
    state.n += 1
    m0 = state.m
    i = state.neiglist.Q['i']
    len_i = length(i)
    while i[state.m+1] <= state.n
        state.m += 1
        if state.m == len_i; break; end
    end
    if m0 == state.m
        return (state.n, [], [], []), state
    else
        ### TODO: allow arbitrary returns! ###
        return (state.n,
                state.neiglist.Q['j'][m0+1:state.m],
                state.neiglist.Q['d'][m0+1:state.m],
                state.neiglist.Q['D'][m0+1:state.m, :]'), state
    end

    # TODO: in the above loop we could also remove all those neighbours
    #       which are outside the cutoff?
end


###########################################################################
## Some Calculators Optimised for use with MatSciPy.NeighbourList 
###########################################################################



end
