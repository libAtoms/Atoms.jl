###
#
# TODO
#   * add links to key arrays so they can be modified in-place
#     or accessed cheaply
#
#   * in all my julia codes I use self_interaction=false and bothways=true
#      - is it a bad idea if I make these default?
#
#   * rcut in ASE denotes spheres of overlap?!?!? i.e. it is in effect
#     half of the cut-off of the potential?
#      


"""
# module ASE

## Summary

Provides Julia wrappers for some of ASE's functionality.

todo: write more documentation
"""
module ASE

using AtomsInterface
importall AtomsInterface

export ASEAtoms
export convert, get_array, set_array!, get_positions, set_positions!, positions
export get_cell, set_cell!
export set_calculator, get_forces, get_potential_energy, get_stress
export repeat, bulk, length
export ASENeighborList, get_neighbors, neighbors
export get_cell, cell


using PyCall
@pyimport ase
@pyimport ase.lattice as lattice
@pyimport ase.calculators.neighborlist as ase_neiglist

#################################################################
###  Wrapper for ASE Atoms object and its basic functionality
################################################################


"""`type ASEAtoms <: AbstractAtoms`

Julia wrapper for the ASE `Atoms` class

If `at` is of type `ASEAtoms` and `nlist` of type `ASENeighbourList` then 
iterate over atoms using the following syntax
```
for (n, inds, s, r) in (at, nlist)
    # n : index of current atom
    # inds : indices of neighbours
    # s : distances of neighbours
    # r : relative position vectors of neighbours
end
```
"""
type ASEAtoms <: AbstractAtoms
    po::PyObject      # ase.Atoms instance
end



"Return the PyObject associated with `a`"
pyobject(a::ASEAtoms) = a.po


# # [Q] Why is `convert` needed? Can't we just use the contructor?
import Base.convert
convert{T <: ASEAtoms}(::Type{T}, po::PyObject) = ASEAtoms(po)


get_array(a::ASEAtoms, name) = a.po[:get_array(name)]
set_array!(a::ASEAtoms, name, value) = a.po[:set_array(name, value)]


"""`get_positions(at)` returns a copy of the atom positions array

*NOTE:* ASE stores an N x 3, while the `Atoms.jl` convention is to use 
3 x N arrays. The downside is that, for now, get_positions() creates 3 copies:
 first, within python, then from python to julia, then in the transpose.

TODO: rewrite to create a single copy.
"""
get_positions(a::ASEAtoms) = a.po[:get_positions]()'


"alias for `get_positions`"
positions = get_positions


"""`set_positions!(at::ASEAtoms, p::Array{Float64, 2})`

Sets the position array in `at`; Note that `p` must be 3 x N 
(Atoms.jl convention), and is automatically converted to the N x 3 array
that ASE expects.
"""
set_positions!(a::ASEAtoms, p::Array{Float64, 2}) = a.po[:set_positions](p')

get_cell(a::ASEAtoms) = a.po[:get_cell]()
set_cell!(a::ASEAtoms, p::Array{Float64,2}) = a.po[:set_cell](p)

import Base.length
length(a::ASEAtoms) = a.po[:get_number_of_atoms]()


"""`get_cell(at::ASEAtoms) = at.po[:get_cell]() -> Matrix`

Alas for `ase.Atoms.get_cell()`; returns a matrix describing the computational
cell.
"""
get_cell(at::ASEAtoms) = at.po[:get_cell]()
"alias for get_cell"
cell = get_cell


# TODO: tie in properly with AtomsInterface
set_calculator!(a::ASEAtoms, calculator::PyObject) = a.po[:set_calculator](calculator)
get_forces(a::ASEAtoms) = a.po[:get_forces]()
get_potential_energy(a::ASEAtoms) = a.po[:get_potential_energy]()
get_stress(a::ASEAtoms) = a.po[:get_stress]()



############################################################
### Some additional useful functionality that ASE
### provides
############################################################

import Base.repeat
"""`repeat(a::ASEAtoms, n::(Int64, Int64, Int64)) -> ASEAtoms`

Takes an `ASEAtoms` configuration / cell and repeats is n_j times
into the j-th dimension.

For example,
```
    atm = repeat( bulk("C"), (3,3,3) )
```
creates 3 x 3 x 3 unit cells of carbon.
""" 
repeat(a::ASEAtoms, n::NTuple{3, Int64}) =
    convert(ASEAtoms, a.po[:repeat](n))


"""`bulk(name::AbstractString; kwargs...) -> ASEAtoms`

Generates a unit cell of the element described by `name`
"""
bulk(name::AbstractString; kwargs...) =
    convert(ASEAtoms, lattice.bulk(name; kwargs...))


############################################################
### Some extra hacks for convenience or performance
############################################################

"""`_get_positions_ref_(atm::ASEAtoms) -> PyArray`

This returns a *reference* to the list of positions stored in  `atm.po`.
By contrast, `get_positions` returns a copy.
"""
_get_positions_ref_(atm::ASEAtoms) =
    pycall(atm.po[:get_array], PyArray, "positions", copy=false)



############################################################
###  ASE Neighborlist implementation
############################################################


"""`type ASENeighborList <: AbstractNeighborList`

This makes available the functionality of the ASE neighborlist implementation.
The neighborlist itself is actually stored in `ASEAtoms.po`, but 
attaching and `ASENeighborList` will indicate this.

Keyword arguments:
 * skin = 0.3
 * sorted
 * self_interaction
 * bothways
"""
type ASENeighborList #  <: AbstractNeighborList
    po::PyObject
end

# default constructor from a list of cut-offs
ASENeighborList(cutoffs::Vector{Float64}; kwargs...) =
    ASENeighborList(ase_neiglist.NeighborList(cutoffs; kwargs...))

# constructor from an ASEAtoms object, with a single cut-off
# this generates a list of cut-offs, then the neighborulist, then
# builds the neighborlist from the ASEAtoms object
ASENeighborList(atm::ASEAtoms, cutoff::Float64; kwargs...) =
    ASENeighborList(atm, cutoff * ones(length(atm)); kwargs...)

# constructor from an ASEAtoms object, with multiple cutoffs
# this also builds the neighborlist
function ASENeighborList(atm::ASEAtoms, cutoffs::Vector{Float64}; kwargs...) 
    nlist = ASENeighborList(cutoffs; kwargs...)
    update!(nlist, atm)
    return nlist
end

# regain the cutoffs vector
_get_cutoffs_ref_(nlist::ASENeighborList) =
    PyArray(nlist.po["cutoffs"])



"""`update!(nlist::ASENeighborList, atm::ASEAtoms)`

checks whether the atom positions have moved by more than the skin
and rebuilds the list if they have done so.
"""
update!(nlist::ASENeighborList, atm::ASEAtoms) =  nlist.po[:update](atm.po)


"""`build!(nlist::ASENeighborList, atm::ASEAtoms)`
    
force rebuild of the neighborlist
"""
build!(nlist::ASENeighborList, atm::ASEAtoms) = nlist.po[:build](atm.po)


"""`get_neighbors(n::Integer, neiglist::ASENeighborList) -> (indices, offsets)`

Return neighbors and offsets of atom number n. The positions of the neighbor
atoms can then calculated like this: (python code)

```
          indices, offsets = nl.get_neighbors(42)
          for i, offset in zip(indices, offsets):
              print(atoms.positions[i] + dot(offset, atoms.get_cell()))
```

If `get_neighbors(a)` gives atom b as a neighbor,
    then `get_neighbors(b)` will not return a as a neighbor, unless
    `bothways=True` was used.
""" 
function get_neighbors(n::Integer, neiglist::ASENeighborList)
    indices, offset = neiglist.po[:get_neighbors](n-1)
    indices .+= 1
    return (indices, offset)
end
# "alias for `get_neighbors`"
# get_neighbors = get_neighbors
# "alias for `get_neighbors`"
# neighbors = get_neighbors
"alias for `get_neighbors`"
neighbors = get_neighbors


"""`get_neighbors(n::Integer, neiglist::ASENeighborList, atm::ASEAtoms)
              -> (indices::Vector{Int}, s::Vector{Float64}, r::Matrix{Float64})`

 * `indices`: indices of atom positions
 * `s`: scalar distances of neighbours
 * `r`: relative positions of neighbours (vectorial dist)

This is a convenience function that does some of the work of constructing the 
neighborhood. This is probably fairly inefficient to use since it has 
to construct a `PyArray` object for the positions every time it is called.
Instead, use the iterator ***TODO***.
"""
function get_neighbors(n::Integer, neiglist::ASENeighborList, atm::ASEAtoms;
                       rcut=Inf)
    (inds, offsets) = ASE.neighbors(n, neiglist)
    # the next three lines are horrendously expensive !!!!!
    #     create links so they don't have to be reconstructed, or
    #     an iterator; makes you wonder though how much in the first line
    #     is also cost due to conversion?!
    # ----------
    X = _get_positions_ref_(atm)
    cutoffs = _get_cutoffs_ref_(neiglist)
    cell = get_cell(atm)
    # ----------
    r = X[inds, :]' + cell' * offsets' .- slice(X, n, :)
    s = sqrt(sumabs2(r, 1))
    if rcut < Inf
        I = find(s .<= rcut)
        return inds[I], s[I], r[:, I]
    else
        return inds, s, r
    end
end



##################### NEIGHBOURHOOD ITERATOR ###################
# this is about twice as fast as `get_neighbours`
# which indicates that, either, the ASE neighbour list is very slow
#   or, the overhead from the python call is horrendous!
#

type ASEAtomIteratorState
    at::ASEAtoms
    neiglist::ASENeighborList
    n::Int         # iteration index
    X::PyArray
    cutoffs::PyArray
    cell_t::Matrix{Float64}
end

ASEAtomIteratorState(at::ASEAtoms, neiglist::ASENeighborList) =
    ASEAtomIteratorState( at, neiglist, 0,
                          _get_positions_ref_(at),
                          _get_cutoffs_ref_(neiglist),
                          get_cell(at)' )

import Base.start
start(I::Tuple{ASEAtoms,ASENeighborList}) =
    ASEAtomIteratorState(I...)

import Base.done
done(I::Tuple{ASEAtoms,ASENeighborList}, state::ASEAtomIteratorState) =
    (state.n == length(state.at))

import Base.next
function next(I::Tuple{ASEAtoms,ASENeighborList}, state::ASEAtomIteratorState)
    state.n += 1
    (inds, offsets) = neighbors(state.n, state.neiglist)
    # indices, offset = state.neiglist.po[:get_neighbors](n-1)
    r = state.X[inds, :]' + state.cell_t * offsets' .- slice(state.X, state.n, :)
    s = sqrt(sumabs2(r, 1))
    return (state.n, inds, s, r), state
    # now find the indices that are actually within the cut-off
    #  TODO: resolve the nasty issue of 1/2 cut-off first?!?!?
    # I = find(s .< 3.0)   # state.cutoffs[state.n])  (hard-coded for debugging)
    # return (state.n, inds[I], s[I], r[:,I]), state
end


end
