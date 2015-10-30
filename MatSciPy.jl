
"""
## module MatSciPy

### Summary

Julia wrappers for the [matscipy](https://github.com/libAtoms/matscipy) library.
At the moment, only a neighbourlist is implemented. 

* `MatSciPy.neighbour_list` : the raw neighbour_list wrapper
* `MatSciPy.NeighbourList` : some boiler plate code, including an iterator

`neighbourlist` currently uses the Python interface and therefore requires
ASE. 

TODO: rewrite so that ASE is no longer required!

The `MatSciPy` module also implements some simple "one-line" calculators that
exploit the specific structure of the data returned by `neighbour_list`. These
are very well optimised. As a rough guidance, for the `PairCalculator`, with
`LennardJonesPotential` potential and `ShiftCutoff`, the cost of one force
assembly is about 2/3 the cost of one `neighbour_list` call, whereas an 
optimised implementation would be only about 1/10th of a `neighbourlist` call.
With the `SWCutoff` the cost of one force assembly is about twice the cost of 
one `neighbour_list` call. 
"""
module MatSciPy

using ASE, PyCall, Potentials, AtomsInterface
importall AtomsInterface

@pyimport matscipy.neighbours as matscipy_neighbours



# # import some faster exponential for fast potential assembly
# try
#     import AppleAccelerate
#     function _fast_exponential(Lc, Rc, r) 
#         c2 = Rc-r
#         AppleAccelerate.rec!(c2, c2)
#         c2 = Lc * c2
#         AppleAccelerate.exp!(c2, c2)
#         return c2
#     end
# catch
#     _fast_exponential(Lc, Rc, r) = exp( Lc ./ (Rc - r) )
# end


export update!, Sites, NeighbourList, Bonds
export potential_energy, potential_energy_d, forces
export cutoff
export simple_binsum


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
    if length(quantities)==1
        return results
    end
    results = collect(results) # tuple -> array so we can change in place
    # translate from 0- to 1-based indices
    for (idx, quantity) in enumerate(quantities)
        if quantity == 'i' || quantity == 'j'
            results[idx] += 1
        end
    end
   return tuple(results...)
end

function neighbour_list(atoms::ASEAtoms, cutoff::Float64)
    results = matscipy_neighbours.neighbour_list("ijdDS", atoms.po, cutoff)
    i=results[1]::Vector{Int32}
    j=results[2]::Vector{Int32}
    r=results[3]::Vector{Float64}
    R=results[4]::Matrix{Float64}
    S=results[5]::Matrix{Int32}
    i += 1
    j += 1
    return i, j, r, R, S
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
    skin::Float64
    i::Vector{Int32}
    j::Vector{Int32}
    r::Vector{Float64}
    R::Matrix{Float64}
    S::Matrix{Int32}
    X::Matrix{Float64}
end

# overload getindex to allow direct access to quantities stored in Q.
Base.getindex(nlist::NeighbourList, key) = nlist.Q[key]

# empty constructor
NeighbourList(cutoff::Float64; skin=0.0) =
    NeighbourList(cutoff, skin,
                  Int32[], Int32[], Float64[],
                  Matrix{Float64}(), Matrix{Int32}(),
                  Matrix{Float64}())

# construct from ASEAtoms
NeighbourList(cutoff::Float64, at::ASEAtoms; kwargs...) =
    build!(NeighbourList(cutoff; kwargs...), at)
NeighbourList(at::ASEAtoms, cutoff::Float64; kwargs...) =
    NeighbourList(cutoff, at; kwargs...)

import Base.length
"returns the number of (i,j) pairs in the raw neighbourlist"
length(nlist::NeighbourList) = length(nlist.i)

# build the neighbourlist
"""`build!(nlist::NeighbourList, at::ASEAtoms) -> NeighbourList

Forces a rebuild of the neighbourlist.
"""
function build!(nlist::NeighbourList, at::ASEAtoms)
    # by deleting the "X" (if it exists) we force update! to rebuild
    nlist.X = Matrix{Float64}()
    return update!(nlist, at)
end


# update / build the neighbourlist
function update!(nlist::NeighbourList, at::ASEAtoms)
    ####### TODO: use reference for better performance? ######
    Xnew = positions(at)
    # if nlist.Q contains an X array, then we should check whether
    # the new X array has moved enough to rebuild
    if ~isempty(nlist.X)
        ##### TODO: this is AWFUL; need to make this a periodic distance! #####
        if vecnorm(Xnew - nlist.X, Inf) < nlist.skin / 2
            return nlist
        end
    end
    
    # we decided that we need to rebuild
    nlist.X = Xnew
    nlist.i, nlist.j, nlist.r, nlist.R, nlist.S =
        neighbour_list(at, nlist.cutoff+nlist.skin)
    
    return nlist
end


######################################################################
#### implementation of some iterators


import Base.start
import Base.done
import Base.next


"""
# `type Bonds`

Basic iterator over all pairwise bonds. Usage:

```
for (i, j, r, R) in Bonds(atm)
   # do something with r the length and R the directional vector
   # and i, j the indices of the bonds (R points to j)
end
```

Can also be instantiated with `Bonds(neiglist)`
"""
type Bonds
    nlist::NeighbourList
end
Bonds(at::ASEAtoms, rcut) = Bonds(NeighbourList(at, rcut))
start(b::Bonds) = 1::Int
done(b::Bonds, s::Int) = (s == length(b.nlist.i)+1)
next(b::Bonds, s::Int) = (b.nlist.i[s], b.nlist.j[s], b.nlist.r[s],
                          copy(slice(b.nlist.R,s,:)),
                          copy(slice(b.nlist.S,s,:)) ), s+1


"""`Sites`: helper to define an iterator over sites. Usage:
```{julia}
for n, ... in Sites(nlist)

end
```

Equivalently, one can just call
```{julia}
for n, ... in nlist
```

Yet another way to loop over sites is
```{julia}
for n, ... in Sites(at, rcut)
```
where `at` is an `ASEAtoms` object and `rcut` the desired cut-off radius.
"""
type Sites
    neiglist::NeighbourList
end

# a simpler constructor, directly from an atoms object
Sites(at::ASEAtoms, rcut) = Sites(NeighbourList(at, rcut))

"""`type AtomIteratorState` : iterator state for iterating over
sites via the `MatSciPy.NeighbourList`; see `?Sites`.
"""
type AtomIteratorState
    n::Int         # site index
    m::Int         # index on where in neiglist we are
end

start(s::Sites) = AtomIteratorState(0, 0)
done(s::Sites, state::AtomIteratorState) = (size(s.neiglist.X, 2) == state.n)
function next(s::Sites, state::AtomIteratorState)
    state.n += 1
    m0 = state.m
    len_i = length(s.neiglist)
    while s.neiglist.i[state.m+1] <= state.n
        state.m += 1
        if state.m == len_i; break; end
    end
    ### TODO: allow arbitrary returns! ### (really necessary? maybe via macro)
    ret_tuple = (state.n,
                 s.neiglist.j[m0+1:state.m],
                 s.neiglist.r[m0+1:state.m],
                 s.neiglist.R[m0+1:state.m, :]',
                 s.neiglist.S[m0+1:state.m, :]')
    return ret_tuple, state    
    # TODO: in the above loop we could also remove all those neighbours
    #       which are outside the cutoff?
end




###########################################################################
## Implementation of some basic calculators using MatSciPy.NeighborList
###########################################################################


"""`simple_binsum` : this is a placeholder for a more general function, 
`binsum`, which still needs to be written! Here, it is assumed that
`size(A, 1) = 3`, and  only summation along the second dimension is allowed.
"""
function simple_binsum{TI <: Integer, TF <: AbstractFloat}(i::Vector{TI},
                                                           A::Matrix{TF})
    if size(A, 1) != 3
        error("simple_binsum: need size(A,1) = 3")
    end
    if size(A, 2) != length(i)
        error("simple_binsum: need size(A,2) = length(i)")
    end
    B = zeros(TF, 3, maximum(i))
    for m = 1:size(A,1)
        # @inbounds @simd
        for n = 1:length(i)
            B[m, i[n]] = B[m,i[n]] + A[m, n]
        end
    end
    return B
end

function simple_binsum{TI <: Integer, TF <: AbstractFloat}(i::Vector{TI},
                                                           A::Vector{TF})
    if length(A) != length(i)
        error("simple_binsum: need length(A) = length(i)")
    end
    B = zeros(TF, maximum(i))
    # this ought to be a SIMD loop. but that gives me a wrong answer! why?!
    for n = 1:length(i)
        B[i[n]] += + A[n]
    end
    return B
end



"""`PairCalculator` : basic calculator for pair potentials.
"""
type PairCalculator <: AbstractCalculator
    pp::PairPotential
end

import Potentials.cutoff
cutoff(calc::PairCalculator) = cutoff(calc.pp)

function potential_energy(at::ASEAtoms, calc::PairCalculator)
    r = neighbour_list(at, "d", cutoff(calc))
    return sum( calc.pp(r) )
end

function potential_energy_d(at::ASEAtoms, calc::PairCalculator)
    i, r, R = neighbour_list(at, "idD", cutoff(calc))
    return - 2.0 * simple_binsum(i, @GRAD calc.pp(r, R') )
end

forces(at::ASEAtoms, calc::PairCalculator) = - potential_energy_d(at, calc)


"`EAMCalculator` : basic calculator using the `EAMPotential` type"
type EAMCalculator <: AbstractCalculator
    p::EAMPotential
end

cutoff(calc::EAMCalculator) = max(cutoff(calc.p.V), cutoff(calc.p.rho))

function potential_energy(at::ASEAtoms, calc::EAMCalculator)
    i, r = neighbour_list(at, "id", cutoff(calc))
    return ( sum(calc.p.V(r))
             + sum( calc.p.embed( simple_binsum( i, calc.p.rho(r) ) ) ) )
end

function potential_energy_d(at::ASEAtoms, calc::EAMCalculator)
    i, j, r, R = neighbour_list(at, "ijdD", cutoff(calc))
    # pair potential component
    G = - 2.0 * simple_binsum(i, @GRAD calc.p.V(r, R'))
    # EAM component
    dF = @D calc.p.embed( simple_binsum(i, calc.p.rho(r)) )
    dF_drho = dF[i]' .* (@GRAD calc.p.rho(r, R'))
    G += simple_binsum(j, dF_drho) - simple_binsum(i, dF_drho)
    return G
end




# ###########################################################################
# ## Some Calculators Optimised for use with MatSciPy.NeighbourList 
# ###########################################################################


# """
# `lennardjones_old(at::ASEAtoms, nlist::NeighbourList;
#                       r0=1.0, e0=1.0, quantities="EG")`

# A fast LJ assembly which exploits the `@simd` and `@inbounds` macros as well
# as the specific structure of `MatSciPy.NeighbourList`.

# **This has not yet been tested for correctness!**

# **This version does not have a cutoff radius!**
# """
# function lennardjones_old(at::ASEAtoms, nlist::NeighbourList;
#                       r0=1.0, e0=1.0, quantities="EG")
#     update!(nlist, at)
#     E = 0.0
#     r = nlist.Q['d']::Vector{Float64}
#     R = nlist.Q['D']::Array{Float64,2}
#     i = nlist.Q['i']::Vector{Int32}
#     t = Vector{Float64}(length(r))

#     @simd for n = 1:length(t) @inbounds begin
#         t[n] = r0 / r[n]
#         t[n] = t[n]*t[n]*t[n]
#         t[n] = t[n]*t[n]
#         E = E + t[n]*(t[n]-2.0)
#     end end
#     E *= e0

#     if 'G' in quantities
#         G = zeros(3, i[end])
#         @simd for n = 1:length(t) @inbounds begin
#             t[n] = e0 * 24.0 * t[n]*(t[n]-1.0) / (r[n]*r[n])
#             G[1,i[n]] = G[1,i[n]] + t[n] * R[n,1]
#             G[2,i[n]] = G[2,i[n]] + t[n] * R[n,2]
#             G[3,i[n]] = G[3,i[n]] + t[n] * R[n,3]
#         end end
#     end
    
#     ret = ()
#     for c in quantities
#         if c == 'E'
#             ret = tuple(ret..., E)
#         elseif c == 'G'
#             ret = tuple(ret..., G)
#         end
#     end
#     return ret
# end




# """
# `lennardjones(at::ASEAtoms, nlist::NeighbourList;
#                       r0=1.0, e0=1.0, Rc=2.7, Lc=1.0, quantities="EG")`

# A fast LJ assembly which exploits the specific structure of the 
#  `MatSciPy.NeighbourList` to use  `@simd`, `@inbounds`, `@fastmath` macros 
# and the `AppleAccalerate` package.

# **This has not yet been tested for correctness!**
# """
# function lennardjones(at::ASEAtoms, nlist::NeighbourList;
#                       r0=1.0, e0=1.0, Rc=2.7, Lc = 1.0,  quantities="EG")
#     update!(nlist, at)
#     E = 0.0
#     r = nlist.Q['d']::Vector{Float64}
#     R = nlist.Q['D']::Array{Float64,2}
#     i = nlist.Q['i']::Vector{Int32}
#     t = Vector{Float64}(length(r))
#     c1 = Vector{Float64}(length(r))
#     # c2 = Vector{Float64}(length(r))
#     Rc_ = Rc-0.001

#     # c2 = exp(Lc ./ (Rc-r))
#     c2 = _fast_exponential(Lc, Rc, r)
    
#     @fastmath @inbounds @simd for n = 1:length(t) 
#         # c2[n] = exp( Lc / (Rc-r[n]) )
#         c1[n] = (1.0 / (1.0 + c2[n]))  * (r[n] >= Rc_)
#         t[n] = r0 / r[n]
#         t[n] = t[n]*t[n]*t[n]
#         t[n] = t[n]*t[n]
#         E = E + t[n] * (t[n]-2.0) * c1[n]
#     end
#     E *= e0

#     if 'G' in quantities
#         G = zeros(3, i[end])
#         @fastmath @inbounds @simd for n = 1:length(t)
#             c2[n] = -Lc * c2[n] * c1[n] / ( (Rc-r[n])*(Rc-r[n]) )
#             c2[n] = c2[n] * t[n] * (t[n]-2.0)
#             t[n] = e0 * 24.0 * t[n] * (t[n]-1.0) / (r[n]*r[n])
#             t[n] = t[n] * c1[n] + c2[n]
#             G[1,i[n]] = G[1,i[n]] + t[n] * R[n,1]
#             G[2,i[n]] = G[2,i[n]] + t[n] * R[n,2]
#             G[3,i[n]] = G[3,i[n]] + t[n] * R[n,3]
#         end 
#     end
    
#     ret = ()
#     for c in quantities
#         if c == 'E'
#             ret = tuple(ret..., E)
#         elseif c == 'G'
#             ret = tuple(ret..., G)
#         end
#     end
#     return ret
# end


end
