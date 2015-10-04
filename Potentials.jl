
"""
## module Potentials

### Summary

This module implements some basic interatomic potentials in pure Julia.The
implementation is done in such a way that they can be used either in "raw" form
or within abstract frameworks.

### Types



"""
module Potentials

using Prototypes

export PairPotential, ScalarFunction
export LennardJonesPotential, SWCutoff, ShiftCutoff, EAMPotential
export SimpleExponential, GuptaEmbed, GuptaPotential
export evaluate, evaluate_d, evaluate_dd, @D, @DD, @GRAD, grad
export call, cutoff
    
_eps_ = eps()
_eps2_ = 1e2

#########################################################
###         Pair Potentials                          
#########################################################

abstract ScalarFunction
abstract PairPotential <: ScalarFunction

"""`evaluate(pp::ScalarFunction, r)`: evaluate a scalar potential at `r`; 
typically a pair potential, where `r` may be a scalar or an array (of scalars)
"""
@protofun evaluate(pp::ScalarFunction, r)
"`evaluate_d(pp::PairPotential, r)`: first derivative of pair potential"
@protofun evaluate_d(pp::ScalarFunction, r)
"`evaluate_d2(pp::PairPotential, r)`: second derivative of pair potential"
@protofun evaluate_dd(pp::ScalarFunction, r)

# # vectorise these functions
# for f in (:evaluate, :evaluate_d, :evaluate_dd)
#     @eval $f{N}(pp::ScalarFunction, r::Array{Float64,N}) =
#         reshape(Float64[ $f(pp, s) for s in r ], size(r))
# end



### The next block of code is an attempt to use call-overloading and macros to
### allow the following syntax:
###   pp  ... a pair potential
###   r ... a length or array of lengths
###   ϕ = pp(r)
###   ϕ' = @D pp(r)
###   ϕ'' = @DD pp(r)
###
# create an alias to allow the potential to be called directly
#  make it an inline to ensure that nothing is lost here!
import Base.call
@inline call(pp::ScalarFunction, r) = evaluate(pp, r)
@inline call(pp::ScalarFunction, r, ::Type{Val{:D}}) = evaluate_d(pp, r)
@inline call(pp::ScalarFunction, r, ::Type{Val{:DD}}) = evaluate_dd(pp, r)

@inline call(pp::PairPotential, R, ::Type{Val{:GRAD}}) = grad(pp, R)
@inline call(pp::PairPotential, r, R, ::Type{Val{:GRAD}}) = grad(pp, r, R)

"""`grad(pp::PairPotential, R)`: if `R` is a 3 x N matrix (or 3 -vector) then
`grad` returns a 3x N array `G` whose columns are the gradient of the 
pair potential as a function of |R[:,i]|.

if `r = sqrt(sumabs2(R, 1))` is already available, then calling 
`grad(pp, r, R)` may be more efficient.

see also `@GRAD`.
"""
@inline grad(pp::PairPotential, R) = grad(pp, sqrt(sumabs(R, 1)), R)
@inline grad(pp::PairPotential, r, R) = R .* (evaluate_d(pp, r) ./ r)'


# next create macros that translate
"""`@D`

This macro can be used to evaluate the derivative of a potential, e.g., 
of a pair potential. For example, to compute the Lennard-Jones potential,
```
lj = LennardJonesPotential()
r = 1.0 + rand(10)
ϕ = lj(r)
ϕ' = @D lj(r)
```

see also `@DD`.
"""
macro D(fsig::Expr)
    @assert fsig.head == :call
    for n = 1:length(fsig.args)
        fsig.args[n] = esc(fsig.args[n])
    end
    push!(fsig.args, Val{:D})
    return fsig
end

"`@DD` : analogous to `@D`"
macro DD(fsig::Expr)
    @assert fsig.head == :call 
    for n = 1:length(fsig.args)
        fsig.args[n] = esc(fsig.args[n])
    end
    push!(fsig.args, Val{:DD})
    return fsig
end

"""`@GRAD` : If `p` is a `PairPotential`, and 
R a d x N array, then  `@GRAD p(R)` returns an array G
"""
macro GRAD(fsig::Expr)
    @assert fsig.head == :call 
    for n = 1:length(fsig.args)
        fsig.args[n] = esc(fsig.args[n])
    end
    push!(fsig.args, Val{:GRAD})
    return fsig
end


#########################################################
### Cut-off potentials

""" An `AbstractCutoff` is a type that, when evaluated will return the cut-off
variant of that potential. 

All `Cutoff <: AbstractCutoff` must have a constructor of the form
```
    Cutoff(pp::PairPotential, args...)
```
"""
abstract AbstractCutoff <: PairPotential

# abstract implementation for getting the cut-off radius, assuming that
# it is stored as p.Rc
cutoff(p::AbstractCutoff) = p.Rc


########################

"""`cutsw(r, Rc, Lc)`

Implementation of the C^∞ Stillinger-Weber type cut-off potential
    1.0 / ( 1.0 + exp( lc / (Rc-r) ) )

`d_cutinf` implements the first derivative
"""
@inline cutsw(r, Rc, Lc) =
    1.0 ./ ( 1.0 + exp( Lc ./ ( max(Rc-r, 0.0) + _eps_ ) ) )

"derivative of `cutsw`"
@inline function cutsw_d(r, Rc, Lc)
    t = 1 ./ ( max(Rc-r, 0.0) + _eps_ )   # a numerically stable (Rc-r)^{-1}
    e = 1.0 ./ (1.0 + exp(Lc * t))                     # compute exponential only once
    return - Lc * (1.0 - e) .* e .* t.^2    
end

"""
`type SWCutoff`: SW type cut-off potential with C^∞ regularity
     1.0 / ( 1.0 + exp( Lc / (Rc-r) ) )

This is not very optimised: one could speed up `evaluate_d` significantly
by avoiding multiple evaluations.
"""
type SWCutoff <: AbstractCutoff
    pp::PairPotential
    Rc::Float64
    Lc::Float64
end

@inline evaluate(p::SWCutoff, r) =
    p.pp(r) .* cutsw(r, p.Rc, p.Lc)

@inline evaluate_d(p::SWCutoff, r) =
    p.pp(r) .* cutsw_d(r, p.Rc, p.Lc) + (@D p.pp(r)) .* cutsw(r, p.Rc, p.Lc)


########################

"""
`ShiftCutoff` : takes the pair-potential and shifts and truncates it
    f_cut(r) = (f(r) - f(rcut)) .* (r <= rcut)

Default constructor is
```
    ShiftCutoff(Rc, pp)
```
"""
type ShiftCutoff <: AbstractCutoff
    pp::PairPotential
    Rc::Float64
    Jc::Float64
end

ShiftCutoff(pp, Rc) = ShiftCutoff(pp, Rc, pp(Rc))

@inline evaluate(p::ShiftCutoff, r) = (p.pp(r) - p.Jc) .* (r .<= p.Rc)

@inline evaluate_d(p::ShiftCutoff, r) = (@D p.pp(r)) .* (r .<= p.Rc)



########################





#########################################################
###         Lennard-Jones potential


"""`LennardJonesPotential <: PairPotential`

Implementation of the lennard-Jones potential    
   e0 * ( (r0/r)¹² - 2 (r0/r)⁶ )

Constructor: `LennardJonesPotential(;r0=1.0, e0=1.0)`
"""
type LennardJonesPotential <: PairPotential
    r0::Float64
    e0::Float64
end

LennardJonesPotential(; r0=1.0, e0=1.0) = LennardJonesPotential(r0, e0)


@inline evaluate(p::LennardJonesPotential, r) =
    p.e0 * ((p.r0./r).^12 - 2.0 * (p.r0./r).^6)
@inline evaluate_d(p::LennardJonesPotential, r) =
    -12.0 * p.e0 * ((p.r0./r).^12 - (p.r0./r).^6) ./ r


#########################################################
###         Simple Exponential
"""`SimpleExponential`

   A exp( B (r/r0 - 1) )
}
   TODO: this is acting as if it had 3 parameters, but there are actually only
      two - rewrite accordingly?
"""
type SimpleExponential <: PairPotential
    A::Float64
    B::Float64
    r0::Float64
end
@inline evaluate(p::SimpleExponential, r) =
    p.A * exp( p.B * (r/p.r0 - 1.0) )
@inline evaluate_d(p::SimpleExponential, r) =
    p.A*p.B/p.r0 * exp( p.B * (r/p.r0 - 1.0) )


#########################################################
###         Gupta Potential

abstract SitePotential

type EAMPotential <: SitePotential
    V::PairPotential
    rho::PairPotential
    embed::ScalarFunction
end

"embedding function for the Gupts potential"
type GuptaEmbed <: ScalarFunction
    xi
end
@inline evaluate(p::GuptaEmbed, r) = p.xi * sqrt(r)
@inline evaluate_d(p::GuptaEmbed, r) = 0.5*p.xi ./ sqrt(r)

"""`GuptaPotential`:
    E_i = A ∑_{j ≠ i} v(r_ij) - ξ ∑_i √ ρ_i
        v(r_ij) = exp[ -p (r_ij/r0 - 1) ]
        ρ_i = ∑_{j ≠ i} exp[ -2q (r_ij / r0 - 1) ]
"""
GuptaPotential(A, xi, p, q, r0, TC::Type, TCargs...)  =
    EAMPotential( TC( SimpleExponential(A, p, r0), TCargs... ),      # V
                  TC( SimpleExponential(1.0, 2*q, r0), TCargs...),   # rho
                  GuptaEmbed( xi ) )                                 # embed


end
