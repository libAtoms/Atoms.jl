#myplot()

PyPlot.matplotlib[:rc]("figure",figsize=(6,4))

using Interact
using Optim
using ForwardDiff

using tip3p
using grids


@pyimport IPython.display as ipydisp # enable showing web pages in IFrames, and direct HTML for imolecule
@pyimport quippy # quippy is the python wrapper for QUIP. it is somewhat ASE compatible

function miframe(url)
    ipydisp.IFrame(url, width=800, height=500)
end

# for imolecule 
@pyimport ase.io.cif as asecif
@pyimport ase.io.xyz as asexyz
@pyimport imolecule
@pyimport io as pio

# for preconditioned minimization
@pyimport ase.optimize.precon as ase_precon
@pyimport ase.constraints as ase_const

# shorthand for imolecule
function imolecule_draw(atoms)
    sio = pio.BytesIO()
    asexyz.write_xyz(sio, [atoms])
    ipydisp.HTML(imolecule.draw(sio[:getvalue](), format="xyz",  camera_type="orthographic", size=(300,200), display_html=false))
end

function imolecule_draw(atomlist::Array{Any})
    s = ""
    for atoms in atomlist
        sio = pio.BytesIO()
        asexyz.write_xyz(sio, [atoms])
        s = join(s,imolecule.draw(sio[:getvalue](), format="xyz",  camera_type="orthographic", size=(300,200), display_html=false))
    end
    ipydisp.HTML(s)
end

# for drawing patches in matplotlib
@pyimport matplotlib.patches as patch
function patch_h2o(p = [0.0,0.0])
    ax = gca()
    ax[:set_aspect]("equal")
    ax[:add_artist](patch.Rectangle([0,-0.05]+p, 0.9,0.1, fc="grey", fill=true))
    ax[:add_artist](patch.Rectangle([0.05,-0.05]+p, 0.9,0.1, 104, fc="grey", fill=true))
    ax[:add_artist](patch.Circle([0,0]+p, 0.25, fc="red", fill=true))
    ax[:add_artist](patch.Circle([0.96,0.0]+p, 0.15, fc="white", fill=true))
    ax[:add_artist](patch.Circle([-0.24,0.93]+p, 0.15, fc="white", fill=true))
end

# water monomer geometry
θ0 = 104.4/180*π
r0 = 0.96
h2o_p0 = [0 0 0 ; r0 0 0 ; r0*cos(θ0) r0*sin(θ0) 0]

function make_h2o(p=[0.0 0.0 0.0])
    h2o = quippy.Atoms(n=3, lattice=[10 0 0 ; 0 10 0 ; 0 0 10], numbers=[8,1,1], positions=h2o_p0+repeat(p,outer=[3,1]));
end

function make_h4o2(p = [0.0 0.0 0.0])
    h4o2 = make_h2o(p)
    h2o = make_h2o(p+[2.5 0 0]) 
    h4o2[:add_atoms](h2o)

    h4o2
end

function make_h12o6()
    quippy.Atoms(n=18, lattice=[20 0 0 ; 0 20 0 ; 0 0 20], numbers=repeat([8,1,1], outer=[6]),
                        positions=[ h2o_p0 ; 
                                    h2o_p0 +  repeat([3.5 0.0 0.0], outer=[3,1]) ; 
                                    h2o_p0 +  repeat([5.0 2.5 0], outer=[3,1]) ; 
                                    h2o_p0 +  repeat([3.5 5.0 0.0], outer=[3,1]) ; 
                                    h2o_p0 +  repeat([0.0 5.0 0.0], outer=[3,1]) ; 
                                    h2o_p0 +  repeat([-1.5 2.5 0], outer=[3,1])]);
end

# radius of gyration of water hexamer
function h12o6_Rg(x)
    rO = reshape(x, (3,18))[:,1:3:18]; mr = mean(rO, 2)
    sqrt(mean(sum((rO-repeat(mr, inner=[1,6])).^2,1)))
end

# velocity Verlet LAngevin integrator

function velocityVerletLangevin!(q::Vector{Float64}, p::Vector{Float64}, Fq::Vector{Float64}, F, m::Vector{Float64}, h::Float64, kT::Float64, gamma::Float64, ph2::Vector{Float64}, qh2::Vector{Float64})
    ph2[:] = p + h/2*Fq
    qh2[:] = q + h/2*ph2./m
    ph2[:] = exp(-gamma*h)*ph2 + sqrt(kT*(1-exp(-2*gamma*h)))*randn(length(q)).*sqrt(m)
    q[:]  = qh2 + h/2*ph2./m
    Fq[:] = F(q)
    p[:]  = ph2 + h/2*Fq
    return
end
    ;