
push!(LOAD_PATH, "/Users/ortner/gits/Atoms.jl")

using TestAtoms
using TightBinding, NRLTB
using Potentials
using ASE
using JLD, PyCall
@pyimport ase

infile = ARGS[1]
outfile = ARGS[2]

# load data from the file
C, X, pbc = load(infile, "C", "X", "pbc")
# construct an Atoms object
at = ASEAtoms( ase.Atoms("Si$(size(X,2))") )
set_cell!(at, C)
set_positions!(at, X)
set_pbc!(at, pbc)

# create a TB model
tbm = NRLTB.NRLTBModel(elem = NRLTB.Si_sp)
H, M = TightBinding.hamiltonian(at, tbm)
H = full(H)
M = full(M)

# save them
save(outfile, "H", H, "M", M)
