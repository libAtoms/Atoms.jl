

push!(LOAD_PATH, ".", "..")

using TestAtoms, Potentials


# println("Testing Lennard-Jones Potential")
# test_ScalarFunction(LennardJonesPotential(), 0.9 + rand(20))

# println("Testing Lennard-Jones Potential with cutoff")
# test_ScalarFunction(SWCutoff(LennardJonesPotential(), 2.1, 1.0), 1.5 + rand(20))

# println("Testing the Components of the Gupta Potential")
# println("1. the SimpleExponential: ")
# test_ScalarFunction(SimpleExponential(1.234, 2.345, 1.321), 0.5 + rand(20))
# println("2. the EmbeddingFunction: ")
# test_ScalarFunction(GuptaEmbed(1.234), 0.5 + rand(20))


# p = SimpleExponential(1.234, 2.345, 1.321)
# r = 0.1 + rand(5)
# println(  @D p(r) )
