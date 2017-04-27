module AtJuLIP

import JuLIP
import AtomsInterface
import ASE

import TightBinding
JTB = TightBinding

include("./TightBinding.jl")
ATB = TightBinding


type JuLIPCalculator
   calc::AtomsInterface.AbstractCalculator
end


j2a(at::JuLIP.ASE.ASEAtoms) = ASE.ASEAtoms(at.po)

JuLIP.energy(calc::JuLIPCalculator, at::JuLIP.AbstractAtoms) =
   AtomsInterface.potential_energy(j2a(at), calc.calc)

JuLIP.forces(calc::JuLIPCalculator, at::JuLIP.AbstractAtoms) =
   AtomsInterface.forces(j2a(at), calc.calc)

JuLIP.Potentials.site_energy(calc::JuLIPCalculator, at::JuLIP.AbstractAtoms, i0::Int) =
   AtomsInterface.site_energy(i0, j2a(at), calc.calc)

JuLIP.Potentials.site_energy_d(calc::JuLIPCalculator, at::JuLIP.AbstractAtoms, i0::Int) =
   scale!(AtomsInterface.site_forces(i0, j2a(at), calc.calc), -1)


# TB Model datastructure

type JuLIPTB
   tbm::ATB.TBModel
end

# import JTB.hamiltonian
JTB.hamiltonian(tbm::JuLIPTB, atm::JuLIP.AbstractAtoms, args...) =
   ATB.hamiltonian(j2a(atm), tbm.tbm, args...)


function JuLIPTB(s::Symbol; nkpoints = (0,0,0))
   if s == :Si || s == :Si4
      tbm = ATB.NRLTB.NRLTBModel(elem = ATB.NRLTB.Si_sp, nkpoints = (0,0,0))
      return JuLIPTB(tbm)
   elseif s == :Si9
      tbm = ATB.NRLTB.NRLTBModel(elem = ATB.NRLTB.Si_spd, nkpoints = (0,0,0))
      return JuLIPTB(tbm)
   end

   error("unkown symbol")
end


end
