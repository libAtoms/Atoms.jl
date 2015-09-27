

# commented, since it is not actually part of ASE
#    move to a separate module?
#
# @pyimport matscipy.neighbours as matscipy_neighbours


# function neighbours(atoms::ASEAtoms, quantities, cutoff)
#     results = matscipy_neighbours.neighbour_list(quantities,
#                                                  pyobject(atoms),
#                                                  cutoff)    
#     results = collect(results) # tuple -> array so we can change in place
#     # translate from 0- to 1-based indices
#     for (idx, quantity) in enumerate(quantities)
#         if quantity == 'i' || quantity == 'j'
#             results[idx] += 1
#         end
#     end
#    return results
# end
