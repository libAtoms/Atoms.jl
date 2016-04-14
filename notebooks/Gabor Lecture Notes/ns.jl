module ns

import pbc
#@pyimport numpy

global energymodel # global pointer to the energy model used. it is set in the nestedsampling() function. 

#
# an excursion under a given energy limit, given by Elimit
# returning the new postiions, the number of accepted moves and the energy history
# we get the number of spatial dimensions and number of particles from the input vector x
# the particles are moved one by one in random order, nsteps is the number of sweeps, so the
# total number of MC moves of nsteps*(number of particles)
#
function walk(Elimit, x, nsteps, stepsize)
    dim = size(x,1)         # first dimension of vector is the number of spatial dimensions
    N = size(x,2)           # second dimension of vector gives the number of atoms
    E = energymodel.energy(x) # initial energy

    naccepted = 0           # initialis counter
    xwalk = copy(x)         # make a copy of the configuration
    ewalk = zeros(nsteps*N) # initialise energy trajectory
    order = [[randperm(N) for i=1:nsteps]...] # create series of random permutation of particle order, so each particle attempts to move
    # order = rand(1:N,N*nsteps); # alternatively, this creates moves in which particle choices are independent
    for s = 1:nsteps*N # main loop
        ewalk[s] = E;  # store current energy for reporting purposes
        at = order[s]; # pick the next particle to move      
        dx = (rand(dim)-0.5)*stepsize; # propose a random step
        dEw = energymodel.energy1(xwalk, at, dx); # calculate the energy change of the move
        #println("$at $dx $dEw  $(E+dEw) $Elimit")
        if E+dEw < Elimit      # test if the move is acceptable
            xwalk[:,at] += dx; # accept the move
            E += dEw;          # update the energy
            naccepted += 1;    # update counter
            if pbc.L > 0.0     # if we have periodic boundary conditions, remap the particles into the cell
                pbc.map_into_cell!(xwalk);
            end
        end
    end
    return (xwalk, naccepted/(nsteps*N), ewalk); # return the moved configuration, the number of accepted moves and the energy history
end

#
# making test excursions for the purposes of adjusting the step size
# and trying to achieve an acceptance ratio between 0.2 and 0.8
# these excursions solely for the purpose of determining the step size and
# the configurations are thrown away and do not participate in the sampling. 
#
function adjust_stepsize(x, stepsize; diagnostics=false)
    accept_ratio = 1.0;         # initialise the acceptance ratio measure
    e = energymodel.energy(x)     # compute the first energy
    if diagnostics
        println("adjusting step size with pilot run. current stepsize=$stepsize")
    end
    for n=1:50                 # outer loop for adjusting the step size, at most 200 iterations
        xinit = copy(x);        # make a copy of the initial 
        (xnew, accept_ratio) = walk(e, xinit, 50, stepsize);
        if accept_ratio > 0.8 
            stepsize *= 1.2;
        elseif accept_ratio < 0.2
            stepsize /= 1.2;
        else
            break
        end
        if pbc.L > 0 && stepsize > pbc.L/2
            break
        end
    end
    if accept_ratio < 0.2
        error("failed to find good step size. stepsize=$stepsize, accept_ratio = $accept_ratio ")
    end
    if diagnostics
        println("Acceptance ratio = $accept_ratio, new stepsize=$stepsize")
    end
    return stepsize;
end

#
# A full NS run
#
# model:        a module that constitutes an energy model, e.g. "lj"
# dim:          dimensionality of physical space (e.g. 2 or 3)
# Nat:          number of particles
# Nw:           number of walkers
# Niter:        number of NS iterations 
# Nsteps:       number of steps (sweeps) in the excursions
# alwaysclone:  if set to true, always throw away the top walker and clone a new one 
# diagnostics:  if set to true, return various measurements about the run
# extrawalks:   if true, many samples are walked at each iteration
# walksource:   if true, walk the source of a clone as well
# seed:         random seed
# init_size:    range of randomisation for initialising walkers. 
function nestedsampling(model, dim, Nat, Nw, Niter, Nsteps; seed=1, init_size=2.0, alwaysclone=true, diagnostics=false, walksource=true, extrawalks=false, init_step_size=0.0)
    global energymodel
    energymodel = model          # set global pointer to the energy model we got as input
    println(energymodel.info())  # print some info about the model 
    srand(seed)                  # initialise random number generator

    # zero out lots of variables, both dynamical and diagnostic
    xs = zeros(dim, Nat, Nw)            # configuration coordinates xs(dim, Nat, Nw)
    Elist = zeros(Nw)                   # dynamically updated list of energies
    xhistory = zeros(Niter, dim*Nat)    # history of "outermost" configurations
    Ehistory = zeros(Niter)             # history of "outermost" energies - this is the main output of NS
    accept_ratio_history = zeros(Niter) # history of MC acceptance ratios
    stepsize_history = zeros(Niter)     # history of MC step sizes
    age = rand(Nw, 1)                   # the 'age' variable is a dynamically updated list that stores the number of iterations since a configuration was cloned
    clone = zeros(Bool,Niter)           # history of decisions whether a new configuration was cloned randomly or just continued from the outermost one
    emove = zeros(Niter)                # 
    ewalk = zeros(Nsteps)

    # initialise Nw sane walker configurations
    Elim = Nat*500.0; # set initial energy limit. needs to be high enough that everything is accessible, but too high a value leads to waste
    w = 0;
    while w < Nw
        x = (rand(dim,Nat)-0.5)*init_size; # propose a configuration
        if energymodel.energy(x) < Elim    # check if energy less than limit
            w += 1;                        
            xs[:,:,w] = x;                 # store configuration
        end
    end
    Elist = [energymodel.energy(xs[:,:,i]) for i=1:Nw]; # calculate initial list of energies

    
    if init_step_size == 0.0
     #stepsize = adjust_stepsize(xs[:,:,1], 1e-6); # set the first step size
     stepsize = 1.0; # this is just a wild guess, maybe should scale with the periodic system size? 
    else
        stepsize = init_step_size
    end
    
    #
    # outer NS loop
    #
    # this is the main NS loop which descends through decreasing energy levels and collectst he density of states. 
    for i = 1:Niter

        # find new outermost sample, i.e. the one with the highest energy
        (Elim,outer) = findmax(Elist);

        if diagnostics
            println("NS iteration $(i) energy of outermost walker $(outer) is $(Elim)")
        end		

        Ehistory[i] = Elim; # store the just found highest energy in our history that we will return
        xhistory[i,:] = reshape(xs[:,:,outer], (1,dim*Nat)) # store associated configuration also
        
        if diagnostics
            println("NS iteration $(i) energy limit=$(Elim)")
        end		

        # the code below decides whether we want to clone a new sample to replace the
        # outermost one, or we should just continue the outermost sample with the new energy limit
        # if we never clone, then local minima will never lose their trapped samples, causing huge systematic errors
        # if we always clone, then deep local minima will go "extinct" by chance before they should. 
        # in materials systems, it _seems_ that always cloning is still the better choice
        #
        oldest = indmax(age);             # find the sample which has walked the most (called "age")
        if alwaysclone || outer == oldest # clone if we are either always cloning OR the outermost sample is also the oldest. 
            # throw away outer sample, clone random new one and walk it
            source = mod1(outer+rand(1:Nw-1),Nw)
            xs[:,:,outer] = xs[:,:,source];
            
            age[outer] = floor(age[source])+rand(); # the cloned sample copies the age of its source plus a random number (so that we can distinguish the two copies)
            clone[i] = true;
        else
            clone[i] = false;
        end

        # adjust step size every now and then 
          if i%10 == 0
               stepsize = adjust_stepsize(xs[:,:,outer], stepsize; diagnostics=diagnostics) # this is calling a sophisticated "pilot run"-based stepsize adjuster. 
              #if accept_ratio_history[i-1] < 0.2               # this is a very simple step size adjuster
              #    stepsize /= 2;
              #end
          end
        

        # select samples to move and store them in the list called ws
        ws = [outer]; # the just-cloned or outermost sample (even if it wasn't replaced by a clone) must move
        if extrawalks # we may want to move other samples
            if walksource # walk the source of the clone? 
                append!(ws, [source])
            end
            l = setdiff([1:Nw], ws);
            for li=1:(walksource?14:15)      # if we are walking the source, then select 14 others, if not, select 15 others
                r = rand(1:length(l));       # pick a random sample to walk
                append!(ws, [splice!(l,r)]); # add it to the walk-list
            end
        end
        
        # walk samples in walk-list ws
        accept_ratio = zeros(length(ws));
        local ewalk
        for n=1:length(ws)
            (xs[:,:,ws[n]], accept_ratio[n], ewalk) = walk(Elim, xs[:,:,ws[n]], int(Nsteps/(extrawalks?16:1)), stepsize); # do the walk!            
            Elist[ws[n]] = energymodel.energy(xs[:,:,ws[n]]); # store final state of walked sample
            age[ws[n]] += 1; # increment age of all samples that walked
        end
	    if diagnostics
            println("Walked ", ws);
        end	  
      
        # store data for post mortem analysis   
        accept_ratio_history[i] = mean(accept_ratio);
        stepsize_history[i] = stepsize;

    end
        
    return(xhistory, Ehistory, xs, accept_ratio_history, stepsize_history);

end

# compute thermodynamic quantities for an array of temperatures
# given an energy history number of atoms and samples
function thermo(T, E, Nat, Nw)
    Niter = length(E);
    loga = log(Nw)-log(Nw+1); # log of the phase space compression ratio
    logZ=zeros(T); U=zeros(T); Cv=zeros(T);
    for i=1:length(T)
        beta = 1.0/T[i];
        shift = maximum([1:Niter]*loga-beta*E); # shifting is to avoid over/under-flow
        p = exp([1:Niter]*loga-beta*E-shift);
        Z = sum_kbn(p);
        Upot = sum_kbn(p.*E)/Z;
        Cv[i] = 3.0*Nat/2.0 + beta^2*(sum_kbn(p.*(E.^2))/Z-Upot^2);
        U[i] = 3.0/2.0*Nat/beta+Upot;
        #compute log Z, and unshift
        logZ[i] = log(Z) + shift
    end
    return (logZ, U, Cv);
end


end
