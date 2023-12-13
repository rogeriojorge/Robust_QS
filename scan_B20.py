#!/usr/bin/env python
import time
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from neat.fields import StellnaQS
from neat.tracing import ChargedParticleEnsemble, ParticleEnsembleOrbit, ParticleOrbit, ChargedParticle

### Make a single particle orbit of a super banana
# For B=B00+B10cos(theta), E/μ of super bananas ≈ B00 + 0.8B10.

## INPUT PARAMETERS
Rmajor_ARIES = 7.7495  # Major radius double double of ARIES-CS
Rminor_ARIES = 1.7044  # Minor radius
s_initial = 0.5
B0 = 5.3267  # Tesla, magnetic field on-axis
energy = 3.52e6  # electron-volt
charge = 2  # times charge of proton
mass = 4  # times mass of proton
ntheta = 11  # resolution in theta
nphi = 5  # resolution in phi
nlambda_trapped = 15  # number of pitch angles for trapped particles
nlambda_passing = 3  # number of pitch angles for passing particles
nsamples = 10000  # resolution in time
tfinal = 1e-4  # seconds
nthreads = 4
stellarator_indices = [1,2]#[1,2,3,4,5]
constant_b20 = False
B20_scaling_array = np.linspace(-30,30,12)#np.logspace(-1.5, 2.0, num=12, base=10.0)
## MPI STUFF
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
num_elements = len(B20_scaling_array)
elements_per_process = num_elements // size
start_index = rank * elements_per_process
end_index = start_index + elements_per_process
if rank == size - 1: end_index = num_elements
local_B20_scaling_array = B20_scaling_array[start_index:end_index]
## Do the loop
for stellarator_index in stellarator_indices:
    if rank==0: print(f"Stellarator {stellarator_index}/{len(stellarator_indices)}")
    g_field_basis = StellnaQS.from_paper(stellarator_index, B0=B0, nphi=71)
    g_field = StellnaQS(
        rc=g_field_basis.rc * Rmajor_ARIES, zs=g_field_basis.zs * Rmajor_ARIES,
        etabar=g_field_basis.etabar / Rmajor_ARIES,
        B2c=g_field_basis.B2c * (B0 / Rmajor_ARIES / Rmajor_ARIES),
        B0=B0,nfp=g_field_basis.nfp,order="r3",nphi=g_field_basis.nphi)
    basis_B20 = g_field.B20
    g_field.constant_b20 = constant_b20
    g_particle_ensemble = ChargedParticleEnsemble(r_initial=Rminor_ARIES * np.sqrt(s_initial),r_max=Rminor_ARIES,energy=energy,charge=charge,mass=mass,ntheta=ntheta,nphi=nphi,nlambda_trapped=nlambda_trapped,nlambda_passing=nlambda_passing)
    g_particle = ChargedParticle(r_initial=Rminor_ARIES * np.sqrt(s_initial),energy=energy,charge=charge,mass=mass,
                                 Lambda=0.95,theta_initial=1.5,phi_initial=0.1,vpp_sign=-1)
    local_loss_fractions = []
    local_B20_arrays = []
    for i, B20_scaling in enumerate(local_B20_scaling_array):
    # for i, B20_scaling in enumerate(B20_scaling_array):
        start_time = time.time()
        g_field.B20 = B20_scaling*(basis_B20-np.mean(basis_B20))+np.mean(basis_B20)
        local_B20_arrays.append(g_field.B20)
        g_orbits = ParticleEnsembleOrbit(g_particle_ensemble,g_field,nsamples=nsamples,tfinal=tfinal,nthreads=nthreads,constant_b20=constant_b20)
        g_orbits.loss_fraction(r_max=Rminor_ARIES, jacobian_weight=True)
        local_loss_fractions.append(g_orbits.loss_fraction_array[-1]*100)
        total_time = time.time() - start_time
        print(f"  B20 scaling={B20_scaling:.1e} and {g_orbits.nparticles} particles took {total_time:.1f}s and loss fraction is {g_orbits.loss_fraction_array[-1]*100:.1f}%")
    #     try:
    #         g_orbit = ParticleOrbit(g_particle,g_field,nsamples=nsamples,tfinal=tfinal,constant_b20=constant_b20)
    #         total_time = time.time() - start_time
    #         print(f"  #{i+1}/{len(B20_scaling_array)} with B20 scaling={B20_scaling:.1e} took {total_time:.1e}s, final s is {(g_orbit.r_pos[-1]/Rminor_ARIES)**2:.2f}, max energy conservation error is {np.max(np.abs((g_orbit.total_energy-g_orbit.total_energy[0])/g_orbit.total_energy[0])):.1e} and max momentum conservation error is {np.max(np.abs((g_orbit.p_phi-g_orbit.p_phi[0])/g_orbit.p_phi[0])):.1e}")
    #         g_orbit.plot()
    #     except Exception as e:
    #         total_time = time.time() - start_time
    #         print(f"  #{i+1}/{len(B20_scaling_array)} with B20 scaling={B20_scaling:.1e} took {total_time:.1e}s and failed with error {e}")
    #     g_field.B20 = basis_B20
    # exit()
    local_loss_fractions = np.array(local_loss_fractions)
    local_B20_arrays = np.array(local_B20_arrays)
    # # plt.figure()
    if rank == 0:
        all_loss_fractions = np.empty(num_elements, dtype=float)
        all_B20_arrays = np.empty((num_elements,len(g_field.B20)), dtype=float)
    else:
        all_loss_fractions = None
        all_B20_arrays = None
    comm.Gather(local_loss_fractions, all_loss_fractions, root=0)
    comm.Gather(local_B20_arrays, all_B20_arrays, root=0)
    if rank == 0:
        plt.xlabel("B20 scaling")
        plt.ylabel("Loss Fraction (%)")
        line, = plt.plot(B20_scaling_array, all_loss_fractions, label=f"From_paper {stellarator_index} with B20 mean={np.mean(g_field.B20):.3f} and variation={g_field.B20_variation:.3f}")
        plt.legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(f"scan_B20.png")
MPI.Finalize()