import sys
import time
import emcee
import numpy as np
from schwimmbad import MPIPool
from mpi4py import MPI

#global_count = 0

def log_prob(theta):
    t = time.time() + 5
    while True:
        if time.time() >= t:
            break
    comm = MPI.COMM_WORLD
    mpirank = comm.Get_rank()
    processor_name = MPI.Get_processor_name()
    print(f'Running {mpirank:3d}, {processor_name}: Running MPI')
    #global_count += 1
    return -0.5*np.sum(theta**2)

with MPIPool() as pool:
    comm = MPI.COMM_WORLD
    mpirank = comm.Get_rank()
    processor_name = MPI.Get_processor_name()
    # print(f'Running {mpirank:3d}, {processor_name}: Running MPI')
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    np.random.seed(42)
    initial = np.random.randn(80, 5)
    nwalkers, ndim = initial.shape
    nsteps = 10

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool)
    start = time.time()
    sampler.run_mcmc(initial, nsteps)
    end = time.time()
    print(end - start)
