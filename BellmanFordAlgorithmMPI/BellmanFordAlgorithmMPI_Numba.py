from mpi4py import MPI
import numpy as np
import sys
import random as rd
import argparse
import re
from numba import njit


MAX_NUMBER = np.inf

@njit
def matrix_processing(loc_start:int, N:int,loc_mat:np.ndarray,dist:np.ndarray,loc_has_change:np.ndarray):
  for u in range(loc_mat.shape[0]):
    for v in range(N):
      weight = loc_mat[u, v]
      if weight < MAX_NUMBER:
        if dist[loc_start+u] + weight < dist[v]:
          dist[v] = dist[loc_start+u] + weight
          loc_has_change[0] = True
  return dist , loc_has_change


def bellman_ford(
  loc_start:int,#loc_end:int,  
  comm:MPI.Intracomm, N:int, 
  loc_mat:np.ndarray,dist:np.ndarray, 
  has_negative_cycle:bool):
    #loc_dist = np.copy(dist)
    loc_iter_num = 0
    for iter in range(N-1):
        loc_has_change = np.zeros(1, dtype=bool)
        loc_iter_num = loc_iter_num + 1

        dist , loc_has_change = matrix_processing(loc_start, N,loc_mat,dist,loc_has_change)
        # for u in range(loc_mat.shape[0]):
        #   for v in range(N):
        #     weight = loc_mat[u, v]
        #     if weight < MAX_NUMBER:
        #       if dist[loc_start+u] + weight < dist[v]:
        #         dist[v] = dist[loc_start+u] + weight
        #         loc_has_change[0] = True
        comm.Allreduce(
          MPI.IN_PLACE,[loc_has_change, 1, MPI.CXX_BOOL],op=MPI.LOR)
        if loc_has_change is False:
          break
        comm.Allreduce(MPI.IN_PLACE,dist, MPI.MIN)
    # do one more step
    if loc_iter_num == N - 1:
      loc_has_change = np.zeros(1, dtype=bool)
      for u in range(loc_mat.shape[0]):
        for v in range(N):
          weight = loc_mat[u,v]
          if weight < MAX_NUMBER:
            if dist[loc_start+u] + weight < dist[v]:
              dist[v] = dist[loc_start+u] + weight
              loc_has_change[0] = True
              break
      comm.Allreduce([has_negative_cycle, 1, MPI.CXX_BOOL],[loc_has_change,1, MPI.CXX_BOOL],MPI.LOR)
    if my_rank != 0:
      del loc_mat
      del dist


################################################################################

########################### Start of the function #############################
comm = MPI.COMM_WORLD # processor group
p = comm.Get_size() # we get the number of processors of participants
my_rank = comm.Get_rank()


##### initialization
if my_rank ==0:
  t1 = MPI.Wtime()
has_negative_cycle = np.zeros(1, dtype=bool)  

loc_start = np.array(0,dtype=np.int32)
loc_end  = np.array(0,dtype=np.int32)

parser = argparse.ArgumentParser()
parser.add_argument('file', type=argparse.FileType('r',encoding='utf-8'))
args = parser.parse_args()
# if my_rank == 0:
with args.file as file:
  numpyMatrixPattern = []
  for line in file.readlines():
    if len(line.split()) == 1:
      N = np.array(int(line.split()[0]))
    else:
      newLIST = list(map(int,re.sub(r'[A-Za-z\\]+|\ufeff',' ',line).split()))
      numpyMatrixPattern.append(newLIST)
    
  mat = np.array(numpyMatrixPattern)
  dist = np.empty(N,dtype=np.float64)
  
  for i in range(N):
      dist[i] = MAX_NUMBER
  dist[0] = 0
  if my_rank == 0: 
    
    ave, res = np.divmod(N, p)
    rCounts = np.empty(p+1,dtype=np.int32)
    displs = np.empty(p+1,dtype=np.int32)
    rCounts[0] = displs[0] = 0
    for k in range(1,p+1):
      if k < 1 + res:
        rCounts[k] = ave + 1
      else:
        rCounts[k] = ave
      displs[k] = displs[k - 1] + rCounts[k - 1] if k !=0 else 0
    rCounts = np.delete(rCounts, 0)
    displs = np.delete(displs, 0)
    start = displs
    end = displs + rCounts
  else:
    rCounts = None
    displs = None
    start = None
    end = None
  
comm.Scatterv(
  [start,    np.ones(p), np.array(range(p)), MPI.INT],
  [loc_start,            1,                  MPI.INT],
  root=0
)

comm.Scatterv(
  [end,    np.ones(p), np.array(range(p)), MPI.INT],
  [loc_end,           1,                   MPI.INT],
  root=0
)

#if my_rank == 0 and p == 1: print(f"MAT: \n{mat}\n")

loc_mat = np.copy(mat[loc_start:loc_end,:])
del mat
####calculation#####
bellman_ford(loc_start,comm,N,loc_mat,dist,has_negative_cycle)
#end timer
if my_rank == 0:
  t2 = MPI.Wtime()
  print(f'{str(t2 - t1).replace(".",",")}')
  # print(f'Time(s): {t2 - t1}')
  # print(f'Number of nodes: {N}')
  # print(f'After: dist \n{dist}')








