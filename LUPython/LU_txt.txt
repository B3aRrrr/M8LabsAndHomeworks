from mpi4py import MPI
import numpy as np
import sys
import random as rd
import argparse
import re

def forwardGauss(  diag_ref:int, row_ref:np.array,  A_part :np.ndarray, L_part :np.ndarray,  N:int, rankRowList:np.array,  RANK:int):
    assert A_part.shape[0] == len(rankRowList)
    for i in range(A_part.shape[0]):
      if rankRowList.tolist()[i] > diag_ref:             
        a1 =  A_part[i,diag_ref].copy()
        a2 =  row_ref[0,diag_ref].copy()
        k = a1 / a2
        L_part[i,diag_ref]  = L_part[i,diag_ref] + k
      else:
        k = 0
      A_part[i,:] =  A_part[i,:] - k * row_ref[0,:]
    return L_part, A_part

### начало функции
comm = MPI.COMM_WORLD # группа процессоров
numProcs = comm.Get_size() # получаем количество процессоров участников
rank = comm.Get_rank()

### отправляем каждому rank информацию о делении матрицы

A = None
L = None
A_part = None # заготовки для A
L_part = None # заготовки для L
N = np.array(0,dtype=np.int32)

if rank == 0:
  #### Второй вариант: считываение txt- файла
  parser = argparse.ArgumentParser()
  parser.add_argument('file', type=argparse.FileType('r',encoding='utf-8'))
  args = parser.parse_args()

  with args.file as file:
      numpyMatrixPattern = []
      for line in file.readlines():
        newLIST = list(
          map(
            float,
            re.sub(r'[A-Za-z\\]+|\ufeff',' ',line).split()
            )
          )
          
        
        #print(newLIST)
        numpyMatrixPattern.append(newLIST)
      A = np.array(numpyMatrixPattern)
      N = np.array(A.shape[0],dtype=np.int32)
      L = np.eye(N)

  ave, res = np.divmod(N, numProcs-1)
  rCounts = np.empty(numProcs,dtype=np.int32)
  displs = np.empty(numProcs,dtype=np.int32)
  rCounts[0] = displs[0] = 0
  for k in range(1,numProcs):
    if k < 1 + res:
      rCounts[k] = ave + 1
    else:
      rCounts[k] = ave
    displs[k] = displs[k - 1] + rCounts[k - 1]
else:
  A = None
  L = None
  rCounts = None
  displs = None


comm.Bcast(buf=[N,1,MPI.INT],root=0)
N_part = np.array(0,dtype=np.int32)
#рассылка для каждого процессора количество строк матрицы
comm.Scatterv(
  [rCounts, np.ones(numProcs), np.array(range(numProcs)), MPI.INT],
  [N_part,1,MPI.INT],
  root=0
)
# каждому процессору необходимо сообщить за какими строками он сведущь

# создаем матрицу исходную 
if rank == 0: 
  startTime = MPI.Wtime()
  print(f"U (BEFORE):\n{A}")
  print(f"L (BEFORE):\n{L}")
else:
  A_part = np.empty((N_part,N)) # заготовки для A
  L_part = np.empty((N_part,N)) # заготовки для L
if rank == 0:
  comm.Scatterv(
    [A, rCounts*N, displs*N, MPI.DOUBLE],
    [A_part , N_part*N, MPI.DOUBLE], 
    root=0
  )
  comm.Scatterv(
    [L, rCounts*N, displs*N, MPI.DOUBLE],
    [L_part , N_part*N, MPI.DOUBLE], 
    root=0
  )

else:
  comm.Scatterv(
    [None, None, None, None],
    [A_part , N_part*N, MPI.DOUBLE], 
    root=0
  )
  comm.Scatterv(
    [None, None, None, None],
    [L_part , N_part*N, MPI.DOUBLE], 
    root=0
  )
if rank == 0:
  for iterRank in range(1,numProcs):
    rangeRows = range(displs[iterRank], displs[iterRank] + rCounts[iterRank])
    rankRowList = np.array(list(rangeRows),dtype=np.int32) # информация о строках данного процессора
    comm.Send(
      buf=[rankRowList, rankRowList.shape[0], MPI.INT],
      dest=iterRank,
      tag=0
    )
else:
  rankRowList = np.empty((N_part,),dtype=np.int32) # заготовки для  информация о строках доверенных процессору
  comm.Recv(
      buf=[rankRowList,rankRowList.shape[0],MPI.INT],
      source=0,
      tag=0
  ) 
  

comm.Barrier()
# теперь когда на каждом процессоре выделена память и сделаны заготовки участков матрицы 
# отправляем каждому вычитаемую строку от нулевого сервера, на остальных принимаем её и проводим вычисления
status = MPI.Status()
for i in range(N):
  if rank == 0: # корень
    row_ref = A[i,:].copy().reshape((1,N))
    comm.Bcast(buf=[row_ref,1*N,MPI.DOUBLE],root=0)

    diag_ref = np.array(i,dtype=np.int32).reshape((1,))
    comm.Bcast(buf=[diag_ref,1,MPI.INT],root=0)

    ############################################################
    comm.Barrier()
    ############################################################

    comm.Gatherv(
      [A_part, N_part*N, MPI.DOUBLE],
      [A, rCounts*N, displs*N, MPI.DOUBLE], 
      root=0
    )  
    comm.Gatherv(
      [L_part , N_part*N, MPI.DOUBLE],
      [L, rCounts*N, displs*N, MPI.DOUBLE], 
      root=0
    )
    
  else:# принимающий
    row_ref = np.empty((1,N),dtype=np.float64)
    comm.Bcast( buf=[row_ref,1*N,MPI.DOUBLE], root=0)

    diag_ref = np.empty(1,dtype=np.int32)
    comm.Bcast(buf=[diag_ref,1,MPI.INT],root=0)

    L_part, A_part = forwardGauss(int(diag_ref[0]),row_ref, A_part, L_part, N,rankRowList,RANK=rank)
    ############################################################
    comm.Barrier()
    ############################################################
    comm.Gatherv(
      [A_part , N_part*N, MPI.DOUBLE],
      [None,None,None,None],
      root=0
    )  
    comm.Gatherv(
      [L_part , N_part*N, MPI.DOUBLE],
      [None,None,None,None],
      root=0
    )

if rank == 0:
  np.set_printoptions(suppress=True)
  print(f"U (AFTTER):\n{A}")
  print(f"L (AFTTER):\n{L}")
  print(f"{numProcs - 1} processes (Matrix.shape = {A.shape}): {MPI.Wtime() - startTime} sec")
  


