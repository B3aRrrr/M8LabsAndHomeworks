import numpy as np

for N in [2 ** i for i in range(7, 11)]:
    mat = np.random.uniform(-100, 100, size=(N, N))
    # np.fill_diagonal(mat, 0)

    filename = f"C:/Users/Dmitry/Putty/visualprojects/LUСonsistent/LUPython/ExampleLU_{N}.txt"
    path = np.savetxt(filename, mat,fmt='%s')#  delimiter=';', fmt='%s')