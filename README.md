# gpusolver
CUDA linear solver with python API

This is a general purpose linear solver based on CUDA libraries with python API. 

Requirements:
```
CUDA8.0 (cublas, cudart, cusparse, cusolver)
python2.7( numpy, scipy, cython)
```

To begin navigate to a directory:
```
git clone https://github.com/yunfanz/gpusolver.git ./
cd ./gpusolver
python setup.py build_ext -i
```

### Features

Current features include:
1. single precision float solver for 2 dimensional matrix and 1 dimensional right hand side
2. Passing data in either dense or csr format
3. AtAx=Atb solver for general/non-square matrix A; Ax=b solver for square matrices
4. Four options of kernel backends (0:QR decomposition, 1:Cholesky decomposition, 2:LU, 3:SVD)

Features to be added:
1. Complex solver

### Sparse branch

Sparse branch contains a batched sparse solver. No multiplication is currently supported. Example usage see benchSp.py.
Input is reordered with the Reverse-Cuthill-Mckee algorithm and fed in batch into GPU, with optimal batchsize automatically determined from available GPU memory. 


### General structure of usage:
1. Initialize solver by calling ```solver = gpusolver.DnSolver(num_rows, num_cols)```
2. Solve using either ```solver.solve(n) or solver.solve_Axb(n)```
3. Retrieve the result ```x = solver.retrieve()```

Sample usage is given in unit_test.py

### Performance suggestions:

GPU code performance is a balance of data transfer and kernel execution. This implementation does not overlap the two,
so here's some guidelines in choosing what to use:

| size\sparsity        | Sparsity>0.5  | Sparsity<0.5|
| -------------       |:-------------:| -----:|
| small A     | Compare cpu vs. from_csr |Compare cpu vs. from_dense |
| large nrows>>ncols     | from_csr, solve<------|------> AtA on cpu, solve_Axb  |
| large A | from_csr |  from_csr vs. from_dense |
