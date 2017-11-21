from __future__ import print_function
import numpy as np
from scipy.signal import fftconvolve
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import time
import pycuda.autoinit
from pycuda.tools import make_default_context
from skcuda import linalg
linalg.init()

def run(inputs, blocks=(16,16,1), grids=None):
	kernel_source = open("./corrMatrix.cu").read()
	kernel_code = kernel_source
	main_module = nvcc.SourceModule(kernel_code)
	get_correlation = main_module.get_function("get_correlation")
	outputs = []
	for corr_vector in inputs:
		w = np.int32(corr_vector.size)
		corrM_d = gpuarray.zeros((w,w), dtype=np.complex64)
		corrV_d = gpuarray.to_gpu(corr_vector)
		if grids is None:
			g = int(w/blocks[0])+1
			grids = (g,g,1)
		print(blocks, grids)
		get_correlation(corrM_d, corrV_d, w, block=blocks, grid=grids)
		import IPython; IPython.embed()
		S =  linalg.eig(corrM_d, 'N', 'N').get()
		outputs.append(S)
	return outputs

if __name__ == "__main__":
	sig = np.load('/data1/KLT/train_snrm5/raw/0.npy')
	print(sig.dtype, sig.size)
	start = time.time()
	autocorr = fftconvolve(sig, np.conj(sig[::-1]))
	autocorr = autocorr[len(autocorr)//2:len(autocorr)//2+64]
	t1 = time.time()
	inputs = [autocorr for i in range(100)]
	S = run(inputs)
	t2 = time.time()
	import IPython; IPython.embed()