import numpy as np
itr_limit = 20
conv_crit = 0.0000001


def stop_cond(A, b, x):

	n = len(b)
	res = (A@x - b)
	magnitude = (res@res)**0.5
	
	if magnitude < conv_crit:
		return True
	else:
		return False 


def jacobi(A, b, N = -1, x_old = None):

	n = len(b)
	itr = 0
	converged = False
	x_new = np.empty(n)
	if x_old == None:
		x_old = np.zeros(n)

	while not converged and itr < itr_limit and itr != N:

		for i in range (n):
			dot = 0
			for j in range(n):
				if(i!=j):
					dot += A[i,j]*x_old[j]

			x_new[i] = (b[i] - dot)/A[i,i]

		itr += 1
		converged = stop_cond(A,b,x_new)
		x_old = x_new.copy()
		
	if not converged:
		print("Iteration limit reached without convergence.")	

	return x_new	
	
M = np.array([[4,1],[1,4]],dtype=float)
x = np.array([5,5],dtype=float)
print(jacobi(M,x))