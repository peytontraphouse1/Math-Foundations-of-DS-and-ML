#Question 2

#Asked ChatGPT the following:
#To solve a linear inverse problem (i.e., 𝑨𝒙 = 𝒃) using the SVD, 𝑨 = 𝑼𝚺𝑽⊤, 
# we typically solve three linear systems for the problem 𝒙 = 𝑨−𝟏𝒃 = 𝑽𝚺−𝟏𝑼⊤𝒃: 
# Solve 𝒚 = 𝑼⊤𝒃 for 𝒚.
#  Solve 𝒛 = 𝚺−𝟏𝒚, where Σ−1 = diag(1/𝜎𝑖) for 𝒛. 
# Solve 𝒙 = 𝑽𝒛 for x. 
# Develop an algorithm in python without numpy using your SVD derivation that solves these three systems.
# Apply this to the previous problem to find 𝒙 with 𝐴 = [ 1 4 9 3 2 3 6 4 5 ] , 𝑏 = [18,10,19]⊤



def matvec(A, x):
    return [sum(A[i][j] * x[j] for j in range(len(x)))
            for i in range(len(A))]

def transpose(A):
    return [list(row) for row in zip(*A)]


def svd_solve(U, Sigma, V, b, tol=1e-12):
   

   
    UT = transpose(U)
    y = matvec(UT, b)

    
    z = []
    for i in range(len(Sigma)):
        sigma = Sigma[i][i]
        if abs(sigma) < tol:
            z.append(0.0)     
        else:
            z.append(y[i] / sigma)

   
    x = matvec(V, z)

    return x
