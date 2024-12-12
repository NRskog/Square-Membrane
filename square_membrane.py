import numpy as np
#####Make class so I dont have to pass around same variables constantly



a = 10  #sqaure side
Nx, Ny = 100, 100
x = np.linspace(0, a, Nx)
y = np.linspace(0,a, Ny)
X, Y = np.meshgrid(x,y)


def f(x, y,  a): #using something similar to excersice 16, where function 0 at limits. more modes means multiply in sin with 2,3,4 etc
    return np.sin(np.pi * x / a) * np.sin(np.pi * y / a)



def G_nm(x, y, n, m, a ):
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    f_values= f(X, Y, a) # Values at each grid point, faster than prev when i calculated current value at each iteration
    
    integrand = f_values * np.sin(np.pi * X * n / a) * \
                np.sin(np.pi * Y * m / a)  
    integral = np.sum(integrand) * dx * dy


    return (4/(a**2)) * integral


def psi(x, y, n, m, c, a, t , N_terms = 20): # Depends on (x,y,t)
    res = np.zeros_like(X)
    
    for n in range(1, N_terms +1):
        for m in range(1, N_terms + 1):

            omega = (np.pi * c / a) * np.sqrt(n**2 + m**2)
            G = G_nm(x, y, n, m, a)
            res += G * np.sin(np.pi * n * X/a) * np.sin(np.pi * m * Y / a) *\
                np.cos(omega * t )

    return res



    