import numpy as np
import sympy as sp
from sympy.abc import phi, omega, delta

def dynamicModel():
    # Define symbols
    x, y, vx, vy, d = sp.symbols('x y vx vy d') # remaining state and control variables
    lf, lr, m, Jz = sp.symbols('lf lr m Jz') # vehicle constants
    Cm1, Cm2, Cm3, Cm4 = sp.symbols('Cm1 Cm2 Cm3 Cm4') # drivetrain constants
    Df, Dr, Cf, Cr, Bf, Br = sp.symbols('Df Dr Cf Cr Bf Br') # Pacejka constants

    # Drivetrain model
    Fx = (Cm1 - Cm2 * vx) * d - Cm3 - Cm4 * vx ** 2

    # Pacejka model
    alphaf = -sp.atan((omega * lf + vy) / vx) + delta
    alphar = sp.atan((omega * lr - vy) / vx)

    Ffy = Df * sp.sin(Cf * sp.atan(Bf * alphaf))
    Fry = Dr * sp.sin(Cr * sp.atan(Br * alphar))

    f = sp.Matrix([vx * sp.cos(phi) - vx * sp.sin(phi),
                   vx * sp.sin(phi) + vy * sp.cos(phi),
                   omega,
                   (Fx - Ffy * sp.sin(delta) + Fx * sp.cos(delta) + m * vy * omega) / m,
                   (Fry + Ffy * sp.cos(delta) + Fx * sp.sin(delta) - m * vx * omega) / m,
                   (lf * Ffy * sp.cos(delta) + lf * Fx * sp.sin(delta) - lr * Fry) / Jz])

    A = f.jacobian([x, y, phi, vx, vy, omega])
    B = f.jacobian([d, delta])
    
    print(sp.latex(A))
    print(B)

def kinematicModel():
    # Define symbols
    x, y, v = sp.symbols('x y v') # remaining state and control variables
    lf, lr = sp.symbols('lf lr') # vehicle constants

    # Sideslip angle
    beta = sp.atan(sp.tan(delta) * lr / (lf + lr))

    f = sp.Matrix([v * sp.cos(phi + beta),
                   v * sp.sin(phi + beta),
                   v * sp.cos(beta) * sp.tan(delta) / (lf + lr)])

    A = f.jacobian([x, y, phi])
    B = f.jacobian([v, delta])
    
    print(A)
    print(B)

    delta_val = 0

    print(A.subs({delta:delta_val}))

    print(A.subs({phi:0, v:5, delta:0}))
    
    A_np = np.array(A.subs({phi:0, v:5, delta:0})).astype(np.float64)
    print(A_np)
    
if __name__ == '__main__':
    #dynamicModel()
    kinematicModel()
