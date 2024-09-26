import sympy as smp
from sympy import Q

x, n = smp.symbols('x n')
print(smp.integrate( (smp.pi/2)/(1+smp.exp(x)) ,(x, -1/smp.sqrt(3), 1/smp.sqrt(3))).simplify())
print(smp.trigsimp(smp.acos(2*x/(1+x**2))).simplify())