#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 12:32:50 2022
Implementação do método de interpolação de uma função com diferenças divididas.
@author: rayssa
"""
import numpy as np
import sympy as sym #Symbolic math

def main():
    """
    Testes da função diferenças_divididas()
    """
    x = sym.symbols("x")
    
    print("\n---------- Exemplo 01")
    xo = np.array([1.0, 1.3, 1.6, 1.9, 2.2])
    y = np.array([0.7651977, 0.6200860, 0.4554022, 0.2818186, 0.1103623])
    
    dif = diferencas_divididas(xo, y)
    print(f"\nDiferenças divididas: \n{dif}")
    dif = diferencas_divididas(xo, y)[0, :]
    
    P = polinomio(dif, xo)
    print(f"\nP(x) = {P}")
    
    
    print("\n---------- Exercise set 3.3")
    print(f"\n-----1.a)")
    xo = np.array([8.1, 8.3, 8.6, 8.7])
    y = np.array([16.94410, 17.56492, 18.50515, 18.82091])
    
    dif = diferencas_divididas(xo, y)
    print(f"\nDiferenças divididas: \n{dif}")
    dif = diferencas_divididas(xo, y)[0, :]
    
    P = polinomio(dif, xo)
    print(f"\nP(x) = {P}")
    
    P = sym.lambdify(x, P)
    print(f"\nAproximação de f(8.4): {P(8.4)}")

    print(f"\n-----1.b)")
    xo = np.array([0.6, 0.7, 0.8, 1.0])
    y = np.array([-0.17694460, 0.01375227, 0.22363362, 0.65809197])
    
    dif = diferencas_divididas(xo, y)
    print(f"\nDiferenças divididas: \n{dif}")
    dif = diferencas_divididas(xo, y)[0, :]
    
    P = polinomio(dif, xo)
    print(f"\nP(x) = {P}")
    
    P = sym.lambdify(x, P)
    print(f"\nAproximação de f(0.9): {P(0.9)}")
    
    
def diferencas_divididas(xo, y):
    """
    (array, array) -> array
    Recebe um vetor xo de dados observados e um vetor y correspondente a xo.
    Retorna um array com todas as i-ésimas diferenças divididas.
    """
    n = len(y)
    dif = np.zeros([n, n])
    #0-ésima diferença dividida
    dif[:,0] = y
    
    for j in range(1,n):
        for i in range(n-j):
            dif[i][j] = \
           (dif[i+1][j-1] - dif[i][j-1]) / (xo[i+j]-xo[i])
            
    return dif    


def polinomio(dif, xo):
    """
    (array, array) -> sym object
    Recebe as diferenças divididas dif e o vetor xo de dados observados.
    Retorna o polinômio que interpola a função que associa xo a y nesses pontos,
    com os coeficientes sendo as diferenças divididas.
    """
    x = sym.symbols("x")
    n = len(xo)
    P = dif[0]

    for k in range(1, n):
        prod = 1
        for j in range(k):
            prod *= (x - xo[j])
        P += (dif[k]*prod)
    
    return P
    
    
    


if __name__ == "__main__":
    main()
