#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 21:01:43 2022
Implementação do método de interpolação de uma função com polinômios na base 
de Lagrange.
@author: rayssa
"""
import numpy as np
import sympy as sym #Symbolic math
import math

def main():
    """
    Testes da função polinomio_lagrange().
    
    Obs.: para calcular o valor do polinômio num ponto específico use a função
    p = sym.lambdify(x, p), em que p é o polinômio de Lagrange.
    """
    x = sym.symbols("x")            #variável x                    

    print("Testes da função polinomio_lagrange")
    
    print("\nExemplo 01")
    xo = np.array([2, 5])       #vetor de valores observados (tabela)
    y = [4, 1]
    print(f"P(x): {polinomio_lagrange(xo, y)}")
    
    print("\nExemplo 02: ")
    xo = np.array([-1, 0, 2])       
    y = [4, 1, -1]
    print(f"P(x): {sym.expand(polinomio_lagrange(xo, y))}")
    
    print("\nExemplo 03")
    xo = np.array([2, 2.75, 4])
    y = [1/2, 1/2.75, 1/4]
    p = sym.simplify(polinomio_lagrange(xo, y))
    print(f"P(x): {p}")
    p = sym.lambdify(x, p)
    print(f"Aproximação de f(3) = 1/3: {p(3)}")
    
    
    print("\nExercise set 3.1")
    
    print(f"\n1.a)")
    xo = np.array([0, 0.6, 0.9])
    y = [math.cos(0), math.cos(0.6), math.cos(0.9)]
    p = sym.simplify(polinomio_lagrange(xo, y))
    print(f"P(x): {p}")
    p = sym.lambdify(x, p)
    pol = p(0.45)
    fun = math.cos(0.45)
    print(f"Aproximação de f(0.45) = {fun}: {pol}")
    print(f"|f(x) - p(x)| = {abs(fun-pol)}")
    
    print(f"\n1.b)")
    xo = np.array([0, 0.6, 0.9])
    y = [math.sqrt(1+0), math.sqrt(1+0.6), math.sqrt(1+0.9)]
    p = sym.simplify(polinomio_lagrange(xo, y))
    print(f"P(x): {p}")
    p = sym.lambdify(x, p)
    pol = p(0.45)
    fun = math.sqrt(1+0.45)
    print(f"Aproximação de f(0.45) = {fun}: {pol}")
    print(f"|f(x) - p(x)| = {abs(fun-pol)}")

    
    print(f"\n1.c)")
    xo = np.array([0, 0.6, 0.9])
    y = [math.log(0+1), math.log(0.6+1), math.log(0.9+1)]
    p = sym.simplify(polinomio_lagrange(xo, y))
    print(f"P(x): {p}")
    p = sym.lambdify(x, p)
    pol = p(0.45)
    fun = math.log(1+0.45)
    print(f"Aproximação de f(0.45) = {fun}: {pol}")
    print(f"|f(x) - p(x)| = {abs(fun-pol)}")

    
    print(f"\n1.d)")
    xo = np.array([0, 0.6, 0.9])
    y = [math.tan(0), math.tan(0.6), math.tan(0.9)]
    p = sym.simplify(polinomio_lagrange(xo, y))
    print(f"P(x): {p}")
    p = sym.lambdify(x, p)
    pol = p(0.45)
    fun = math.tan(0.45)
    print(f"Aproximação de f(0.45) = {fun}: {pol}")
    print(f"|f(x) - p(x)| = {abs(fun-pol)}")
 
    
def polinomio_lagrange(xo, y):
    """
    (array, array) ->  sympy object
    Recebe um vetor xo de dados observados e um vetor y correspondente a xo.
    Retorna o polinômio de Lagrange que interpola a função que associa xo a y
    nesses pontos.
    """
    x = sym.symbols('x')            #variável x                    
    n = len(xo) - 1                 #grau máximo do polinômio                 
    
    #Determinando a base de Lagrange
    L = []
    for k in range(n+1):
        L += [base_lagrange(x, xo, n, k)]
    
    #print(f"\nBase de Lagrange: \n{L}")
    
    #Polinômio na base de Lagrange
    P = 0
    for k in range(n+1):
        P += y[k]*L[k]
    
    return P


def base_lagrange(x, xo, n, k):
    """
    (sympy object, array, int, int) -> sympy object
    Recebe uma variável algébrica x, um vetor xo de dados observados, o grau
    máximo do polinômio de Lagrange n e um índice k. Retorna o k-ésimo vetor
    da base de Lagrange.
    """
    L = 1
    for i in range(n+1):
        if xo[i] != xo[k]:
            L *= (x - xo[i])/(xo[k] - xo[i])
            
    return L    
    

if __name__ == "__main__":
    main()