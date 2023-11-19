#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 23:58:50 2022
Implementação do método de interpolação de uma função com polinômio de Hermite.
@author: rayssa
"""

import numpy as np

def main():
    """
    Testes da função hermite().
    """
    print("\n---------- Exemplo 01")
    xo = np.array([1, 3])
    y = np.array([2, 5])
    f_ = np.array([1, -1])
    q = Q(xo, y, f_)
    print(f"Q = {q}")
    
    print("\n---------- Exemplo 02")
    xo = np.array([1.3, 1.6, 1.9])
    y = np.array([0.6200860, 0.4554022, 0.2818186])
    f_ = np.array([-0.5220232, -0.5698959, -0.5811571])
    q = Q(xo, y, f_)
    print(f"Q = {q}")


def Q(xo, y, f_):
    """
    (array, array) -> array
    Recebe um vetor xo de dados observados e um vetor y correspondente a xo.
    Retorna um array com todas as i-ésimas diferenças divididas.
    """
    n = len(y)
    Q = np.zeros([n, n])
    
    #0-ésima diferença dividida
    Q[:,0] = y
        
    #Primeira diferença dividida
    Q[:,1] = f_
            
    return Q


if __name__ == "__main__":
    main()
                
        
    
        
    
    

