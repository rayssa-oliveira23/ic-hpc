#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EP02 - Fundamentos de Análise Numérica (MAP2220)
Rayssa Oliveira Santos
N. USP: 12558450

Implementação do Método de Simpson Composto, Método de Romberg e da Quadratura
Gaussiana.
"""

import numpy as np
import quadratura as quad

def Simpson_Composto(f, a, b, n):
    """
    (function, float, float, int) -> float
    
    Recebe uma função f, os extremos de um intervalo [a, b] e um inteiro
    positivo n. 
    Retorna uma aproximação da integral da função f no intervalo [a, b] 
    pelo Método de Simpson Composto. 
    
    Obs.: Se o n fornecido não for par ou a for maior que b, a função
    retorna None.
    """
    if n%2 != 0 or a >= b:
        return None     #Caso em que n não é par ou a > b.
    
    h = (b-a)/n         #Distância entre os pontos
    
    XI0 = f(a) + f(b)   #Soma do valor de f nos extremos do intervalo
    XI1 = 0             #Será a soma dos valores de f para x_i com i ímpar
    XI2 = 0             #Será a soma dos valores de f para x_i com i par
    
    for i in range(n):
        X = a + i*h
        if i%2 == 0:
            XI2 += f(X)
        else:
            XI1 += f(X)
    
    XI = h*(XI0 + 2*XI2 + 4*XI1)/3
    
    return XI


def Romberg(f, a, b, n):
    """
    (function, float, float, int) -> array
    
    Recebe uma função f, os extremos de um intervalo [a, b] e um inteiro 
    positivo n. 
    Retorna uma tabela com aproximações da integral da função f no intervalo
    [a, b] pelo Método de Romberg.
    
    Obs.: se a for maior que b, a função retorna None.
    """
    if a >= b:
        return None         #Caso em que a > b.
    
    h = b-a                 #Distância entre os pontos
    R = np.zeros((2,n))     #Vetor com duas linhas e n colunas
    
    #print("Linha 0")
    R[0][0] = (h/2)*(f(a) + f(b))
    #print(f"R[0][0] = {R[0][0]}")
    
    for i in range(1,n):
        #Aproximação com Método dos Trapézios Composto
        soma = 0
        for k in range(1,int((2**(i-1)))+1):
            soma += f(a + (k-0.5)*h)
            
        R[1][0] = (1/2)*(R[0][0] + h*soma)
        
        #Extrapolação de Richardson
        for j in range(1,i+1):
            R[1][j] = R[1][j-1] + (R[1][j-1]-R[0][j-1])/((4**j)-1)
        
        """
        print(f"\nLinha {i}:")
        for j in range(i+1):
            print(f"R[1][{j}] = {R[1][j]}")
        """
        
        h /= 2
        
        if i == n-1:
            return R
        
        #Atualizando a primeira linha de R
        for j in range(i+1):
            R[0][j] = R[1][j]
        
    return R


def Quadratura_Gaussiana(f, a, b, n):
    """
    (function, float, float, int) -> float
    
    Recebe uma função f, os extremos de um intervalo [a, b] e um inteiro 
    positivo n. 
    Retorna uma aproximação da integral da função f no intervalo [a, b] 
    pelo Método da Quadratura Gaussiana. 
    
    Obs.: se a for maior que b, a função retorna None.
    """
    if a >= b:
        return None         #Caso em que a > b.
    
    mult = 1
    soma = 0
    
    #Verificando se é necessária mudança de variável
    if a != -1 or b != 1:
        mult = (b-a)/2
        soma = (b+a)/2
    
    somatoria = 0
    r = quad.raizes(n)
    c = quad.coeficientes(n)
    for i in range(n):
        somatoria += c[i]*f(mult*r[i]+soma)*mult
    
    return somatoria
        
            
    
    

