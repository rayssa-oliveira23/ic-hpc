"""
EP01 - Fundamentos de Análise Numérica (MAP2220)
Rayssa Oliveira Santos
N. USP: 12558450
"""

import numpy as np
import matplotlib.pyplot as plt

def Fatoracao_Crout(A):
    """
    (array) -> array
    Essa função recebe uma matriz aumentada (b é a última coluna) A tridiagonal,
    resolve o sistema linear dado por A e retorna essa solução. 
    Foi utilizado como base o Algoritmo 6.7 de Burden & Faires.
    """
    n = A.shape[0]  #Dimensão da matriz A

    #---- Primeira parte: construir e resolver Lz = b
    #Matriz L, triangular inferior
    L = np.zeros((n,n))
    L[0][0] = A[0][0]

    #Matriz U, triangular inferior
    U = np.zeros((n,n))
    U[0][1] = A[0][1]/L[0][0]

    #Vetor z, que será solução de Lz = b
    z = np.zeros(n)
    z[0] = A[0][n]/L[0][0]

    for i in range(1, n-1):
        L[i][i-1] = A[i][i-1]                           #i-ésima linha de L
        L[i][i] = A[i][i] - L[i][i-1]*U[i-1][i]
        U[i][i+1] = A[i][i+1]/L[i][i]                   #(i+1)-ésima coluna de U
        z[i] = (A[i][n] - L[i][i-1]*z[i-1])/L[i][i]
    
    L[n-1][n-2] = A[n-1][n-2]
    L[n-1][n-1] = A[n-1][n-1] - L[n-1][n-2]*U[n-2][n-1]
    z[n-1] = (A[n-1][n]-L[n-1][n-2]*z[n-2])/L[n-1][n-1]

    """
    print(f"L: \n{L}")
    print(f"\nU: \n{U}")
    print(f"\nz = {z}")
    """

    #---- Segunda parte: resolver Ux = z
    #Vetor x, que será solução de Ux = z
    x = np.zeros(n)
    x[n-1] = z[n-1]
    for i in range(n-2, -1, -1):
        x[i] = z[i] - U[i][i+1]*x[i+1]
    
    return x


def Spline_Cubico_Natural(x, y):
    """
    (array, array) -> array, array, array, array

    Essa função recebe um array x de pontos a serem interpolados, um array y que
    corresponde ao valor de f(x) e retorna os arrays a, b, c e d cujas entradas
    são coeficientes para o spline natural que interpola f nos pontos dados.
    """
    n = len(x)-1        #n é o índice máximo de x e y
    a = y

    #----Passo 1
    h = np.zeros(n)
    alpha = np.zeros(n+1)
    #print(f"\nx = {x}")
    for i in range(n):
        h[i] = x[i+1]-x[i]
    #print(f"\n\nh: \n{h}")


    #----Passo 2
    for i in range(1, n):
        alpha[i] = (3/h[i])*(a[i+1]-a[i]) - (3/h[i-1])*(a[i]-a[i-1])
    #print(f"\n\nalpha: \n{alpha}")

    #----Passos 3, 4 e 5 adaptados para utilizar a função Fatoracao_Crout()
    A = np.zeros((n+1, n+2)) #A matriz A (aumentada) tem n+2 colunas porque b (= alpha) será a última delas
    
    #Primeira e última entrada A (iguais a 1)
    A[0][0], A[n][n] = 1, 1

    #Adicionando as demais entradas de A
    j = 0
    for i in range(1, n):
        A[i][j] = h[i-1]
        A[i][j+1] = 2*(h[i-1]+h[i])
        A[i][j+2] = h[i]
        j += 1

    A[:, n+1] = alpha      #Agora, A é a matriz aumentada do sistema linear Ax = b (alpha = b)
    #print(f"\n\nA: \n{A}")

    #Pode-se agora utilizar a função Fatoracao_Crout()
    c = Fatoracao_Crout(A)
    #print(f"\nc = {c}")

    #----Passo 6
    b = np.zeros(n)
    d = np.zeros(n)
    for j in range(n-1, -1, -1):
        b[j] = (a[j+1]-a[j])/h[j] - (h[j]/3)*(c[j+1] + 2*c[j])
        d[j] = (c[j+1]-c[j])/(3*h[j])
    #print(f"\nb = {b}")
    return a, b, c, d


def Spline_Cubico_Fixo(x, y, fpo, fpn):
    """
    (array, array, float, float) -> array, array, array, array

    Essa função recebe um array x de pontos a serem interpolados, um array y que
    corresponde ao valor de f(x), o valor da derivada de f nos pontos extremos fpo 
    e fpn e retorna os arrays a, b, c e d cujas entradas são coeficientes para o 
    spline fixo (clamped spline) que interpola f nos pontos dados.
    """
    n = len(x) - 1          #n é o índice máximo de x e y
    #----Passo 1
    h = np.zeros(n)
    for i in range(n):
        h[i] = x[i+1] - x[i]
    
    #----Passo 2
    alpha = np.zeros(n+1)
    a = y
    alpha[0] = 3*(a[1]-a[0])/h[0] - 3*fpo
    alpha[n] = 3*fpn - 3*(a[n]-a[n-1])/h[n-1]

    #----Passo 3
    for i in range(1, n):
        alpha[i] = (3/h[i])*(a[i+1]-a[i])-(3/h[i-1])*(a[i]-a[i-1])
    #print(f"\n\nalpha: \n{alpha}")

    #----Passos 4, 5 e 6 adaptados para utilizar a função Fatoracao_Crout()
    A = np.zeros((n+1, n+2)) #A matriz A (aumentada) tem n+2 colunas porque b (= alpha) será a última delas

    #Primeira e última linha de A (iguais a 1)
    A[0][0], A[0][1] = 2*h[0], h[0]
    A[n][n-1], A[n][n] = h[n-1], 2*h[n-1]

    #Adicionando as demais entradas de A
    j = 0
    for i in range(1, n):
        A[i][j] = h[i-1]
        A[i][j+1] = 2*(h[i-1]+h[i])
        A[i][j+2] = h[i]
        j += 1

    A[:, n+1] = alpha      #Agora, A é a matriz aumentada do sistema linear Ax = b (alpha = b)
    #print(f"\n\nA: \n{A}")

    #Pode-se agora utilizar a função Fatoracao_Crout()
    c = Fatoracao_Crout(A)
    #print(f"\nc = {c}")
    
    #----Passo 6
    b = np.zeros(n)
    d = np.zeros(n)
    for j in range(n-1, -1, -1):
        b[j] = (a[j+1]-a[j])/h[j] - h[j]*(c[j+1] + 2*c[j])/3
        d[j] = (c[j+1]-c[j])/(3*h[j])

    return a, b, c, d


def Interpolacao_Polinomial(x, y):
    """
    (array, array) -> array
    Essa função recebe um array x de pontos a serem interpolados, um array y que
    corresponde ao valor de f(x) e retorna o array dif cuja primeira linha contém os
    coeficientes para o polinômio que interpola f nos pontos dados.
    """
    n = len(y)
    dif = np.zeros([n, n])
    #0-ésima diferença dividida
    dif[:,0] = y
    
    for j in range(1,n):
        for i in range(n-j):
            dif[i][j] = (dif[i+1][j-1] - dif[i][j-1]) / (x[i+j]-x[i])
            
    return dif


def Calcula_Spline(no, a, b, c, d, h):
    """
    (array, array, array, array, array, float) -> array, array
    Essa função recebe o vetor de nós no, os vetores de coeficientes a, b, c, d do spline
    e o passo h. Retorna os vetores x e y = S(x), para serem plotados.
    """
    y = []
    x = []

    #Gerando pontos com espaço h para calcular no spline
    i = no[0]       #Primeiro elemento do vetor de nós
    while i <= no[len(no)-1]:
        x += [i]
        i += h
    x = np.array(x)

    n = len(x)
    j = 0
    for i in range(n):
        flag = True
        while flag:
            if no[j] <= x[i] < no[j+1]:    #Se x está entre os nós j e j+1:
                y += [a[j] + b[j]*(x[i] - no[j]) + c[j]*(x[i] - no[j])**2 + d[j]*(x[i]-no[j])**3]
                flag = False
            else:
                j += 1
    
    return x, y


def Calcula_Polinomio(x, no, c, n):
    """
    (float, array, array, int) -> float
    Essa função recebe um valor x o que se deseja calcular P(x), em que P é o
    polinômio interpolador de f, um array de nós no, o vetor c de coeficientes
    do polinômio e o grau n do polinômio. Retorna o valor de P(x).
    """
    P = c[0]

    for k in range(1, n):
        prod = 1
        for j in range(k):
            prod *= (x - no[j])
        P += (c[k]*prod)

    return P


def main():
    """
    Essa função tem como objetivo mostrar a aplicação das demais funções nos exemplos solicitados.
    """
    
    print("\n\nAPLICAÇÃO 1 ___________________________________________________________________________________")
    #----(1) EXEMPLO: RUDDY DUCK IN FLIGHT--------------------------------------------------------
    print("\n(1) EXEMPLO: RUDDY DUCK IN FLIGHT")
    #Tabela fornecida:
    x0 = [0.9, 1.3, 1.9, 2.1, 2.6, 3.0, 3.9, 4.4, 4.7, 5.0, 6.0, 7.0, 8.0, 9.2, 10.5, 11.3, 11.6, 12.0, 12.6, 13.0, 13.3]
    x0 = np.array(x0)
    #y = f(x)
    y0 = [1.3, 1.5, 1.85, 2.1, 2.6, 2.7, 2.4, 2.15, 2.05, 2.1, 2.25, 2.3, 2.25, 1.95, 1.4, 0.9, 0.7, 0.6, 0.5, 0.4, 0.25]
    y0 = np.array(y0)

    a, b, c, d = Spline_Cubico_Natural(x0, y0)

    #Exibindo a tabela com os coeficientes
    print("---- Tabela 1: ")
    print ("-----------------------------------------------------------------------------------------------")
    print ("%-15s" % "j", "%-15s" % "xj", "%-15s" % "aj", "%-15s" % "bj", "%-15s" % "cj", "%-15s" % "dj")
    print ("-----------------------------------------------------------------------------------------------")

    n = len(x0) - 1
    
    for j in range(n):
        print("%-15s" % j, "%-15s" % x0[j], "%-15s" % a[j], "%-15s" % round(b[j], 2), "%-15s" % round(c[j], 2), "%-15s" % round(d[j], 2),)
    print("%-15s" % n, "%-15s" % x0[n], "%-15s" % a[n])
    
    #Utilizando a função para gerar os pontos para o gráfico
    x, y = Calcula_Spline(x0, a, b, c, d, 0.1)
    #Gráfico
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y = S(x)")
    plt.title("Gráfico 1: Exemplo Ruddy Duck in Flight")
    plt.show()

    print("\n_______________________________________________________________________________________________")


    #----(1) EXEMPLO: SNOOPY NOBLE BEAST ---------------------------------------------------
    print("\n\n(1) EXEMPLO: SNOOPY NOBLE BEAST")

    #--------Spline 1
    #Tabela fornecida:
    x1 = [1, 2, 5, 6, 7, 8, 10, 13, 17]
    x1 = np.array(x1)
    #y = f(x)
    y1 = [3.0, 3.7, 3.9, 4.2, 5.7, 6.6, 7.1, 6.7, 4.5]
    y1 = np.array(y1)
    fpo1 = 1.0
    fpn1 = 0.67

    a1, b1, c1, d1 = Spline_Cubico_Fixo(x1, y1, fpo1, fpn1)

    #Exibindo a tabela com os coeficientes
    print("---- Tabela 1.1: Coeficientes do Spline 1")
    print ("-----------------------------------------------------------------------------------------------")
    print ("%-15s" % "j", "%-15s" % "xj", "%-15s" % "aj", "%-15s" % "bj", "%-15s" % "cj", "%-15s" % "dj")
    print ("-----------------------------------------------------------------------------------------------")

    n = len(x1) - 1
    
    for j in range(n):
        print("%-15s" % j, "%-15s" % x1[j], "%-15s" % a1[j], "%-15s" % round(b1[j], 5), "%-15s" % round(c1[j], 5), "%-15s" % round(d1[j], 5),)
    print("%-15s" % n, "%-15s" % x1[n], "%-15s" % a1[n])

    print("\n")

    #--------Spline 2
    #Tabela fornecida:
    x2 = [17, 20, 23, 24, 25, 27, 27.7]
    x2 = np.array(x2)
    #y = f(x)
    y2 = [4.5, 7.0, 6.1, 5.6, 5.8, 5.2, 4.1]
    y2 = np.array(y2)
    fpo2 = 3.0
    fpn2 = -4.0

    a2, b2, c2, d2 = Spline_Cubico_Fixo(x2, y2, fpo2, fpn2)

    #Exibindo a tabela com os coeficientes
    print("---- Tabela 1.2: Coeficientes do Spline 2")
    print ("-----------------------------------------------------------------------------------------------")
    print ("%-15s" % "j", "%-15s" % "xj", "%-15s" % "aj", "%-15s" % "bj", "%-15s" % "cj", "%-15s" % "dj")
    print ("-----------------------------------------------------------------------------------------------")

    n = len(x2) - 1
    
    for j in range(n):
        print("%-15s" % j, "%-15s" % x2[j], "%-15s" % a2[j], "%-15s" % round(b2[j], 5), "%-15s" % round(c2[j], 5), "%-15s" % round(d2[j], 5),)
    print("%-15s" % n, "%-15s" % x2[n], "%-15s" % a2[n])

    print("\n")

    #--------Spline 3
    #Tabela fornecida:
    x3 = [27.7, 28, 29, 30]
    x3 = np.array(x3)
    #y = f(x)
    y3 = [4.1, 4.3, 4.1, 3.0]
    y3 = np.array(y3)
    fpo3 = 0.33
    fpn3 = -1.5

    a3, b3, c3, d3 = Spline_Cubico_Fixo(x3, y3, fpo3, fpn3)

    #Exibindo a tabela com os coeficientes
    print("---- Tabela 1.3: Coeficientes do Spline 3")
    print ("-----------------------------------------------------------------------------------------------")
    print ("%-15s" % "j", "%-15s" % "xj", "%-15s" % "aj", "%-15s" % "bj", "%-15s" % "cj", "%-15s" % "dj")
    print ("-----------------------------------------------------------------------------------------------")

    n = len(x3) - 1
    
    for j in range(n):
        print("%-15s" % j, "%-15s" % x3[j], "%-15s" % a3[j], "%-15s" % round(b3[j], 5), "%-15s" % round(c3[j], 5), "%-15s" % round(d3[j], 5),)
    print("%-15s" % n, "%-15s" % x3[n], "%-15s" % a3[n])

    #Utilizando a função para gerar os pontos para o gráfico
    x1, y1 = Calcula_Spline(x1, a1, b1, c1, d1, 0.01)   #Spline 1
    x2, y2 = Calcula_Spline(x2, a2, b2, c2, d2, 0.01)   #Spline 2
    x3, y3 = Calcula_Spline(x3, a3, b3, c3, d3, 0.01)   #Spline 3
    #Juntando os pontos gerados por cada parte do spline
    x = np.concatenate((x1, x2, x3)) 
    y = np.concatenate((y1, y2, y3))
    #Gráfico
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y = S(x)")
    plt.title("Gráfico 2: Exemplo Snoopy Noble Beast")
    plt.show()

    print("\n_______________________________________________________________________________________________")


    #----(2) INTERPOLAÇÃO POLINOMIAL (NEWTON) ---------------------------------------------------
    print("\n\n(2) INTERPOLAÇÃO POLINOMIAL (NEWTON)")


    print("---- Tabela 2.1: Coeficientes do polinômio no exemplo Ruddy Duck in Flight")
    c1 = Interpolacao_Polinomial(x0, y0)[0]
    n = len(c1)
    print("---------------------")
    print("%-5s" % "i", "%-s" % "c[i]")
    print("---------------------")
    for i in range(n):
        print("%-5i" % i, "%-5f" % c1[i])
    
    #Gerando pontos com espaço 0.1 para calcular no polinômio
    x = []
    i = 0.9
    while i <= 13.3:
        x += [i]
        i += 0.1
    #Calculando y = P(x)
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        y[i] = Calcula_Polinomio(x[i], x0, c1, len(c1))

    #Gráfico
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y = P(x)")
    plt.title("Gráfico 3: Exemplo Ruddy Duck in Flight (com interpolação polinomial)")
    plt.show()


    print("\n\n---- Tabela 2.2: Coeficientes do polinômio no exemplo Snoopy Noble Beast")
    xs = [1, 2, 5, 6, 7, 8, 10, 13, 17, 20, 23, 24, 25, 27, 27.7, 28, 29, 30]
    ys = [3.0, 3.7, 3.9, 4.2, 5.7, 6.6, 7.1, 6.7, 4.5, 7.0, 6.1, 5.6, 5.8, 5.2, 4.1, 4.3, 4.1, 3.0]

    c2 = Interpolacao_Polinomial(xs, ys)[0]
    n = len(c2)
    print("---------------------")
    print("%-5s" % "i", "%-s" % "c[i]")
    print("---------------------")
    for i in range(n):
        print("%-5i" % i, "%-5f" % c2[i])
    
    #Gerando pontos com espaço 0.1 para calcular no polinômio
    x = []
    i = 1
    while i <= 30:
        x += [i]
        i += 0.1
    #Calculando y = P(x)
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        y[i] = Calcula_Polinomio(x[i], xs, c2, len(c2))

    #Gráfico
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y = P(x)")
    plt.title("Gráfico 4: Exemplo Snoopy Noble Beast (com interpolação polinomial)")
    plt.show()


    print("\n\nAPLICAÇÃO 2 ___________________________________________________________________________________")
    #Item 1
    x = [1, 2, 3, 4, 5, 6]
    y = [1.00, 1.25, 1.75, 2.25, 3.00, 3.15]
    fpo = 0.125
    fpn = 0.15
    a, b, c, d = Spline_Cubico_Fixo(x, y, fpo, fpn)

    #Exibindo a tabela com os coeficientes
    print("\n---- Tabela 3: Coeficientes da trajetória do robô")
    print ("-----------------------------------------------------------------------------------------------")
    print ("%-15s" % "j", "%-15s" % "xj", "%-15s" % "aj", "%-15s" % "bj", "%-15s" % "cj", "%-15s" % "dj")
    print ("-----------------------------------------------------------------------------------------------")

    n = len(x) - 1
    
    for j in range(n):
        print("%-15s" % j, "%-15s" % x[j], "%-15s" % a[j], "%-15s" % round(b[j], 5), "%-15s" % round(c[j], 5), "%-15s" % round(d[j], 5),)
    print("%-15s" % n, "%-15s" % x[n], "%-15s" % a[n])

    #Utilizando a função para gerar os pontos para o gráfico
    x, y = Calcula_Spline(x, a, b, c, d, 0.1)
    #Gráfico
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y = S(x)")
    plt.title("Gráfico 5: Trajetória do robô")
    plt.show()

    #Note que o elemento de índice 5 do array x é 1.5, então y[5] é o valor correspondente.
    print(f"\nUma aproximação para 1.5: f(1.5) = {y[5]}\n\n")



if __name__ == "__main__":
    main();