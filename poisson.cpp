/*
    Implementação do algoritmo para resolução da Equação de Poisson com a Transformada Seno Discreta.

    Compilar com: g++ poisson.cpp -o poisson -lfftw3 -lm
*/

#include <iostream>
#include <fstream>
#include <cassert>
#include <string>
#include <cmath>
#include <complex.h>
#include <fftw3.h>
#include <algorithm>

using namespace std;

double* fast_poisson(int n, double dom, double *F, double* contorno);
void tsd(int N, double *in, double *out, bool normalize);
void F_contorno(int N, double *F, double *contorno);

double* U_teste01(int n);
double* F_teste01(int N);
double* c_teste01(int n);
void write(int n, double dom, double *U);


int main(int argc, char* argv[])
{
    //Teste 01: u(x,y) = x(1-x)+y(1-y), x,y \in [0,1]
    int n = 1000;
    int N = n-2;
    double dom = 1;
    double *F, *U, *U_real, *c;
    F = F_teste01(N);
    U = new double[N*N];
    c = c_teste01(n);
    U = fast_poisson(n, dom, F, c);
    U_real = new double[N*N];
    U_real = U_teste01(n);

    write(n, dom, U);
    
    for(int i=0; i<N*N; i++)
    {
        cout << abs(U[i]-U_real[i]) << "\n";
    }

    delete[] F;
    delete[] U;
    delete[] c;

    return 0;
}

double* fast_poisson(int n, double dom, double *F, double* contorno)
{
    //Passando os valores do contorno para o lado direito
    int N = n-2;
    F_contorno(N, F, contorno);

    //Multiplicação de h^2*F
    double h2 = pow(dom/(n-1), 2.0);
    for(int i=0; i<N*N; i++)
    {
        F[i] *= h2;
    }

    //Transformada seno discreta bidimensional de F = U_aux
    double* U_aux;
    U_aux = new double[N*N];
    tsd(N, F, U_aux, false);
    
    //vetor com os autovalores de T_N
    double *d;
    d = new double[N];
    for(int i=0; i<N; i++)
    {
        d[i] = 4.0*(pow(sin((i+1)*M_PI/(2*N + 2)), 2.0));
    }

    //U^ = F^/(autovalores) = F^/(d[i] + d[j])
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<N; j++)
        {
            U_aux[N*i + j] = U_aux[N*i + j]/(d[i] + d[j]);
        }
    }  

    //Transformada seno bidimensional inversa de U_aux = V
    double *U;
    U = new double[N*N];
    tsd(N, U_aux, U, true);

    delete[] U_aux;
    delete[] d;

    return U;
}

void tsd(int N, double *in, double *out, bool normalize=false)
/*
    Essa função utiliza a biblioteca fftw3 para executar a transformada seno discreta 2d.
    Para executar sua inversa, utilize normalize = true.
*/
{
    fftw_plan p;
    unsigned flags;
    p = fftw_plan_r2r_2d(N, N, in, out, FFTW_RODFT00, FFTW_RODFT00, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);

    if(normalize == true)
    {
        double k = pow(2*(N+1), 2.0);
        int M = N*N;
        for(int i=0; i<M; i++)
        {
            out[i] /= k;
        }
    }

    return;
}

void F_contorno(int N, double *F, double *contorno)
{
    for(int i=0; i<N; i++)
    {
        for(int j=0; j<N; j++)
        {
            if(i==0)
            {
                F[j] += contorno[j+1];
            }
            else
            {
                if(i==N-1)
                {
                    F[i*N + j] += contorno[(N+2) + 2*(i+1)+ j+1];
                }
            }
            if(j==0)
            {
                F[N*i] += contorno[(N+2) + 2*i];
            }   
            else
            {
                if(j==N-1)
                {
                    F[i*N + j] += contorno[(N+1) + 2*i + j];
                }
            }         
        }
    }
}

double* c_teste01(int n)
{
    int M = 2*n + 2*(n-2);
    double *c;
    c = new double[M];

    double h = 1/(n-1);
    double x, y;
    //Linha 0
    for(int j=0; j<n; j++)
    {
        y = j*h;
        c[j] = y*(1.0-y);
    }
    //Linhas intermediárias
    for(int k=0; k<(n-2); k++)
    {
        x = (k+1)*h;
        y = (n-1)*h;
        c[n + k] = x*(1.0-x); //Coluna 0
        c[n + k+1] = x*(1.0-x) + y*(1.0-y); //Coluna n-1
    }
    //Linha n-1
    for(int j=0; j<n; j++)
    {
        x = (n-1)*h;
        y = j*h;
        c[n + 2*(n-2) + j] = x*(1.0-x) + y*(1.0-y);
    }
    
    return c;
}

//funções auxiliares --------------------------------------------------------------------
double* U_teste01(int n)
{
    double* F;
    int N = n-2;
    F = new double[N*N];
    double h = 1/(n-1);
    double x, y;
    for(int i=0; i<N; i++)
    {
        x = i*h;
        for(int j=0; j<N; j++)
        {             
            y = j*h;
            F[n*i + j] = x*(1.0-x)+y*(1.0-y);
        }
    }
    return F;
}

double* F_teste01(int N)
{
    double* F;
    F = new double[N*N];

    for(int i=0; i<N*N; i++)
    {
        F[i] = 4.0;
    }
    return F;
}





void write(int n, double dom, double *U)
{
    double h2 = pow(dom/(n-1), 2.0);
    int N = n-2;

    //Escrevendo os dados em arquivos .dat
    std::ofstream write_output1("xy.dat", std::ios::app);
    std::ofstream write_output2("U.dat", std::ios::app);
    assert(write_output1.is_open());
    assert(write_output2.is_open());

    for(int i=0; i<N; i++)
    {
        write_output1 << (i+1)*h2 << "\n";
    }
    for(int i=0; i<N*N; ++i)
    {
        write_output2 << U[i] << "\n";
    }

    write_output1.close();
    write_output2.close();
    return;
}