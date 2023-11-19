/*
    Implementação do Multigrid 2D.

    Ao executar o programa, é realizado um teste que passa por v0 V-Ciclos e pelo Full Multigrid, 
    exibindo o erro máximo em relação à solução exata e o resíduo máximo em cada uma dessas etapas.

    Compilar com: g++ multigrid.cpp -o multigrid -lm
*/

#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <omp.h>

std::vector<double> gaussSeidel(int n, std::vector<double> u, std::vector<double> f, int n1);
std::vector<double> residual(int n, std::vector<double> u, std::vector<double> f);
std::vector<double> restriction(int n, std::vector<double> u);
std::vector<double> interpolation(int n, std::vector<double> u);
std::vector<double> vCycle(int n, std::vector<double> u, std::vector<double> f, int n1, int n2);
std::vector<double> fullMultigrid(int n, double dom, std::vector<double> u, std::vector<double> f, double (*func)(double, double), int n0);

std::vector<double> contorno(std::vector<double> u, double dom, double (*func)(double, double));
std::vector<double> geraTeste(int n, double dom, double (*func)(double, double), bool ch2);
double maxError(int n, double dom, std::vector<double> u, double (*func)(double, double));
void print(std::vector<double> u);
void write(std::vector<double> u, double dom, std::string name);
double c01(double x, double y);
double c02(double x, double y);
double f02(double x, double y);

int main(int argc, char* argv[])
{
    std::cout << std::fixed;
    int n = 128;
    int n1 = 1, n2 = 1;
    double dom;
    double h2;
    std::vector<double> u((n+1)*(n+1));
    std::vector<double> v((n+1)*(n+1));
    std::vector<double> f((n+1)*(n+1));
    
    //Teste 01: u(x,y) = x(1-x)+y(1-y), (x,y) \in [0, 1]²
    dom = 1.0;
    h2 = pow(dom/n, 2.0);
    u = contorno(u, dom, c01);
    std::fill(&f[0], &f[(n+1)*(n+1)-1], 4.0*h2);  
    
    //V-ciclo
    std::vector<double> r;
    std::cout << "\n---- V-ciclo\n";
    int v0 = 6;     //Número de V-ciclos
    for(int i=0; i<v0; i++)
    {
        std::cout << i+1 << "º V-ciclo\n";
        u = vCycle(n, u, f, n1, n2);
        std::cout << "Erro = " << maxError(n, dom, u, c01) << "\n";
        r = residual(n, u, f);
        std::cout << "Resíduo = " << *std::max_element(r.begin(), r.end()) << "\n\n";
    }
    
    //Full Multigrid
    std::cout << "\n---- Full Multigrid\n";
    std::fill(&u[0], &u[(n+1)*(n+1)-1], 0.0);  //retornando ao vetor inicial nulo
    u = contorno(u, dom, c01);
    u = fullMultigrid(n, dom, u, f, c01, 1);
    std::cout << "Erro  = " << maxError(n, dom, u, c01) << "\n";
    r = residual(n, u, f);
    std::cout << "Resíduo = " << *std::max_element(r.begin(), r.end()) << "\n";
    
    //Para gerar um arquivo com o domínio e o resultado do teste:
    //write(u, dom, "teste01");
    
    
    //Teste 02: u(x,y) = sin(x)sin(y), (x,y) \in [0, 2pi]²
    dom = 2.0*M_PI;
    h2 = pow(dom/n, 2.0);
    u = contorno(u, dom, c02);
    f = geraTeste(n, dom, f02, true);

    //V-ciclo
    std::vector<double> r;
    std::cout << "\n---- V-ciclo\n";
    int v0 = 6;     //Número de V-ciclos
    for(int i=0; i<v0; i++)
    {
        std::cout << i+1 << "º V-ciclo\n";
        u = vCycle(n, u, f, n1, n2);
        std::cout << "Erro = " << maxError(n, dom, u, f02) << "\n";
        r = residual(n, u, f);
        std::cout << "Resíduo = " << *std::max_element(r.begin(), r.end()) << "\n\n";
    }

    //Full Multigrid
    std::cout << "\n---- Full Multigrid\n";
    std::fill(&u[0], &u[(n+1)*(n+1)-1], 0.0);  //retornando ao vetor inicial nulo
    u = contorno(u, dom, c02);
    u = fullMultigrid(n, dom, u, f, c02, 1);
    std::cout << "Erro  = " << maxError(n, dom, u, f02) << "\n";
    r = residual(n, u, f);
    std::cout << "Resíduo = " << *std::max_element(r.begin(), r.end()) << "\n";

    //Para gerar um arquivo com o domínio e o resultado do teste:
    write(u, dom, "teste02");
    
    return 0;
}

std::vector<double> gaussSeidel(int n, std::vector<double> u, std::vector<double> f, int n1)
//Gauss-Seidel Red-Black
{
    n += 1;
    //std::cout << "\n\nn = " << n << "\n";
    for(int k=0; k<n1; k++)
    {   
        int i,j;
        #pragma omp parallel shared(n, u) private(i, j)
        #pragma omp for collapse(2)
        for(int i=1; i<n-1; i++)
        {
            for(int j=2; j<n-1; j+=2)
            {
                u[n*i+j] = (f[n*i+j] + u[n*(i-1)+j] + u[n*(i+1)+j] + u[n*i+j-1] + u[n*i+j+1])/4.0;
                //std::cout << "\n" << f[n*i+j] << "\n"; 
            }
        }
        #pragma omp parallel shared(n, u) private(i, j)
        #pragma omp for collapse(2)
        for(int i=1; i<n-1; i++)
        {
            for(int j=1; j<n-1; j+=2)
            {
                u[n*i+j] = (f[n*i+j] + u[n*(i-1)+j] + u[n*(i+1)+j] + u[n*i+j-1] + u[n*i+j+1])/4.0;
                //std::cout << "\n" << f[n*i+j] << "\n"; 
            }
        }
    }
    return u;
}

std::vector<double> residual(int n, std::vector<double> u, std::vector<double> f)
{
    std::vector<double> r((n+1)*(n+1));
    n += 1;

    for(int i=0; i<n; i++)
    {
        for(int j=0; j<n; j++)
        {
            if(i==0 || j==0 || i==n-1 || j==n-1) r[n*i+j] = 0.0;
            else
            {
                r[n*i+j] = f[n*i+j] - 4.0*u[n*i+j] + u[n*(i-1)+j] + u[n*(i+1)+j] + u[n*i+j+1] + u[n*i+j-1];
            }
        }
    }
    return r;
}

std::vector<double> restriction(int n, std::vector<double> u)
/*
    Full-weighting.
*/
{
    int m = n/2+1;
    n += 1;
    std::vector<double> r(m*m);
    
    for(int i=0; i<m; i++)
    {
        for(int j=0; j<m; j++)
        {
            if(j == 0 || i == 0 || j == m-1 || i == m-1) r[m*i+j] = u[n*(2*i)+2*j];
            else
            {
                r[m*i+j] = (u[n*(2*i-1)+2*j-1] + u[n*(2*i-1)+2*j+1] + u[n*(2*i+1)+2*j-1] + u[n*(2*i+1)+2*j+1] +
                        2.0*(u[n*(2*i)+2*j-1] + u[n*(2*i)+2*j+1] + u[n*(2*i-1)+2*j] + u[n*(2*i+1)+2*j]) +
                        4.0*u[n*(2*i)+2*j])/16.0;
            }
        }
    }
    return r;
}

std::vector<double> interpolation(int n, std::vector<double> u)
/*
    Interpolação bilinear.
*/
{
    int m = n/2+1;
    n += 1;
    std::vector<double> v(n*n);

    for(int i=0; i<m; i++)
    {   
        for(int j=0; j<m; j++)
        {
            v[n*(2*i)+2*j] = u[m*i+j];
            if(i!=m-1) v[n*(2*i+1)+2*j] = (u[m*i+j] + u[m*(i+1)+j])/2.0;
            if(j!=m-1) v[n*(2*i)+2*j+1] = (u[m*i+j] + u[m*i+j+1])/2.0;
            if(i!=m-1 && j!=m-1) v[n*(2*i+1)+2*j+1] = (u[m*i+j] + u[m*(i+1)+j] + u[m*i+j+1] + u[m*(i+1)+j+1])/4.0;
        }
    }
    return v;
}

std::vector<double> vCycle(int n, std::vector<double> u, std::vector<double> f, int n1, int n2)
{
    if(n==2) //coarsest grid
    {
        u = gaussSeidel(n, u, f, 2);
        return u;
    }
    else
    {
        int m = n/2+1;
        u = gaussSeidel(n, u, f, n1);

        std::vector<double> r2 = restriction(n, residual(n, u, f));
        for(int i=0; i<m*m; i++) r2[i] *= 4.0;

        std::vector<double> v(m*m);
        v = vCycle(m-1, v, r2, n1, n2);     //chamada recursiva

        //correction
        for(int i=0; i<(n+1)*(n+1); i++) u[i] += interpolation(n, v)[i];

        //relax n2 times on Au = f
        u = gaussSeidel(n, u, f, n2);
        
        return u;
    }
}

std::vector<double> fullMultigrid(int n, double dom, std::vector<double> u, std::vector<double> f, double (*func)(double, double), int n0)
{
    if(n==2) //malha mais grossa
    {
        u = contorno(u, dom, func); //incorporando valores do contorno
        u = vCycle(n, u, f, 1, 1);
        return u;
    }
    else
    {
        int m = n/2+1;
        std::vector<double> u2(m*m);
        u2 = fullMultigrid(m-1, dom, u2, restriction(n, f), func, 1); //chamada recursiva
        for(int i=0; i<(n+1)*(n+1); i++)
        {
            u[i] = interpolation(n, u2)[i];
        }
        u = contorno(u, dom, func); //incorporando valores do contorno
        for(int i=0; i<n0; i++) u = vCycle(n, u, f, 1, 1);
        return u;
    }
}

//---------------------------------------------------------------------------------------------
std::vector<double> contorno(std::vector<double> u, double dom, double (*func)(double, double))
/*
    Preenche o contorno de u com seus respectivos valores quando aplicada a função func.
*/
{
    int n = sqrt(u.size());
    double h = dom/(n-1);

    for(int j=0; j<n; j++)
    {
        u[j] = func(0.0, j*h);
    }

    for(int i=1; i<n-1; i++)
    {
        u[i*n] = func(i*h, 0.0);
        u[i*n + n-1] = func(i*h, dom);
    }
    
    for(int j=0; j<n; j++)
    {
        u[(n-1)*n + j] = func(dom, j*h);
    }
    return u;
}

std::vector<double> geraTeste(int n, double dom, double (*func)(double, double), bool ch2)
/*
    Essa função serve apenas para gerar os vetores do tamanho desejado com as entradas 
    sendo dadas pela função escolhida, no domínio desejado.
*/
{
    double h = dom/n, x, y;
    double h2;

    if(ch2 == true) h2 = pow(h, 2.0);
    else h2 = 1.0;

    n += 1;
    std::vector<double> f(n*n);
    
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<n; j++)
        {
            x = (i+1)*h;
            y = (j+1)*h;
            f[i*n + j] = func(x, y)*h2;
        }
    }
    return f;
}

void print(std::vector<double> u)
/*
    Printa a matriz u.
*/
{
    int n = sqrt(u.size());
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<n; j++)
        {
            std::cout << u[n*i+j] << " ";
        }
        std::cout << "\n";
    }
}

void write(std::vector<double> u, double dom, std::string name)
/*
    Essa função serve para escrever em arquivos .dat o vetor u obtido e também gerar os
    pontos da malha x y com a finalidade de plotar o gráfico.
*/
{
    int n = sqrt(u.size());
    double h = dom/(n-1);

    //Escrevendo os dados em arquivos .dat
    std::stringstream s_xy, s_u;
    s_xy << name << "_xy.dat";
    s_u << name << "_u.dat";
    std::ofstream write_output1(s_xy.str(), std::ios::app);
    std::ofstream write_output2(s_u.str(), std::ios::app);

    assert(write_output1.is_open());
    assert(write_output2.is_open());

    for(int i=0; i<n; i++)
    {
        write_output1 << i*h << "\n";
    }
    for(int i=0; i<n*n; i++)
    {
        write_output2 << u[i] << "\n";
    }

    write_output1.close();
    write_output2.close();
    return;
}

double maxError(int n, double dom, std::vector<double> u, double (*func)(double, double))
/*
    Calcula o erro máximo do vetor de aproximações obtido u em relação à solução exata, dada
    pela função func.
*/
{
    n += 1;
    std::vector<double> erro(n*n);
    std::vector<double> u_exato = geraTeste(n-1, dom, func, false);
    
    for(int i=0; i<n*n; i++)
    {
        erro[i] = std::abs(u[i]-u_exato[i]);
        //std::cout << erro[i] << "\n";
    }
    double erro_max = *std::max_element(erro.begin(), erro.end());
    return erro_max;
}

double c01(double x, double y)
/*
    Função que dá os valores do contorno/solução exata do teste 01.
*/
{
    return x*(1.0-x) + y*(1.0-y);
}

double c02(double x, double y)
/*
    Função que dá os valores do contorno/solução exata do teste 02.
*/
{
    return 0.0;
}

double f02(double x, double y)
/*
    Função que dá os valores de f do teste 02.
*/
{
    return sin(x)*sin(y);
}
