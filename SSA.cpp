/*
    Simulação numérica dos sistemas apresentados no capítulo 7 de Stochastic Modelling for Systems Biology, de Wilkinson 
    ("Dimerisation kinetics" e "Michaelis-Menten kinetics") com SSA.
    
    Compilar com: g++ SSA.cpp -o SSA -std=c++11 -larmadillo
*/

#include <iostream>
#include <fstream>
#include <cassert>
#include <armadillo>
#include <cmath>
#include <random>
#include <string>


using namespace std;
using namespace arma;

void SSA(int modelo, double tfinal, mat V, vec X, vec c, int sim);
vec propensity(int modelo, vec X, vec c);
vec csum(int modelo, vec a, double asum);


int main(int argc, char** argv)
{  
    /*
    cout << "Digite o número correspondente ao modelo desejado: \n1 - Dimerisation kinetics\n2 - Michaelis-Menten enzyme kinetics\n";
    string modelo, tempo, nsim;
    cin >> modelo;
    cout << "\nDigite quantas simulações serão feitas: \n";
    cin >> nsim;
    cout << "\nDigite a duração de cada simulação (tempo em segundos): \n";
    cin >> tempo;
    */
    string modelo = "1";
    string nsim = "1";
    string tempo = "120";
   
    //Dimerisation kinetics
    vec X1;
    X1.load("X01.dat");
    
    mat V1 = {{-2,   2},
              { 1,  -1}};
    /*
    mat V1 = {{-2,   1},
              { 2,  -1}};
    */
    vec c1 = {1.66*pow(10, -3), 0.2};

    //Michaelis-Menten enzyme kinetics
    vec X2;
    X2.load("X02.dat");

    mat V2 = {{-1,  1,  0},
             {-1,  1,  1},
             { 1, -1, -1},
             { 0,  0,  1}};

    vec c2 = {1.66*pow(10, -3), pow(10, -4), 0.1};


    int n = stoi(nsim);
    double tfinal = stod(tempo);

    if(modelo == "1")
    {
        for(int i=0; i<n; i++)
        {
            SSA(1, tfinal, V1, X1, c1, i);
        }
    }
    else
    {
        if(modelo == "2")
        {
            for(int i=0; i<n; i++)
            {
                SSA(2, tfinal, V2, X2, c2, i);
            }
        }
        else
        {
            cout << "Escolha 1 ou 2 para indicar o modelo. Tente novamente.\n";
        }
    }

    return 0;
}

void SSA(int modelo, double tfinal, mat V, vec X, vec c, int sim)
{
    double asum = 0;
    double j = 0;
    mt19937_64 mt(time(nullptr));
    uniform_real_distribution<double> dis(0.0, 1.0);
    double rand1, rand2;
    double tau;
    double t = 0;
    vec a;

    //Para gerar arquivos com nomes diferentes 
    stringstream sx, st;
    sx << "x" << to_string(sim) << ".dat";
    st << "t" << to_string(sim) << ".dat";

    while(t < tfinal)
    {   
        a = propensity(modelo, X, c);
        //a.print("a = ");
        asum = sum(a);
        //cout << "asum = " << asum << "\n";

        vec cumsum = csum(modelo, a, asum);
        //cumsum.print("cumsum = ");
        rand1 = dis(mt);
        //cout << "rand1 = " << rand1 << "\n";
        j = min(find(rand1 < cumsum));
        //cout << "j = " << j << "\n";
        rand2 = dis(mt);
        //cout << "rand2 = " << rand2 << "\n";
        tau = log(1/rand2)/asum;
        
        ofstream write_output1(sx.str(), std::ios::app);
        assert(write_output1.is_open());
        write_output1 << X << " \n";
        write_output1.close();

        ofstream write_output2(st.str(), std::ios::app);
        assert(write_output2.is_open());
        write_output2 << t << " \n";
        write_output2.close();

        X += V.col(j);
        t += tau;
    }
}

vec propensity(int modelo, vec X, vec c)
{
    vec a(arma::size(c), fill::zeros);

    if(modelo == 1) //Dimerisation kinetics
    {
        //Dimerisation
        a(0) = c(0)*X(0)*(X(0)-1)/2;
        //Dissociation
        a(1) = c(1)*X(1);
    }
    else if(modelo == 2) //Michaelis-Menten enzyme kinetics
    {
        //Binding
        a(0) = c(0)*X(0)*X(1);
        //Dissociation
        a(1) = c(1)*X(2);
        //Conversion
        a(2) = c(2)*X(2);
    }

    return a;
}

vec csum(int modelo, vec a, double asum)
{
    vec cumsum(arma::size(a), fill::zeros);

    for(int i=0; i <= modelo; i++)
    {
        for(int j=0; j<= i; j++)
        {
            cumsum(i) += a(j)/asum;
        }
    }
    
    return cumsum;
}
