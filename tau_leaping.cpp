/*
    Simulação numérica dos sistemas apresentados no capítulo 7 de Stochastic Modelling for Systems Biology, de Wilkinson 
    ("Dimerisation kinetics" e "Michaelis-Menten kinetics") com o método tau-leaping.
    
    Compilar com: g++ tau_leaping.cpp -o tau_leaping -std=c++11 -larmadillo
*/

#include <iostream>
#include <fstream>
#include <cassert>
#include <armadillo>
#include <cmath>
#include <random>
#include <string>


void tau_leaping(int modelo, double tfinal, arma::mat V, arma::vec X, arma::vec c, int sim);
arma::vec propensity(int modelo, arma::vec X, arma::vec c);
arma::vec poisson(arma::vec a, double tau);


int main(int argc, char** argv)
{  
    /*
    std::cout << "Digite o número correspondente ao modelo desejado: \n1 - Dimerisation kinetics\n2 - Michaelis-Menten enzyme kinetics\n";
    std::string modelo, tempo, nsim;
    std::cin >> modelo;
    std::cout << "\nDigite quantas simulações serão feitas: \n";
    std::cin >> nsim;
    std::cout << "\nDigite a duração de cada simulação (tempo em segundos): \n";
    std::cin >> tempo;
    */

    std::string modelo = "1";
    std::string nsim = "1";
    std::string tempo = "120";

    //Dimerisation kinetics
    arma::vec X1;
    X1.load("X01.dat");

    arma::mat V1 = {{-2,   2},
                    { 1,  -1}};
    arma::vec c1 = {1.66*pow(10, -3), 0.2};
    

    //Michaelis-Menten enzyme kinetics
    arma::vec X2;
    X2.load("X02.dat");

    arma::mat V2 = {{-1,  1,  0},
                    {-1,  1,  1},
                    { 1, -1, -1},
                    { 0,  0,  1}};
    arma::vec c2 = {1.66*pow(10, -3), pow(10, -4), 0.1};


    int n = stoi(nsim);
    double tfinal = stod(tempo);

    if(modelo == "1")
    {
        for(int i=0; i<n; i++)
        {
            tau_leaping(1, tfinal, V1, X1, c1, i);
        }
    }
    else
    {
        if(modelo == "2")
        {
            for(int i=0; i<n; i++)
            {
                tau_leaping(2, tfinal, V2, X2, c2, i);
            }
        }
        else
        {
            std::cout << "Escolha 1 ou 2 para indicar o modelo. Tente novamente.\n";
        }
    }

    return 0;
}

void tau_leaping(int modelo, double tfinal, arma::mat V, arma::vec X, arma::vec c, int sim)
{
    double j = 0;
    double tau = 0.04;
    double t = 0;
    arma::vec a;
    arma::vec p;
    arma::mat maux;
    arma::vec vaux;

    //Para gerar arquivos com nomes diferentes 
    std::stringstream sx, st;
    sx << "x" << std::to_string(sim) << ".dat";
    st << "t" << std::to_string(sim) << ".dat";

    while(t < tfinal)
    {   
        a = propensity(modelo, X, c);
        p = poisson(a, tau);

        //Escrevendo no arquivo
        std::ofstream write_output1(sx.str(), std::ios::app);
        assert(write_output1.is_open());
        write_output1 << X << " \n";
        write_output1.close();

        std::ofstream write_output2(st.str(), std::ios::app);
        assert(write_output2.is_open());
        write_output2 << t << " \n";
        write_output2.close();

        maux = V;
        maux.each_row() %= p.t();
        vaux = sum(maux, 1);

        X += vaux;
        t += tau;
    }
}

arma::vec propensity(int modelo, arma::vec X, arma::vec c)
{
    arma::vec a(arma::size(c), arma::fill::zeros);

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

arma::vec poisson(arma::vec a, double tau)
{
    int n = a.n_elem;
    arma::vec p(n, arma::fill::zeros);
    std::mt19937 mt{std::random_device{}()};
    
    //a.print("a = ");

    for(int i=0; i<n; i++)
    {   
        std::poisson_distribution<> pd{a(i)*tau};
        p(i) = pd(mt);
    }

    return p;
}