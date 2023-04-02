#include "training.h"

using namespace std;

tuple<vector<float>,vector<float>,Matrix,Matrix> train(Matrix X, Matrix Y, Matrix W1, Matrix W2, float alpha,int  epoch)
{
    std::vector<float> accuracy;
    std::vector<float> loss; 
    std::vector<float> l;
    
    float acc=0.0f;
    float lsum=0.0f;
    for (int j=0;j<epoch;j++)
    {        
        acc=0.0f;
        lsum=0.0f;
        for(int i=0;i<X.height;i++)
        {
            auto out=feedForward(X[i],W1,W2);               
            auto losserror=mseLoss(out,Y[i]);   
            if (!isinf(losserror)) l.push_back(losserror);            

            auto [tw1,tw2]=backPropagation(X[i],Y[i],W1,W2,alpha);
            W1=tw1;
            W2=tw2;            
           
            // free temporarily allocated memory           
            if (out.width!=0 && out.elements!=nullptr) free(out.elements); //std::cout << "Clean up memory for matrix out...Ok!"<<"\n";            
        }
        
        for(auto e:l) lsum+=e;      

        acc= (1 - (lsum/X.height))*100; 
        accuracy.push_back(acc);
        loss.push_back(lsum/X.height);
        std::cout << "epochs:"<< j + 1<<"============================= accuracy:" << acc << " loss:"<< lsum <<"\n";        
        
        l.clear();       
    }
    return {accuracy,loss,W1,W2};
}