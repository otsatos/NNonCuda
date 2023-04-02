#include<iostream>
#include<cuda_runtime.h>
#include<tuple>
#include<memory>

#include "exceptionext.h"
#include "dataset.h"
#include "activationf.h" 

#include "matrixoperations.h"
#include "memallocations.h"
#include "matrix.h"
#include "training.h"

using namespace std;

#ifdef _WIN32
    const std::string _RTENV = "_WIN32";
#else
    const std::string _RTENV = "_UNIX";
#endif


int main(int argc, char **argv)
{   
    float alpha = 0.01f; 
    modelclasses *modelClasses = modelclasses::createModelclasses();   
    irisdataset ds("./data/iristraining.data", false, 4, 3);

    std::cout << "Get inputs/outputs from file..." << "\n";
    ds.irisDatasetFromfile(modelClasses);
    //ds.printData();
    
    auto labels=modelClasses->getLabels();
    std::cout << "\nmodel Labels" << "\n";
    std::cout << "-----------------------------------------------------" << "\n";
    for (auto e : labels) std::cout << e.first << ":"<<e.second << "\n";
        
    auto X=ds.getInputs();
    auto Y=ds.getOutputs();
    
    X.print();
    Y.print();
   
    //////////////////////////////////////////////////////////////////////// 
    // hidden layer 
    //////////////////////////////////////////////////////////////////////// 
    int hlNeurons=10;
    auto W1=generateWeights(ds.getNumberOfFeatures(),hlNeurons);
    auto W2=generateWeights(hlNeurons,ds.getNumberOfLabels());
    
    std::cout << "\n Initial Weights"<< "\n";
    std::cout << "-----------------------------------------------------" "\n";  
    W1.print();
    W2.print();   
    
    int epoch=50;
    std::cout << "Model Training..." << "\n";
    std::cout << "-----------------------------------------------------" << "\n";
    auto [accuracy,loss,W1t,W2t] = train(X,Y,W1,W2,alpha,epoch);    
    
    std::cout << "\n Final Weights"<< "\n";
    std::cout << "-----------------------------------------------------" "\n";  
    W1t.print();
    W2t.print();

    std::cout << "\nAccuracy" << "\n";
    std::cout << "-----------------------------------------------------" << "\n";
    int i=0,j=0;
    for(auto acc:accuracy) {std::cout << ++i << ":" << acc <<"\n";}

    std::cout << "\nLoss" << "\n";
    std::cout << "-----------------------------------------------------" << "\n";
    for(auto l:loss) {std::cout << ++j << ":" << l <<"\n";}
    
    std::cout << "Model Testing on Test Samples" << "\n";
    std::cout << "-----------------------------------------------------" << "\n";
    irisdataset dsTest("./data/iristest.data", false, 4, 3);
    dsTest.irisDatasetFromfile(modelClasses);
    std::cout << "\n";
    
    auto Xtest=dsTest.getInputs(); 
    auto Ytest=dsTest.getOutputs();
       
    for (int i=0;i<Xtest.height;i++)
    {
        auto predictions = feedForward(Xtest[i],W1t,W2t);

        float pAcc=predictions.maxElement();  
        auto yIdx=predictions.indexOfmax();
        
        if (yIdx >-1) 
        {               
            std::cout << "Actual Y value:" << modelClasses->getLabelByYId(Ytest[i].indexOfmax() + 1) << " "; 
            std::cout << "Predicted Y value:"<<modelClasses->getLabelByYId(yIdx+1) << " Prediction Accuracy: " << pAcc *100 << "\n";
        }              
    }

}