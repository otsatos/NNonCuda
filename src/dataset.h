
#include <fstream>
#include <sstream>
#include "labels.h"
#include "matrix.h"
#include "memallocations.h"

class irisdataset
{
private:   
    float **xx = nullptr;
    float **yy = nullptr;

    Matrix inputs;
    Matrix outputs;

    std::string fname;
    bool headerLine=false;
    int numberOfInputs = 0;
    int numberOfFeautures=0;
    int numberOfLabels = 0;

    int countSamples()
    {
        std::string l;
        numberOfInputs = 0;

        std::ifstream f(fname);
        if (!f.is_open())
            throw std::runtime_error("Could not open file dataset");

        while (std::getline(f, l))
            ++numberOfInputs;
        f.close();

        if (headerLine && numberOfInputs > 0)
            --numberOfInputs;

        return numberOfInputs;
    }

public:
    irisdataset(std::string file, bool firstlineHeaders, int feautures, int labels) : fname(file),
                                                                                      headerLine(firstlineHeaders),
                                                                                      numberOfFeautures(feautures),
                                                                                      numberOfLabels(labels)
    {
    }

    
    void irisDatasetFromfile(modelclasses *modelLabels)
    {
        int samples = countSamples();

        if (0 >=samples) throw std::runtime_error("Can't determine the number of samples!");

        
        inputs  = {numberOfFeautures, samples, allocateHostMemory(numberOfFeautures * samples)}; //{w,h,n}
        outputs = {numberOfLabels, samples, allocateHostMemory(numberOfLabels * samples)};

        for (int i = 0; i < numberOfLabels * samples;i++) outputs.elements[i] = 0.0f;
        ////////////////////////////////////////
        //std::cout << "input output vectorized matrices Initialized...." << "\n";
        ////////////////////////////////////////

        std::string l, w;
        std::ifstream f(this->fname);

        if (!f.is_open()) throw std::runtime_error("Could not open file dataset");

        int i = -1;
        int inputindex = -1;
        while (std::getline(f, l))
        {
            ++i;
            std::stringstream s(l);
           
            int j = -1;
            while (std::getline(s, w, ','))
            {
                ++j;

                if (j < numberOfFeautures && ++inputindex < (numberOfFeautures * samples))                                                      
                    inputs.elements[inputindex] = std::stof(w);                                   
                else 
                {                    
                    int labelIndex = modelLabels->addLabel(w);
                    int outputindex = (i * numberOfLabels) + (labelIndex - 1);
                    if (labelIndex > 0 && outputindex < (numberOfLabels * samples)) outputs.elements[outputindex] = 1.0f;                    
                }
            }
        }
        f.close();
        std::cout << "Input/Output  data where imported successfully...\n";       
    }

    void irisDatasetFromfile2D(modelclasses *modelLabels)
    {
        int samples = countSamples();

        if (0 >=samples) throw std::runtime_error("Can't determine the number of samples!");
        
        xx = new float *[samples];
        yy = new float *[samples];

        std::string l, w;
        std::ifstream f(this->fname);

        if (!f.is_open())
            throw std::runtime_error("Could not open dataset file!");

        int i = -1;
        while (std::getline(f, l))
        {
            ++i;
            xx[i] = new float[numberOfFeautures];
            yy[i] = new float[numberOfLabels];
            for (int k = 0; k < numberOfLabels; k++)
            {
                yy[i][k] = 0.0f;
            }

            std::stringstream s(l);

            int j = -1;
            while (std::getline(s, w, ','))
            {
                ++j;
                if (j < numberOfFeautures)
                {
                    xx[i][j] = std::stof(w);
                }
                else
                {
                    int labelIndex = modelLabels->addLabel(w);
                    yy[i][labelIndex - 1] = 1.0f;
                }
            }
        }
        f.close();
    }

    int getNumberOfInputs() { return numberOfInputs; }

    int getNumberOfFeatures() { return numberOfFeautures; }

    int getNumberOfLabels() { return numberOfLabels; }
    
    Matrix getInputs()
    {
        return inputs;
    }

    Matrix getOutputs()
    {
        return outputs;
    }

    void printData()
    {
        std::cout << "samples: " << numberOfInputs << ", Feautures: " << numberOfFeautures << "\n";
        for (int i = 0; i < numberOfInputs; i++)
        {
            for (int j = 0; j < numberOfFeautures; j++)
                std::cout << inputs.elements[(i + 1) * (j + 1) - 1] << ((j + 1) % numberOfFeautures ==0?"\n":",");          
        }
    }

    void checkdata2D()
    {
        std::cout << numberOfInputs << "," << numberOfFeautures << "\n";
        for (int i = 0; i < numberOfInputs; i++)
        {
            for (int j = 0; j < numberOfFeautures; j++)
                std::cout << xx[i][j] << ",";

            std::cout << " -> ";

            for (int k = 0; k < numberOfLabels; k++)
                std::cout << yy[i][k] << ",";

            std::cout << "\n";
        }
    }

    ~irisdataset()
    {
        if (xx != nullptr)
        {
            for (int i = 0; i < numberOfInputs; i++)
            {
                delete[] xx[i];
            }
            delete[] xx;
        }
        if (yy != nullptr)
        {
            for (int i = 0; i < numberOfLabels; i++)
            {
                delete[] yy[i];
            }
            delete[] yy;
        }

        if (inputs.elements != nullptr)
            free(inputs.elements);
        if (outputs.elements != nullptr)
            free(outputs.elements);
    }
};