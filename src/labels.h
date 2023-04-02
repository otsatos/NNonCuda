#include <iostream>
#include <map>

using namespace std;



class modelclasses
{
private:
    static modelclasses *thisinstance;
    modelclasses()
    {
    }

    int maxLabelindex;
    std::map<std::string, int> labels;

    int labelIndex(std::string key)
    {
        std::map<string, int>::const_iterator pos = labels.find(key);
        if (pos != labels.end())
            return pos->second;
        return -1;
    }

public:
    modelclasses(const modelclasses &obj) = delete;
    void operator=(const modelclasses &obj) = delete;

    static modelclasses *createModelclasses();

    int addLabel(std::string key)
    {
        int y = labelIndex(key);
        if (y < 0)
        {
            ++maxLabelindex;
            y = maxLabelindex;
            this->labels[key] = y;
        }
        return y;
    }

    int getLabelValue(std::string key)
    {
        int y = labelIndex(key);
        if (y < 0)
            throw std::invalid_argument("Label <" + key + "> does not exists!");

        return y;
    }

    std::map<std::string, int> const getLabels()
    {
        return this->labels;
    }

    std::string getLabelByYId(int id)
    {
       std::string r="";

       for (auto it = labels.begin(); it != labels.end(); ++it)
            if (it->second == id) r=it->first;

       return r;     
    }
};

modelclasses *modelclasses::thisinstance = nullptr;
modelclasses *modelclasses::createModelclasses()
{
    if (thisinstance == nullptr)
        thisinstance = new modelclasses();

    return thisinstance;
}
