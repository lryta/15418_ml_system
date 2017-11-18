#include<vector>
#include<tensor.h>

namespace MLLib{
  
using std::vector;

class Layer {
  public:
    // TODO: might be context
    Layer() {
      initWeight();
    }

    virtual void forward(const vector<tensor> &ins, vector<tensor> &ous);
    virtual void backward();
    virtual void initWeight();
};

class LossLayer : public Layer {
  public:
    virtual float getLoss();
};

}
