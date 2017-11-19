#include<vector>
#include<tensor.h>

namespace MLLib{
  
using std::vector;

class Layer {
 public:
  // TODO: might be context
  Layer(vector<tensor> &ins, vector<tensor> &ous) {
    initWeight();
  }

  virtual void forward(const vector<tensor> &ins, vector<tensor> &ous);
  virtual void backward();
  virtual void initWeight();
  virtual vector<Tensor&> getParam();
};

class LossLayer:Layer {
 public:
  virtual float getLoss();
};

}
