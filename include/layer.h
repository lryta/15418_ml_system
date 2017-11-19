#include<vector>
#include<tensor.h>

namespace MLLib{
  
using std::vector;

// TODO: the initialization of layer might require context (cpu vs gpu)

class Layer {
 public:
  // Given the shapes of inputs, infer number & shape of outputs
  Layer(vector<shape> &ins, vector<shape> &ous) {
    inferShape(ins, ous);
    initWeight(ins, ous);
    initIntermediateState(ins, ous);
  }

  virtual void forward(const vector<tensor> &ins, vector<tensor> &ous);
  virtual void backward();
  virtual void inferShape(vector<shape> &ins, vector<shape> &ous);
  virtual void initWeight(vector<shape> &ins, vector<shape> &ous);
  virtual void initIntermediateState(vector<shape> &ins, vector<shape> &ous);
  virtual vector<Tensor&> getParam();
};

class LossLayer:Layer {
 public:
  LossLayer(vector<shape> &ins, vector<shape> &ous):Layer(ins, ous) {
  }

  virtual float getLoss();
};

}
