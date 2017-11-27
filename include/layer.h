#include<vector>
#include<tensor.h>

namespace TinyML{
  
using std::vector;

// TODO: the initialization of layer might require context (cpu vs gpu)

class Layer {
 public:
  // Given the shapes of inputs, infer number & shape of outputs
  // Note: when the number of ous is zero, it means it doens't have an output
  Layer(vector<shape> &ins, vector<shape> &ous) {
    inferShape(ins, ous);
    initWeight(ins, ous);
    initIntermediateState(ins, ous);
  }

  virtual void inferShape(vector<shape> &ins, vector<shape> &ous);
  virtual void initWeight(vector<shape> &ins, vector<shape> &ous);
  virtual void initIntermediateState(vector<shape> &ins, vector<shape> &ous);
  virtual vector<Tensor&> getParam();

  virtual void forward(vector<tensor> &ins, vector<tensor> &ous);
  virtual void backward(vector<tensor> &ins, vector<tensor> &ous);
};

class LossLayer:Layer {
 public:
  LossLayer(vector<shape> &ins, vector<shape> &ous):Layer(ins, ous) {
  }

  void inferShape(vector<shape> &ins, vector<shape> &ous) {
    ous.clear();
  }

  void initWeight(vector<shape> &ins, vector<shape> &ous) {
  }

  virtual int correctlyRecognizedDataNum();
  virtual float getLoss();

};

}
