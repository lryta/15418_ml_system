#ifndef _TINYML_LAYER_H
#define _TINYML_LAYER_H

#include <vector>
#include "tensor.h"
#include "operations/matrixOp.h"

namespace TinyML{
  
using std::vector;

// TODO: the initialization of layer might require context (cpu vs gpu)

class layer {
 public:
  // Given the shapes of inputs, infer number & shape of outputs
  // Note: when the number of ous is zero, it means it doens't have an output
  layer(vector<shape> &ins, vector<shape> &ous) {}

  virtual void init(vector<shape> &ins, vector<shape> &ous) {
    inferShape(ins, ous);
    initWeight(ins, ous);
    initIntermediateState(ins, ous);
  }

  virtual void inferShape(vector<shape> &ins, vector<shape> &ous) {}
  virtual void initWeight(vector<shape> &ins, vector<shape> &ous) {}
  virtual void initIntermediateState(vector<shape> &ins, vector<shape> &ous) {}
  virtual vector<tensor*> getParam() = 0;

  virtual void forward(vector<tensor*> ins, vector<tensor*> ous) = 0;
  virtual void backward(vector<tensor*> ins, vector<tensor*> ous) = 0;

};

class Losslayer: public layer {
 public:
  Losslayer(vector<shape> &ins, vector<shape> &ous):layer(ins, ous) {}

  void inferShape(vector<shape> &ins, vector<shape> &ous) {
    assert(ins.size() == 2);
    ous.clear();
  }

  void initWeight(vector<shape> &ins, vector<shape> &ous) {}

  virtual int correctlyRecognizedDataNum() = 0;
  virtual float getLoss() = 0;

};

}

#endif
