#ifndef _TINYML_NET_H
#define _TINYML_NET_H

#include <tuple>
#include <vector>

#include "layer.h"

namespace TinyML {
using std::vector;

enum ModelType {
  MLPnet
};

class net {
 public:
  virtual void foward(vector<tensor*> ins, vector<tensor*> targets);
  virtual void backward(vector<tensor*> ins, vector<tensor*> targets);
  virtual float getLoss();
  virtual int correctlyRecognizedDataNum();

  virtual vector<tensor*> getParams();
};

class MLPnet : net {
 public:
  MLPnet(shape in_shape, vector<size_t> hidden_dims):net() {
    buildlayers(in_shape, hidden_dims);
  }

  ~MLPnet();

  std::vector<tensor*> getParams();

 private:
  void buildlayers(shape in_shape, vector<size_t> hidden_dims);

  std::vector<layer*> layers_;
  std::vector<tensor*> intermediate_tensors_, params_;
};

}

#endif
