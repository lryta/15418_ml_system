#ifndef _TINYML_NET_H
#define _TINYML_NET_H

#include <tuple>
#include <vector>

#include "layer.h"

namespace TinyML {
using std::vector;

enum ModelType {
  MLPnetType
};

class net {
 public:
  virtual void forward(vector<tensor*> ins, vector<tensor*> targets) = 0;
  virtual void backward(vector<tensor*> ins, vector<tensor*> targets) = 0;
  virtual float getLoss() = 0;
  virtual int correctlyRecognizedDataNum() = 0;

  virtual vector<tensor*> getParams() = 0;
};

class MLPnet : public net {
 public:
  MLPnet(shape in_shape, shape target_shape, vector<size_t> hidden_dims):net() {
    buildlayers(in_shape, target_shape, hidden_dims);
  }
  ~MLPnet();

  virtual void forward(vector<tensor*> ins, vector<tensor*> targets);
  virtual void backward(vector<tensor*> ins, vector<tensor*> targets);

  virtual int correctlyRecognizedDataNum();
  virtual float getLoss();

  std::vector<tensor*> getParams();

 private:
  void buildlayers(shape in_shape, shape target_shape, vector<size_t> hidden_dims);

  std::vector<layer*> layers_;
  std::vector<tensor*> intermediate_tensors_, params_;
};

}

#endif
