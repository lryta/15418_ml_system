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

  virtual vector<layer*> getLayers() = 0;
  virtual vector<tensor*> getParams() = 0;
  virtual vector<size_t> getParamIdToLayerIdMap() = 0;
  virtual void collectTimeStat() = 0;
  virtual void printAvgTimeStat() = 0;
};

class MLPnet : public net {
 public:
  MLPnet(shape in_shape, shape target_shape, vector<size_t> hidden_dims)
    :net() {
    buildlayers(in_shape, target_shape, hidden_dims);
  }
  ~MLPnet();

  virtual void forward(vector<tensor*> ins, vector<tensor*> targets);
  virtual void backward(vector<tensor*> ins, vector<tensor*> targets);

  virtual int correctlyRecognizedDataNum();
  virtual float getLoss();

  virtual vector<layer*> getLayers() {
    return layers_;
  }
  virtual vector<tensor*> getParams();
  virtual vector<size_t> getParamIdToLayerIdMap();

  virtual void collectTimeStat() {
    for (size_t i = 0; i < layers_.size(); ++i)
      layers_[i]->collectTimeStat();
  }

  void printAvgTimeStat() {
    for (size_t i = 0; i < layers_.size(); ++i)
      layers_[i]->printAvgTimeStat();
  }

 private:
  void buildlayers(shape in_shape, shape target_shape, vector<size_t> hidden_dims);

  vector<layer*> layers_;
  vector<tensor*> intermediate_tensors_, params_;
  vector<size_t> param_id_to_layer_id_;
};

}

#endif
