#ifndef _TINYML_LAYER_H
#define _TINYML_LAYER_H

#include <vector>
#include <chrono>
#include <cstdio>
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

  void Forward(vector<tensor*> ins, vector<tensor*> ous) {
    op_start_time_ = std::chrono::system_clock::now();
    forward(ins, ous);
    forward_wall_time_ = std::chrono::system_clock::now() - op_start_time_;
  }

  void Backward(vector<tensor*> ins, vector<tensor*> ous) {
    op_start_time_ = std::chrono::system_clock::now();
    backward(ins, ous);
    backward_wall_time_ = std::chrono::system_clock::now() - op_start_time_;
  }

  void printTimeStat() const {
    printf("forward %f ms, backward %f ms, update %f ms\n",
        forward_wall_time_.count() * 1000,
        backward_wall_time_.count() * 1000,
        update_wall_time_.count() * 1000);
  }

  void resetUpdateWeightTime() {
    update_wall_time_ = std::chrono::duration<float>::zero();
  }

  void startUpdateWeight() {
    op_start_time_ = std::chrono::system_clock::now();
  }

  void endUpdateWeight() {
    update_wall_time_ += std::chrono::system_clock::now() - op_start_time_;
  }

 private:
  std::chrono::time_point<std::chrono::system_clock> op_start_time_;
  std::chrono::duration<float> forward_wall_time_,
    backward_wall_time_, update_wall_time_;
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
