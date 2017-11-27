#include <tuple>
#include <vector>

namespace TinyML {

enum ModelType {
  MLPNet;
};

class Net {
 public:
  virtual void foward(vector<tensor*> ins, vector<tensor*> targets);
  virtual void backward(vector<tensor*> ins, vector<tensor*> targets);
  virtual float getLoss();
  virtual int correctlyRecognizedDataNum();

  virtual vector<tensor*> getParams();
}

class MLPNet : Net {
 public:
  MLP(shape in_shape, vector<size_t> hidden_dims):Net() {
    buildLayers(in_shape, hidden_dims);
  }

  ~MLP();

  std::vector<tensor*> getParams();

 private:
  void buildLayers(shape in_shape, vector<size_t> hidden_dims);

  std::vector<Layer*> layers_;
  std::vector<tensor*> intermediate_tensors_, params_;
}

}
