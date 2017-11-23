#include <tuple>
#include <vector>

namespace MLLib {
class Net {
 public:
  void foward(vector<Tensor*> ins, vector<Tensor*> targets);
  void backward(vector<Tensor*> ins, vector<Tensor*> targets);

  vector<Tensor*> getParams();
}

class MLPNet : Net {
 public:
  MLP(shape in_shape, vector<size_t> hidden_dims):Net() {
    buildLayers(in_shape, hidden_dims);
  }

  ~MLP();

  vector<Tensor*> getParams();

 private:
  void buildLayers(shape in_shape, vector<size_t> hidden_dims);

  std::vector<Layer*> layers_;
  std::vector<Tensor*> intermediate_tensors_, params_;
}
}
