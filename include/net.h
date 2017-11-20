#include <vector>


namespace MLLib {
  class Net {
    public:
      std::tuple<int, int> forward(const vector<Tensor> &ins, vector<Tensor> &ous); // return (loss, acc)
      void backward();
  }

  class SeqNet : Net {
    public:
      std::tuple<int, int> forward(const vector<Tensor> &ins, vector<Tensor> &ous); // return (loss, acc)
  }

  class LogReg : SeqNet {
    public:
      LogReg(size_t inputD):inputDim(inputD) {}
   
    private:
      size_t inputDim_;
      size_t hiddenDim_;
      std::vector<Layer> layers_;
      std::vector<std::vector<Tensor> > cache_;
  }
}
