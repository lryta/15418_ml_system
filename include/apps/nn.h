#include "application.h"

namespace MLLib {

// At present. Support only MLP
enum NNType {
  MLPNet;
}

class NN {
 public:
  NN(NNType type, vector<int> inter_dims, appConfig config);
  void Run();

 private:
  appConfig config_;
  Solver solver_;
};

}
