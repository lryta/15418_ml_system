#include "data.h"
#include "net.h"
#include "optimizer.h"

namespace MLLib {

class trainerConfig {
 public:
  trainerConfig(int e, int b):
    epochNum_(e), batchNum_(b) {}

  int epochNum_;
  int batchNum_;
};

//TODO: Not sure we want to support test()
class modelTrainer {
 public:
  modelTrainer(NNType, std::vector<int>, trainerConfig);
  void train(int epoch);

 private:
  int correctly_labelled_num_, labelled_num_;

  net* net_;
  optimizer* weight_updater_;
  dataIterator* train_iter_;
  trainerConfig config_;
};

}
