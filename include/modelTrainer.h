#include "data.h"
#include "optim.h"

namespace MLLib {

class trainerConfig{
 public:
  trainerConfig(int e):epochNum_(e) {}
  int epochNum_;
};

//TODO: Not sure we want to support test()
class modelTrainer {
 public:
  modelTrainer(net* net, DataIterator *itrainIter, int epoch);
  void train(int epoch);

 private:
  int correctly_labelled_num_, labelled_num_;

  net* net_;
  optim* weight_updater_;
  dataIterator* train_iter_;
  trainerConfig config_;
};

}
