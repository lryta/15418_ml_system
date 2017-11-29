#ifndef _TINYML_MODELTRAINER_H
#define _TINYML_MODELTRAINER_H

#include <string>
#include "data.h"
#include "net.h"
#include "optimizer.h"

namespace TinyML {

class trainerConfig {
 public:
  trainerConfig(int e, int b, std::string dir):
    epochNum_(e), batchNum_(b), dataset_dir_(dir) {}

  int epochNum_;
  int batchNum_;
  std::string dataset_dir_;
};

//TODO: Not sure we want to support test()
class modelTrainer {
 public:
  modelTrainer(trainerConfig*);

  void setModel(ModelType, std::vector<int>);
  void setOptimizer(optimizerConfig);

  void train();

 private:
  int correctly_labelled_num_, labelled_num_;

  net* net_;
  optimizer* weight_updater_;
  dataIterator* train_iter_;
  trainerConfig config_;
};

}

#endif
