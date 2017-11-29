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
    epoch_num_(e), batch_num_(b), dataset_dir_(dir) {}

  int epoch_num_;
  int batch_num_;
  std::string dataset_dir_;
};

//TODO: Not sure we want to support test()
class modelTrainer {
 public:
  modelTrainer(trainerConfig*);
  ~modelTrainer();

  void setModel(ModelType, std::vector<size_t>);
  void setOptimizer(optimizerConfig);

  void train();

 private:
  int correctly_labelled_num_, labelled_num_;

  net* net_;
  optimizer* weight_updater_;
  dataIterator* data_iter_;
  trainerConfig config_;
};

}

#endif
