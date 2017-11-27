#include "application.h"
#include "modelTrainer.h"
#include "optimizer.h"

namespace TinyML {

runMLP::runMLP():train_config_(NULL), trainer_(NULL), opt_conf_(NULL) {
  // 5 iterations. 128 Batch size
  train_config_ = new trainerConfig(5, 128, "../datasets");
  // 1e-4 learning rate
  opt_config_ = new optimizerConfig(1e-4);
  trainer_ = new modelTrainer(train_config_);
  // Two layers
  trainer_->setModel(ModelType::MLPNet, {100, 50});
  trainer_->setOptimizer(opt_config_);
}

runMLP::run() {
  trainer_->train();
}

runMLP::~runMLP() {
  delete trainer_;
  delete opt_config_;
  delete train_config_;
}

}
