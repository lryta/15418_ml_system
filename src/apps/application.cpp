#include "application.h"
#include "modelTrainer.h"

namespace MLLib {

runMLP::runMLP():trainer_(NULL) {
  trainerConfig train_config(5, 128, "../datasets");
  trainer_ = new modelTrainer(train_config);
  trainer_->setModel(ModelType::MLPNet, {100, 50});

  optimizerConfig opt_config(1e-4);
  trainer_->setOptimizer(optimizerConfig);
}

runMLP::run() {
  trainer_->train();
}
runMLP::~runMLP() {
  delete trainer_;
}

}
