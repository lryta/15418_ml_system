#include "modelTrainer.h"

namespace MLLib {

modelTrainer::modelTrainer(net n,
    , DataIterator *itrainIter,
  int epoch)
  :correctly_labelled_num_(0), labelled_num_(0),
  net_(n), weight_updater_(), trainIter(itrainIter),
  config_(epoch) {

  weight_updater_.register(net->getParams());
  trainIter.setBatchSize();
}

void modelTrainer::train() {
  for (int i = 0; i < config_.epoch; ++i) {
    // Reset iterator at very beginning
    trainIter.start();

    labelled_num_ = 0;
    correctly_labelled_num_ = 0;

    while (trainIter.hasNext()) {
      auto data = train_iter_.next();
      auto input = std::get<0>(data);
      auto target = std::get<1>(data);
      // forward
      net_.forwrd({input}, {target});
      correctly_labelled_num_ += net_->correctlyRecognizedDataNum();
      labelled_num_ += batchSize;
      printf("Total Loss is %f\n", net_->getLoss());

      // back prop and update weight
      net_->backward({input}, {target});
      weight_updater_->update();
    }

    printf("Total Accuracy in epoch %d is %f\n",
        i + 1, correctly_labelled_num_(float)/labelled_num_);
  }
}

}
