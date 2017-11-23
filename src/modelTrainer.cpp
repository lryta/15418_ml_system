#include "modelTrainer.h"

namespace MLLib {

modelTrainer::modelTrainer(NNType type,
    std::vector<int> inter_dimes, trainerConfig config)
  :config_(config) {

  data_iter_ = new dataIterator(config_.batch_num_);

  switch (type) {
    case NNType::MLPNet:
      assert(inter_dims.size() > 0);
      net_ = new MLPNet(data_iter_.getDataShape(), inter_dims);
    default:
      throw "Network Type not recognized"
  }

  weight_updater_.register(net->getParams());
}

modelTrainer::~modelTrainer() {
  delete data_iter_;
  delete weight_updater_;
  delete net_;
}

void modelTrainer::train() {
  for (int i = 0; i < config_.epoch; ++i) {
    // Reset iterator at very beginning
    data_iter_.start();

    labelled_num_ = 0;
    correctly_labelled_num_ = 0;

    while (trainIter.hasNext()) {
      auto data = data_iter_.next();
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
