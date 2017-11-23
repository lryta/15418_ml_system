#include <assert>
#include <vector>
#include "nn.h"

namespace MLLib {

NN::NN(NNType type, std::vector<int> inter_dims, appConfig config)
  :application(), config_(config),
  net_(NULL), train_data_iter_(new dataIterator(config.batch_num_)),
  solver_(NULL){

  switch (type) {
    case NNType::MLPNet:
      assert(inter_dims.size() > 0);
      net_ = new MLPNet(train_data_iter.getDataShape(), inter_dims);
    default:
      throw "Network Type not recognized"
  }

  solver_ = new solver(net_, train_data_iter_, config_.epoch_);
}

NN::~NN() {
  delete solver_;
  delete net_;
}

void NN::Run() {
  solver_.train();
}

}
