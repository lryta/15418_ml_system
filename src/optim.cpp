#include "optim.h"

namespace MLLib {

struct OptimizerConfig {
  bool use_monmentum;
  float lr; // learning rate
  float momentum;
  float weightDecay;
}

SGD::SGD(OptimizerConfig config) {
  config_ = config;
}

void SGD::registerParams(std::vector<Tensor*> params) {
  for (auto &param : params) {
    weights.push_back(param);

    if (config_.use_monmentum)
      velocity.push_back(zerosLike(param));
  }
}

// CR(Haoran): I suggest avoid using velocity at first
// if you don't have enough time to finish

// TODO (Larry): Use Matrix Op.
// Tensor class doesn't have arithmetic operation definitions.
void SGD::update() {
  for (int i = 0; i < weights.size(); ++i) {
    if (config_.use_monmentum)
      velocity[i] = config_.momentum * velocity[i]
        + config_.lr * weights[i]->grad;

    if (config_.weightDecay != 0) {
      // TODO: use decay as a .
      Tensor decay = config_.weightDecay * weights[i]->data;
      weights[i]->data -= decay;
    }
    // TODO: no learning rate here?
    // Please use the inplace_add_operation
    weights[i]->data -= weights[i]->grad;
  }
}

void SGD::reset() {
  for (auto &i : velocity) {
    i.setZeros();
  }
}

}
