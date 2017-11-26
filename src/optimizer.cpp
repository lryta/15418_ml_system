#include "optim.h"

namespace TinyML {

SGDOptimizer::SGD(OptimizerConfig config):config_(config)
  {}

void SGDOptimizer::registerParams(std::vector<Tensor*> params) {
  for (auto &param : params) {
    weights.push_back(param);

    if (config_.use_monmentum)
      velocity.push_back(zerosLike(param));
  }
}

// CR(Haoran): I suggest avoid using velocity at first
// if we don't have enough time
void SGDOptimizer::update() {
  for (int i = 0; i < weights.size(); ++i) {
    /*
    TODO: Implement velocity
    if (config_.use_monmentum)
      velocity[i] = config_.momentum * velocity[i]
        + config_.lr * weights[i]->grad;

    TODO: Implement weight decay
    if (config_.weightDecay != 0) {
      Tensor decay = config_.weightDecay * weights[i]->data;
      weights[i]->data -= decay;
    }
    */

    // Implace Update
    linearOpInplace(weights[i]->data, weights[i]->grad, 1, -1);
  }
}

void SGDOptimizer::reset() {
  for (auto &i : velocity) {
    i.setZeros();
  }
}

}
