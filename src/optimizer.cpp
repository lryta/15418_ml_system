#include <cmath>
#include "optimizer.h"
#include "env.h"

namespace TinyML {

SGDOptimizer::SGDOptimizer(optimizerConfig *config):optimizer(), config_(*config),
  weights_(0), velocity_(0)
  {}

void SGDOptimizer::registerParams(std::vector<tensor*> params) {
  for (auto &param : params) {
    weights_.push_back(param);

    //if (config_.use_monmentum_)
    //  velocity_.push_back(zerosLike(param));
  }
}

void SGDOptimizer::randomizeParams() {
  for (auto &weight : weights_) {
    auto w_shape = weight->getShape();
    if (w_shape.getDimNum() == 1)
      env::getInstance()->getWeightInit()->normalDistInit(weight, 0, 0.5);
    else
      env::getInstance()->getWeightInit()->normalDistInit(weight, 0, 1/(2*sqrt(w_shape.getDim(1))));
  }
}

// CR(Haoran): I suggest avoid using velocity at first
// if we don't have enough time
void SGDOptimizer::update() {
  for (size_t i = 0; i < weights_.size(); ++i) {
    /*
    TODO: Implement velocity
    if (config_.use_monmentum)
      velocity[i] = config_.momentum * velocity[i]
        + config_.lr * weights[i]->grad;

    TODO: Implement weight decay
    if (config_.weightDecay != 0) {
      tensor decay = config_.weightDecay * weights[i]->data;
      weights[i]->data -= decay;
    }
    */

    // Implace Update
    auto in_shape = weights_[i]->getShape();
    matrix::UpdateWeightWithReg(weights_[i]->getGrad(), weights_[i]->getData(),
        in_shape.getDim(1),
        (in_shape.getDimNum()>=2)?in_shape.getDim(2):1, config_.lr_, config_.reg_);
  }
}

void SGDOptimizer::reset() {
  //for (auto &i : velocity) {
  //  i.setZeros();
  //}
}

}
