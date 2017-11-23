#include<layer_builder.h>

namespace MLLib {

Layer *LayerBuilder::makeLayer(layerBuildHelper helper,
    vector<shape> ins, vector<shape> ous) {
  switch (helper.type) {
    case LayerType::FullyConnected:
      return new FullyConnectedLayer(ins, ous, helper);
    case LayerType::Sigmoid:
      return new SigmoidLayer(ins, ous, helper);
    case LayerType::L2Loss:
      return new L2LossLayer(ins, ous, helper);
    default:
      throw "Unrecognized Layer Type";
  }
  return NULL;
}


}
