#include <Tuple>
#include <assert>
#include <string>
#include <vector>
#include "shape.h"
#include "tensor.h"

namespace TinyML {

class dataIterator {
 public:
  virtual dataIterator();
  virtual ~dataIterator();
  /* 
   * return tuple<data, target>
   */
  virtual std::tuple<tensor*, tensor*> *next();
  virtual bool getNext();
  virtual shape getDataShape();
  virtual shape getTargetShape();
  virtual int getDataDim();
  virtual int getTargetDim();
  virtual void reset();
}

class MNISTIterator:dataIterator {
 public:
  MNISTIterator(std::string dataset_dir, int batch_num);

  shape getDataShape() {
    // Make sure the data is read to memory
    assert(data_with_target_.size() > 0);
    return shape(batch_num_, in_dim_);
  }

  shape getTargetShape() {
    // Make sure the data is read to memory
    assert(data_with_target_.size() > 0);
    return shape(batch_num_, out_dim_);
  }

  bool hasNext() {
    return pos_ < shuffled_pos_.size();
  }
  
  std::tuple<tensor*, tensor*> next() {
    assert(pos_ < shuffled_pos_.size())
    return data_with_target_[shuffled_pos_[pos_++]];
  }

  int getDataDim() {
    return in_dim_;
  }

  int getTargetDim() {
    return target_dim_;
  }

 private:
  int reverseInt(int);
  int reverseInt();
  void initShuffleData();
  void readInt(std::ifstream &, int &);

  std::vector<std::tuple<tensor*, tensor*>> data_with_target_;
  std::vector<int> shuffled_pos_;

  int batch_num_, total_num_, in_dim_, out_dim_;
  int pos_;
}

}
