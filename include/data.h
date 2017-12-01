#ifndef _TINYML_DATA_H
#define _TINYML_DATA_H

#include <tuple>
#include <cassert>
#include <string>
#include <vector>
#include "shape.h"
#include "tensor.h"

namespace TinyML {

class dataIterator {
 public:
  dataIterator();
  ~dataIterator();
  virtual std::tuple<tensor*, tensor*> next() = 0;
  virtual bool hasNext() = 0;
  virtual shape getDataShape() = 0;
  virtual shape getTargetShape() = 0;
  virtual int getDataDim() = 0;
  virtual int getTargetDim() = 0;
  virtual void reset() = 0;
};

class MNISTIterator: public dataIterator {
 public:
  MNISTIterator(std::string dataset_dir, int batch_num);
  ~MNISTIterator();

  virtual shape getDataShape() {
    // Make sure the data is read to memory
    assert(data_with_target_.size() > 0);
    return shape(batch_num_, in_dim_);
  }

  virtual shape getTargetShape() {
    // Make sure the data is read to memory
    assert(data_with_target_.size() > 0);
    return shape(batch_num_, out_dim_);
  }

  virtual bool hasNext() {
    return pos_ < shuffled_pos_.size();
  }
  
  virtual std::tuple<tensor*, tensor*> next() {
    assert(pos_ < shuffled_pos_.size());
    return data_with_target_[shuffled_pos_[pos_++]];
  }

  virtual int getDataDim() {
    return in_dim_;
  }

  virtual int getTargetDim() {
    return out_dim_;
  }

  void makeDataAndLabel(std::string dir);

  void preprocess();

  virtual void reset();

 private:
  int reverseInt(int);
  int reverseInt();
  void initShuffleData();
  void readInt(std::ifstream &, int &);

  std::vector<std::tuple<tensor*, tensor*>> data_with_target_;
  std::vector<int> shuffled_pos_;

  int batch_num_, total_num_, in_dim_, out_dim_;
  size_t pos_;
};

}

#endif
