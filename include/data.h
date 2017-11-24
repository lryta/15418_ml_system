#include <Tuple>
#include <assert>
#include <string>
#include <vector>
#include "tensor.h"

namespace MLLib {

class dataIterator {
 public:
  virtual dataIterator();
  virtual ~dataIterator();
  /* 
   * return tuple<data, target>
   */
  virtual std::tuple<tensor*, tensor*> *next();
  virtual bool getNext();
  virtual int shape();
  virtual void reset();
}

class MNISTIterator:dataIterator {
 public:
  MNISTIterator(std::string dataset_dir, int batch_num);

  int shape() {
    return shape_;
  }

  bool hasNext() {
    return pos_ < shuffled_pos_.size();
  }
  
  std::tuple<tensor*, tensor*> next() {
    assert(pos_ < shuffled_pos_.size())
    return data_with_target_[shuffled_pos_[pos_++]];
  }

 private:
  int reverseInt(int);
  int reverseInt();
  void initShuffleData();
  void readInt(std::ifstream &, int &);

  std::vector<std::tuple<tensor*, tensor*>> data_with_target_;
  std::vector<int> shuffled_pos_;

  int batch_num_, total_num_, shape_;
  int pos_;
}

}
