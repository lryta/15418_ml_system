#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

#include "data.h"

namespace TinyML {

using std::vector;
dataIterator::dataIterator() {}
dataIterator::~dataIterator() {}

int MNISTIterator::reverseInt(int v) {
  unsigned char ch1, ch2, ch3, ch4;
  int eight_bit_mask = ((int)1 << 8) - 1;

  ch1=v & eight_bit_mask;
  ch2=(v>>8) & eight_bit_mask;
  ch3=(v>>16) & eight_bit_mask;
  ch4=(v>>24) & eight_bit_mask;
  return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

void MNISTIterator::readInt(std::ifstream &file, int &v) {
  file.read((char*)&v, sizeof(int));
  v=reverseInt(v);
}

MNISTIterator::MNISTIterator(std::string dir, int batch_num):
  shuffled_pos_(0), batch_num_(batch_num), pos_(0) {
  makeDataAndLabel(dir);
  initShuffleData();
}

MNISTIterator::~MNISTIterator() {
  for (int i = 0; i < data_with_target_.size(); ++i) {
    delete std::get<0>(data_with_target_[i]);
    delete std::get<1>(data_with_target_[i]);
  }
}

void MNISTIterator::initShuffleData() {
  for (int i = 0; i < data_with_target_.size(); ++i)
    shuffled_pos_.push_back(i);
}

void MNISTIterator::makeDataAndLabel(std::string dir) {

  std::string data_path = dir + "/train-images-idx3-ubyte";
  std::string label_path = dir + "/train-labels-idx1-ubyte";
  
  std::ifstream data_file(data_path.c_str(), std::ios::binary);
  std::ifstream label_file(label_path.c_str(), std::ios::binary);

  if (data_file.is_open() && label_file.is_open()) {
    int magic_n, data_num, label_num;
    int row_num, col_num=0;
    readInt(data_file, magic_n);
    readInt(label_file, magic_n);

    readInt(data_file, data_num);
    readInt(label_file, label_num);
    assert(data_num == label_num);
    printf("Total data num is %d\n", data_num);
    total_num_ = data_num;

    readInt(data_file, row_num);
    readInt(data_file, col_num);
    in_dim_ = row_num * col_num;
    out_dim_ = 10;
    printf("in dim is %d\n", in_dim_);

    auto datas = new vector<vector<float>>(batch_num_,
        vector<float>(in_dim_, 0));
    auto labels = new vector<vector<float>>(batch_num_,
        vector<float>(out_dim_, 0));

    int batch_idx = 0;
    unsigned char value=0;

    for(int img_idx = 0; img_idx < data_num; ++img_idx) {
      for(int pixel_idx = 0 ; pixel_idx < in_dim_; ++pixel_idx) {
        data_file.read((char*)&value, sizeof(value));
        datas->at(batch_idx)[pixel_idx]= (float)value;
      }

      label_file.read((char*)&value, sizeof(value));
      assert(value >= 0 && value < out_dim_);
      std::fill(labels->at(batch_idx).begin(),
          labels->at(batch_idx).end(), 0);
      labels->at(batch_idx)[value] = 1;

      ++batch_idx;

      if (batch_idx == batch_num_) {
        auto data_tensor = new tensor(datas);
        auto label_tensor = new tensor(labels);
        data_with_target_.push_back(std::make_tuple(data_tensor, label_tensor));
        batch_idx = 0;
      }
    }

    delete datas;
    delete labels;
  } else
    throw "not valid inputs files for dataset";
}

void MNISTIterator::reset() {
  pos_ = 0;
  std::random_shuffle(shuffled_pos_.begin(), shuffled_pos_.end());
}

}
