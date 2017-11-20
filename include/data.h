
class DataIterator {
  public:
    DataIterator(string fileName);
    DataIterator(Tensor dataBlock);
    Tensor nextBatch();
    int totalEntry();
    int curPos();
    void reset();
    std::tuple<Tensor*, Tensor*> nextBatch();
    bool batchEnd();
}


class MNISTIterator : DataIterator {
  public:
    MNISTIterator(string filename);
    MNISTIterator(Tensor dataBlock);
}
