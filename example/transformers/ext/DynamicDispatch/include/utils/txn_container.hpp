// Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.

#ifndef TRANSACTION_H
#define TRANSACTION_H

#include <map>
#include <mutex>
#include <string>
#include <vector>

class Transaction {
public:
  static Transaction &getInstance() {
    static Transaction instance;
    return instance;
  }
  Transaction(Transaction const &) = delete;
  void operator=(Transaction const &) = delete;
  std::string get_txn_str(std::string);
  Transaction();

  template <typename T>
  void GetBinData(const std::string &binaryData, std::vector<T> &outVector,
                  bool append = false) {
    const char *dataPtr = binaryData.data();
    size_t dataLen = binaryData.size();

    if (!append) {
      outVector.clear();
    }

    size_t currentSize = outVector.size();
    size_t newSize = dataLen / sizeof(T);
    size_t minSize = std::min(currentSize, newSize);

    for (size_t i = 0; i < minSize; ++i) {
      memcpy(&outVector[i], dataPtr + i * sizeof(T), sizeof(T));
    }

    for (size_t i = currentSize; i < newSize; ++i) {
      T value;
      memcpy(&value, dataPtr + i * sizeof(T), sizeof(T));
      outVector.push_back(value);
    }

    if (append && dataLen % sizeof(T) != 0) {
      T value = 0; // Zero-initialize the value
      memcpy(&value, dataPtr + newSize * sizeof(T), dataLen % sizeof(T));
      outVector.push_back(value);
    }
  }
};
#endif
