#pragma once
inline size_t compute_hash(char *data, size_t size) {
  // const char* data = reinterpret_cast<const char*>(v.data());
  // std::size_t size = v.size() * sizeof(v[0]);
  std::hash<std::string_view> hash;
  return hash(std::string_view(data, size));
}

inline void read_bin_file(std::string filename, char *data) {
  std::ifstream file(filename, std::ios::binary);

  // Check if the file is opened successfully
  if (!file.is_open()) {
    std::cerr << "Error opening file." << std::endl;
    // return 1;
  }

  // Get the file size
  file.seekg(0, std::ios::end);
  std::streampos fileSize = file.tellg();
  file.seekg(0, std::ios::beg);
  file.read(data, fileSize);
}

inline void write_bin_file(std::string filename, char *data, size_t size) {
  std::fstream file;
  file.open(filename, std::ios::out | std::ios::binary);
  file.write(data, size);
}

template <typename inT>
inline void read_data_file(std::string filename, inT *in_ptr) {
  std::fstream file;
  file.open(filename, std::ios::in);
  if (file.is_open()) {
    printf("Opened file: %s for read \n", filename.c_str());
    int count = 0;
    std::string line;
    while (file) {
      std::getline(file, line);
      std::istringstream ss(line);
      inT num;
      while (ss >> num) {
        *(in_ptr + count) = (inT)(num);
        if (count < 10) {
          std::cout << (inT)(*(in_ptr + count)) << std::endl;
        }
        count++;
      }
    }
    printf("Read %d values \n", count);
  } else {
    printf("Unable to open file: %s for read \n", filename.c_str());
    abort();
  }
}

template <typename T>
static void write_txt_file(std::string filename, T *data, size_t size_data) {
  std::ofstream myfile;

  myfile.open(filename);

  for (int i = 0; i < size_data; ++i) {
    myfile << std::to_string((T)data[i]) << std::endl;
  }

  myfile.close();
}

template <typename T>
static void write32BitHexTxtFile(std::string filename, T *data,
                                 size_t size_data) {
  std::ofstream myfile;
  myfile.open(filename);

  std::size_t element_size = sizeof(T);
  switch (sizeof(T)) {
  case 1:
    for (int i = 0; i < size_data; i += 4) {
      myfile << std::hex << std::setw(2) << std::setfill('0')
             << static_cast<int>(data[i + 3]) << std::setw(2)
             << static_cast<int>(data[i + 2]) << std::setw(2)
             << static_cast<int>(data[i + 1]) << std::setw(2)
             << static_cast<int>(data[i]) << std::endl;
    }
    break;

  case 2:
    for (int i = 0; i < size_data; i += 2) {
      myfile << std::hex << std::setw(4) << std::setfill('0')
             << static_cast<int>(data[i + 1]) << std::setw(4)
             << static_cast<int>(data[i]) << std::endl;
    }
    break;

  default:
    for (int i = 0; i < size_data; i++) {
      myfile << std::hex << std::setw(2) << std::setfill('0')
             << static_cast<int>(data[i + 1]) << std::endl;
    }
    break;
  }
  myfile.close();
}

template <typename T>
static void readTxtFileHex(std::string filename, T *data, size_t size_data) {
  std::ifstream myfile;
  myfile.open(filename);

  std::size_t element_size = sizeof(T);

  std::string line;
  int data_index = 0;

  while (std::getline(myfile, line) && data_index < size_data) {
    std::istringstream iss(line);
    uint32_t value;
    iss >> std::hex >> value;

    // Handle different data types
    switch (element_size) {
    case 1: // 8-bit data
    {
      data[data_index++] = static_cast<T>(value & 0xFF);
      if (data_index < size_data)
        data[data_index++] = static_cast<T>((value >> 8) & 0xFF);
      if (data_index < size_data)
        data[data_index++] = static_cast<T>((value >> 16) & 0xFF);
      if (data_index < size_data)
        data[data_index++] = static_cast<T>((value >> 24) & 0xFF);
    } break;

    case 2: // 16-bit data
    {
      uint16_t value_low = static_cast<uint16_t>(value & 0xFFFF);
      uint16_t value_high = static_cast<uint16_t>((value >> 16) & 0xFFFF);
      data[data_index++] = static_cast<T>(value_low);
      if (data_index < size_data)
        data[data_index++] = static_cast<T>(value_high);
    } break;

    case 4: // 32-bit data
    {
      data[data_index++] = static_cast<T>(value);
    } break;
    }
  }

  myfile.close();
}
