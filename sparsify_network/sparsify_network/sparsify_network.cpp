// sparsify_network.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>

void sparsify(
  uint8_t* p_data,
  uint32_t len,
  std::vector<uint8_t> &data,
  std::vector<uint8_t> &sparsity_map);

void build_storage_elements_XYZ(
  std::vector<uint8_t> &data,
  uint32_t X,
  uint32_t Y,
  uint32_t Z,
  std::vector<uint8_t> &packed_data,
  std::vector<uint8_t> &sparsity_map);


int main(int argc, char** argv)
{
  if (argc <= 1)
  {
    printf("sparsify_network [binary_directory]\n");
    exit(1);
  }

  std::vector<uint8_t> buffer;

  uint32_t X = 28;
  uint32_t Y = 28;
  std::string root_dir(argv[1]);
  std::string binary_filename = root_dir;
  binary_filename.append("input_i8.bin");
  buffer.resize(X * Y);
  FILE* input_file = NULL;
  input_file = fopen(binary_filename.c_str(), "rb");

  if (input_file) {
    fread(&buffer[0], 1, buffer.size(), input_file);
    fclose(input_file);
    std::vector<uint8_t> packed_data;
    std::vector<uint8_t> sparsity_map;
    build_storage_elements_XYZ(buffer, X, Y, 1, packed_data, sparsity_map);

    if (packed_data.size())
    {
      printf("non zero\n");
    }
  }
  return 0;
}


void sparsify(
  uint8_t* p_data,
  uint32_t len,
  std::vector<uint8_t> &data,
  std::vector<uint8_t> &sparsity_map)
{
  data.clear();
  sparsity_map.clear();
  data.reserve(len);
  uint32_t sparse_mask_bytes = len >> 3;
  
  if (len > (sparse_mask_bytes << 3))
    sparse_mask_bytes++;

  
  sparsity_map.resize(sparse_mask_bytes);
  uint32_t byte_index = 0;
  uint32_t bit_index = 0;

  for (uint32_t e = 0; e < len; ++e)
  {
    byte_index = e >> 3;
    bit_index = e % 8;

    if (p_data[e])
    {
      data.push_back(p_data[e]);
      sparsity_map[byte_index] |= (1 << bit_index);
    }
  }
}


void build_storage_elements_XYZ(
  std::vector<uint8_t> &data,
  uint32_t X,
  uint32_t Y,
  uint32_t Z,
  std::vector<uint8_t> &packed_data,
  std::vector<uint8_t> &sparsity_map)
{
  std::vector<uint8_t> tensor_packed_data;
  std::vector<uint8_t> tensor_sparsity_map;

  uint8_t* p_data = &data[0];

  for (uint32_t z = 0; z < Z; ++z)
    for (uint32_t y = 0; y < Y; ++y)
    {
      sparsify(p_data, X, tensor_packed_data, tensor_sparsity_map);
      p_data += X;
    }
}