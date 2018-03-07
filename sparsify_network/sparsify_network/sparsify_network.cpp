// sparsify_network.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>

typedef struct weight_header
{
  unsigned X;
  unsigned Y;
  unsigned Z;
  unsigned weights;
}weight_header;

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
  std::vector<uint8_t> &sparsity_map,
  std::vector<uint32_t> &storage_element_relative_address,
  std::vector<uint32_t> &sparse_map_relative_address);

void write_to_file(std::vector<uint8_t> &data, std::string filename);
void write_to_file_u32(std::vector<uint32_t> &data, std::string filename);

int main(int argc, char** argv)
{
  if (argc <= 1)
  {
    printf("sparsify_network [binary_directory]\n");
    exit(1);
  }

  std::vector<uint8_t> buffer;

  uint32_t X = 25;
  uint32_t Y = 6;
  std::string root_dir(argv[1]);
  std::string binary_filename = root_dir;
  std::string output_prefix = "weight0_1";

  binary_filename.append(output_prefix + "_i8.bin");
  buffer.resize(X * Y);
  FILE* input_file = NULL;
  input_file = fopen(binary_filename.c_str(), "rb");

  weight_header wght_header;
  unsigned points = 0;

  if (input_file) {
    fread(&wght_header, sizeof(wght_header), 1, input_file);
    X = wght_header.X * wght_header.Y;
    Y = wght_header.Z;
    points = wght_header.X * wght_header.Y * wght_header.Z;
    points *= wght_header.weights;

    buffer.resize(points);
    fread(&buffer[0], 1, buffer.size(), input_file);
    fclose(input_file);
    std::vector<uint8_t> packed_data;
    std::vector<uint8_t> sparsity_map;
    std::vector<uint32_t> storage_element_relative_address;
    std::vector<uint32_t> sparse_map_relative_address;

    build_storage_elements_XYZ(
      buffer, 
      X, 
      Y, 
      wght_header.weights,
      packed_data, 
      sparsity_map,
      storage_element_relative_address,
      sparse_map_relative_address);

    
    std::string filename = root_dir;
    filename.append(output_prefix + "_packed_data_i8.bin");
    write_to_file(packed_data, filename);
    filename = root_dir;
    filename.append(output_prefix + "_sparsity_map_i8.bin");
    write_to_file(sparsity_map, filename);
    filename = root_dir;
    filename.append(output_prefix + "_se_data_address_i8.bin");
    write_to_file_u32(storage_element_relative_address, filename);
    filename = root_dir;
    filename.append(output_prefix + "_se_sparsity_address_i8.bin");
    write_to_file_u32(sparse_map_relative_address, filename);
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
  std::vector<uint8_t> &sparsity_map,
  std::vector<uint32_t> &storage_element_relative_address,
  std::vector<uint32_t> &sparse_map_relative_address)
{
  std::vector<uint8_t> tensor_packed_data;
  std::vector<uint8_t> tensor_sparsity_map;
  
  const uint32_t tensor_count = Y*Z;

  uint32_t sparse_mask_bytes = X >> 3;

  if (X > (sparse_mask_bytes << 3))
    sparse_mask_bytes++;

  sparse_map_relative_address.resize(tensor_count);
  storage_element_relative_address.resize(tensor_count);
  sparsity_map.resize(tensor_count * sparse_mask_bytes);
  uint8_t* p_data = &data[0];
  uint32_t addr = 0;
  uint32_t sparse_addr = 0;
  uint32_t addr_non_zero = 0;
  uint32_t index = 0;
  uint32_t storage_element_size = X >> 4;

  if ((storage_element_size << 4) < X)
    storage_element_size++;

  storage_element_size <<= 4;

  packed_data.resize(tensor_count * storage_element_size);

  uint8_t* p_tensor_packed_data = &packed_data[0];
  uint8_t* p_tensor_sparsity_map = &sparsity_map[0];

  for (uint32_t z = 0; z < Z; ++z)
    for (uint32_t y = 0; y < Y; ++y)
    {
      sparsify(p_data, X, tensor_packed_data, tensor_sparsity_map);

      if (tensor_packed_data.size())
      {
        memcpy(p_tensor_packed_data, &tensor_packed_data[0], tensor_packed_data.size());
        p_tensor_packed_data += storage_element_size;
        addr = addr_non_zero;
        addr_non_zero += storage_element_size;
      }
      else
      {
        addr = ~0;
      }

      storage_element_relative_address[index] = addr;
      memcpy(p_tensor_sparsity_map, &tensor_sparsity_map[0], sparse_mask_bytes);
      p_tensor_sparsity_map += sparse_mask_bytes;
      sparse_map_relative_address[index++] = sparse_addr;
      sparse_addr += sparse_mask_bytes;
      p_data += X;
    }

  addr_non_zero += storage_element_size;
  packed_data.resize(addr_non_zero);
  addr_non_zero -= storage_element_size;

  for (uint32_t t = 0; t < tensor_count; ++t)
  {
    if (storage_element_relative_address[t] == ~0)
    {
      storage_element_relative_address[t] = addr_non_zero;
    }
  }
}

void write_to_file(std::vector<uint8_t> &data, std::string filename)
{
  FILE* file = NULL;

  file = fopen(filename.c_str(), "wb");

  if (file) {
    fwrite(&data[0], 1, data.size(), file);
    fclose(file);
  }
}

void write_to_file_u32(std::vector<uint32_t> &data, std::string filename)
{
  FILE* file = NULL;

  file = fopen(filename.c_str(), "wb");

  if (file) {
    fwrite(&data[0], sizeof(uint32_t), data.size(), file);
    fclose(file);
  }
}