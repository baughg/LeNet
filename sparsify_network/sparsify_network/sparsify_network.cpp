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

bool expand(
  int8_t* p_data,
  int8_t* p_sparsity_map,
  uint32_t elements,
  int8_t* p_dense_data);

void write_to_file(std::vector<uint8_t> &data, std::string filename);
void write_to_file_u32(std::vector<uint32_t> &data, std::string filename);
void densify_weights_zxy(std::string root_dir);
void sparsify_weights_xyz(std::string root_dir);
void sparsify_input_xyz(std::string root_dir);
void sparsify_weights_zxy(std::string root_dir, std::string output_prefix);
void sparsify_weights_fc_zxy(std::string root_dir);

int main(int argc, char** argv)
{
  if (argc <= 1)
  {
    printf("sparsify_network [binary_directory]\n");
    exit(1);
  }

  int mode = 0;

  if (argc >= 3)
  {
    if (strcmp(argv[2], "dense") == 0)
      mode = 1;
    else if (strcmp(argv[2], "zxy") == 0)
      mode = 2;
    else if (strcmp(argv[2], "fc_zxy") == 0)
      mode = 3;
    else if (strcmp(argv[2], "input") == 0)
      mode = 4;
  }

  std::string root_dir(argv[1]);

  if (!mode)
    sparsify_weights_xyz(root_dir);
  else if(mode == 1)
    densify_weights_zxy(root_dir);
  else if(mode == 2)
    sparsify_weights_zxy(root_dir,"weight1_0");
  else if (mode == 3)
    sparsify_weights_fc_zxy(root_dir);
  if (mode == 4)
    sparsify_input_xyz(root_dir);

  return 0;
}

void densify_weights_zxy(std::string root_dir)
{
  std::vector<uint8_t> buffer;
  std::string output_prefix = "output_c";

  const static uint32_t SE_OUTPUT = 64;
  //const static uint32_t H = 28;
  //const static uint32_t W = 28;
  const static uint32_t H = 28;
  const static uint32_t W = 28;
  const static uint32_t Ho = 1;
  const static uint32_t Wo = 1;
#define OUTPUT_CHANNELS 120

  uint8_t output_data[16][16 * SE_OUTPUT];
  uint8_t output_sparse_map[16][2 * SE_OUTPUT];
  uint32_t output_data_se_address[H*W];
  uint32_t output_sparse_se_address[H*W];
  uint8_t output_data_dense[H*W][OUTPUT_CHANNELS];

  std::string activation_filename = root_dir;
  activation_filename.append(output_prefix + "_packed_data_i8.bin");

  std::string activation_sparsity_filename = root_dir;
  activation_sparsity_filename.append(output_prefix + "_sparsity_map_i8.bin");

  std::string activation_list_filename = root_dir;
  activation_list_filename.append(output_prefix + "_se_data_address_i8.bin");
 
  std::string activation_sparsity_list_filename = root_dir;
  activation_sparsity_list_filename.append(
    output_prefix + "_se_sparsity_address_i8.bin");

  {
    FILE* output_data_file = NULL;
    output_data_file = fopen(activation_filename.c_str(), "rb");
    fread(output_data, 1, sizeof(output_data), output_data_file);
    fclose(output_data_file);

    FILE* output_sparse_map_file = NULL;
    output_sparse_map_file = fopen(activation_sparsity_filename.c_str(), "rb");
    fread(output_sparse_map, 1, sizeof(output_sparse_map), output_sparse_map_file);
    fclose(output_sparse_map_file);

    FILE* addr_file = NULL;
    addr_file = fopen(activation_list_filename.c_str(), "rb");
    fread(output_data_se_address, 1, sizeof(output_data_se_address), addr_file);
    fclose(addr_file);
    
    addr_file = fopen(activation_sparsity_list_filename.c_str(), "rb");
    fread(output_sparse_se_address, 1, sizeof(output_sparse_se_address), addr_file);
    fclose(addr_file);
  }

  const uint32_t storage_elements = Ho*Wo;
  uint32_t min_se_data_addr = output_data_se_address[0];
  uint32_t min_se_sparse_addr = output_sparse_se_address[0];

  for (uint32_t se = 1; se < storage_elements; ++se)
  {
    if (output_data_se_address[se] < min_se_data_addr)
      min_se_data_addr = output_data_se_address[se];

    if (output_sparse_se_address[se] < min_se_sparse_addr)
      min_se_sparse_addr = output_sparse_se_address[se];
  }

  uint8_t* p_se_data = &output_data[0][0];
  uint8_t* p_se_sparse_map = &output_sparse_map[0][0];
  uint32_t data_offset = 0;
  uint32_t sparse_offset = 0;
  int8_t* p_data, *p_sparse_map;

  for (uint32_t se = 0; se < storage_elements; ++se)
  {
    data_offset = output_data_se_address[se] - min_se_data_addr;
    sparse_offset = output_sparse_se_address[se] - min_se_sparse_addr;
    p_data = (int8_t*)p_se_data + data_offset;
    p_sparse_map = (int8_t*)p_se_sparse_map + sparse_offset;

    expand(p_data, p_sparse_map, OUTPUT_CHANNELS, (int8_t*)output_data_dense[se]);
  }

  std::string output_dense_filename = root_dir;
  output_dense_filename.append(
    output_prefix + "_dense_i8.bin");

  FILE* dense_data_file = NULL;
  dense_data_file = fopen(output_dense_filename.c_str(), "wb");
  fwrite(output_data_dense, 1, sizeof(output_data_dense), dense_data_file);
  fclose(dense_data_file);
}

bool expand(
  int8_t* p_data, 
  int8_t* p_sparsity_map, 
  uint32_t elements,
  int8_t* p_dense_data)
{
  if (!p_data || !p_sparsity_map)
    return false;



  int8_t sparse_mask = 0;
  int32_t sparse_byte_index = 0;
  int32_t sparse_bit_index = 0;

  for (int32_t elem = 0; elem < elements; ++elem)
  {
    p_dense_data[elem] = 0;
    sparse_byte_index = elem >> 3;
    sparse_bit_index = elem % 8;
    sparse_mask = p_sparsity_map[sparse_byte_index];
    sparse_mask >>= sparse_bit_index;
    sparse_mask &= 0x1;
    p_dense_data[elem] |= (*p_data * sparse_mask);
    p_data += sparse_mask;
  }

  return true;
}

void sparsify_weights_xyz(std::string root_dir)
{
  std::vector<uint8_t> buffer;

  uint32_t X = 25;
  uint32_t Y = 6;
  
  std::string binary_filename = root_dir;
  std::string output_prefix = "bias5_6";

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

void sparsify_weights_zxy(std::string root_dir, std::string output_prefix)
{
  std::vector<uint8_t> buffer;
  std::vector<uint8_t> buffer_zxy;

  uint32_t X = 25;
  uint32_t Y = 6;

  std::string binary_filename = root_dir;
  //std::string output_prefix = "weight5_6";

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

    buffer_zxy.resize(buffer.size());
    uint8_t* p_buffer = &buffer[0];
    uint8_t* p_weight_set = p_buffer;
    uint8_t* p_buffer_zxy = &buffer_zxy[0];
    uint8_t* p_weight_set_zxy = p_buffer_zxy;
    uint32_t channel_size = wght_header.X * wght_header.Y;
    uint32_t channels = wght_header.weights;
    uint32_t weight_set_size = channel_size * channels;
    uint32_t weight_sets = wght_header.Z;
    uint32_t channel_set_size = channel_size * weight_sets;    
    uint32_t offset = 0;

    for (uint32_t ws = 0; ws < weight_sets; ++ws)
    {
      p_weight_set_zxy = p_buffer_zxy + ws * weight_set_size;

      for (uint32_t c = 0; c < channels; ++c) {
        p_weight_set = p_buffer + ws * channel_size + c * channel_set_size;        

        for (uint32_t e = 0; e < channel_size; ++e)
        {
          offset = e * channels + c;
          p_weight_set_zxy[offset] = p_weight_set[e];
        }

        p_weight_set += channel_size;
      }
    }

    build_storage_elements_XYZ(
      buffer_zxy,
      channels,
      channel_size,
      weight_sets,
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
}

void sparsify_weights_fc_zxy(std::string root_dir)
{
  std::vector<uint8_t> buffer;
  std::vector<uint8_t> buffer_zxy;

  uint32_t X = 25;
  uint32_t Y = 6;

  std::string binary_filename = root_dir;
  std::string output_prefix = "weight5_6";

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

    buffer_zxy.resize(buffer.size());
    uint8_t* p_buffer = &buffer[0];
    uint8_t* p_weight_set = p_buffer;
    uint8_t* p_buffer_zxy = &buffer_zxy[0];
    uint8_t* p_weight_set_zxy = p_buffer_zxy;
    uint32_t channel_size = wght_header.weights * wght_header.Z;
    uint32_t channels = wght_header.Y;
    uint32_t weight_set_size = channel_size * channels;
    uint32_t weight_sets = wght_header.X;
    uint32_t channel_set_size = channel_size * weight_sets;
    uint32_t offset = 0;

    /*for (uint32_t ws = 0; ws < weight_sets; ++ws)
    {
      p_weight_set_zxy = p_buffer_zxy + ws * weight_set_size;

      for (uint32_t c = 0; c < channels; ++c) {
        p_weight_set = p_buffer + ws * channel_size + c * channel_set_size;

        for (uint32_t e = 0; e < channel_size; ++e)
        {
          offset = e * channels + c;
          p_weight_set_zxy[offset] = p_weight_set[e];
        }

        p_weight_set += channel_size;
      }
    }*/

    build_storage_elements_XYZ(
      buffer,
      channels,
      channel_size,
      weight_sets,
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
}


void sparsify_input_xyz(std::string root_dir)
{
  std::vector<uint8_t> buffer;

  uint32_t X = 28;
  uint32_t Y = 28;

  std::string binary_filename = root_dir;
  std::string output_prefix = "input";

  binary_filename.append(output_prefix + "_i8.bin");
  buffer.resize(X * Y);
  FILE* input_file = NULL;
  input_file = fopen(binary_filename.c_str(), "rb");


  unsigned points = 0;

  if (input_file) {    
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
      1,
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
}