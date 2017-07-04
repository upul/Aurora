/*!
 *  Copyright (c) 2017 by Contributors
 * \file cpu_device_api.cc
 */
#include "./cpu_device_api.h"
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace dlsys {
namespace runtime {

void *CPUDeviceAPI::AllocDataSpace(DLContext ctx, size_t size,
                                   size_t alignment) {
  // std::cout << "allocating cpu data" << std::endl;
  void *ptr;
  int ret = posix_memalign(&ptr, alignment, size);
  if (ret != 0)
    throw std::bad_alloc();
  return ptr;
}

void CPUDeviceAPI::FreeDataSpace(DLContext ctx, void *ptr) { free(ptr); }

void CPUDeviceAPI::CopyDataFromTo(const void *from, void *to, size_t size,
                                  DLContext ctx_from, DLContext ctx_to,
                                  DLStreamHandle stream) {
  // std::cout << "copying cpu data" << std::endl;
  memcpy(to, from, size);
}

void CPUDeviceAPI::StreamSync(DLContext ctx, DLStreamHandle stream) {}

} // namespace runtime
} // namespace dlsys
