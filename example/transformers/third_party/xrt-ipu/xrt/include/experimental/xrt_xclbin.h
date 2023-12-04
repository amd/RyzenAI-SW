/*
 * Copyright (C) 2020-2022 Xilinx, Inc
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef XRT_XCLBIN_H_
#define XRT_XCLBIN_H_

#include "xrt.h"
#include "xclbin.h"
#include "xrt/xrt_uuid.h"
#include "xrt/detail/pimpl.h"

#ifdef __cplusplus
# include <utility>
# include <vector>
# include <string>
#endif

/**
 * typedef xrtXclbinHandle - opaque xclbin handle
 */
typedef void* xrtXclbinHandle; // NOLINT

#ifdef __cplusplus
namespace xrt {

/*!
 * @class xclbin
 *
 * @brief
 * xrt::xclbin represents an xclbin and provides APIs to access meta data.
 *
 * @details
 * The xclbin object is constructed by the user from a file.
 *
 * When the xclbin object is constructed from a complete xclbin, then it
 * can be used by xrt::device to program the xclbin onto the device.
 *
 * **First-class objects and class navigation**
 *
 * All meta data is rooted at xrt::xclbin.
 *
 * \image{inline} html xclbin_navigation.png "xclbin navigation"
 *
 * From the xclbin object
 * xrt::xclbin::kernel or xrt::xclbin::ip objects can be constructed.
 *
 * The xrt:xclbin::kernel is a concept modelled only in the xclbin XML
 * metadata, it corresponds to a function that can be executed by one
 * or more compute units modelled by xrt::xclbin::ip objects.  An
 * xrt::xclbin::ip object corresponds to an entry in the xclbin
 * IP_LAYOUT section, so the xrt::xclbin::kernel object is just a
 * grouping of one or more of these.
 *
 * In some cases the kernel concept is not needed, thus
 * xrt::xclbin::ip objects corresponding to entries in the xclbin
 * IP_LAYOUT sections can be accessed directly.
 *
 * An xrt::xclbin::arg object corresponds to one or more entries in
 * the xclbin CONNECTIVITY section decorated with additional meta data
 * (offset, size, type, etc) from the XML section if available.  An
 * argument object represents a specific kernel or ip argument. If
 * the argument is a global buffer, then it may connect to one or more
 * memory objects.
 *
 * Finally the xrt::xclbin::mem object corresponds to an entry in the
 * MEM_TOPOLOGY section of the xclbin.
 */
class xclbin_impl;
class xclbin : public detail::pimpl<xclbin_impl>
{
public:
  /*!
   * @enum target_type
   *
   * @brief
   * Type of xclbin
   *
   * @details
   * See `xclbin.h`
   */
  enum class target_type { hw, sw_emu, hw_emu };

public:
  /*!
   * @class mem
   *
   * @brief
   * xrt::xclbin::mem represents a physical device memory bank
   *
   * @details
   * A memory object is constructed from an entry in the MEM_TOPOLOGY
   * section of an xclbin.
   */
  class mem_impl;
  class mem : public detail::pimpl<mem_impl>
  {
  public:
    /**
     * @enum memory_type - type of memory
     *
     * @details
     * See `xclbin.h`
     */
    enum class memory_type : uint8_t {
      ddr3                 = MEM_DDR3,
      ddr4                 = MEM_DDR4,
      dram                 = MEM_DRAM,
      streaming            = MEM_STREAMING,
      preallocated_global  = MEM_PREALLOCATED_GLOB,
      are                  = MEM_ARE, //Aurora
      hbm                  = MEM_HBM,
      bram                 = MEM_BRAM,
      uram                 = MEM_URAM,
      streaming_connection = MEM_STREAMING_CONNECTION,
      host                 = MEM_HOST
    };

  public:
    mem() = default;

    explicit
    mem(std::shared_ptr<mem_impl> handle)
      : detail::pimpl<mem_impl>(std::move(handle))
    {}

    /**
     * get_name() - Get tag name
     *
     * @return
     *   Memory tag name
     */
    XCL_DRIVER_DLLESPEC
    std::string
    get_tag() const;

    /**
     * get_base_address() - Get the base address of the memory bank
     *
     * @return
     *  Base address of the memory bank, or -1 for invalid base address
     */
    XCL_DRIVER_DLLESPEC
    uint64_t
    get_base_address() const;

    /**
     * get_size() - Get the size of the memory in KB
     *
     * @return
     *  Size of memory in KB, or -1 for invalid size
     */
    XCL_DRIVER_DLLESPEC
    uint64_t
    get_size_kb() const;

    /**
     * get_used() - Get used status of the memory
     *
     * @return
     *  True of this memory bank is used by kernels in the xclbin
     *  or false otherwise.
     *
     * A value of false indicates that no buffer can be allocated
     * in this memory bank.
     */
    XCL_DRIVER_DLLESPEC
    bool
    get_used() const;

    /**
     * get_type() - Get the type of the memory
     *
     * @return
     *  Memory type
     *
     */
    XCL_DRIVER_DLLESPEC
    memory_type
    get_type() const;

    /**
     * get_index() - Get the index of the memory
     *
     * @return
     *  Index of the memory within the memory topology
     *
     * The returned index can be used when allocating buffers using
     * \ref xrt::bo provided the memory bank is connected / used.
     */
    XCL_DRIVER_DLLESPEC
    int32_t
    get_index() const;
  };

  /*!
   * @class arg
   *
   * @brief
   * class arg - xrt::xclbin::arg represents a compute unit argument
   *
   * @details
   * The argument object constructed from the xclbin connectivity
   * section.  An argument is connected to a memory bank or a memory
   * group, which dictates where in device memory a global buffer
   * used with this kernel argument must be allocated.
   */
  class arg_impl;
  class arg : public detail::pimpl<arg_impl>
  {
  public:
    arg() = default;

    explicit
    arg(std::shared_ptr<arg_impl> handle)
      : detail::pimpl<arg_impl>(std::move(handle))
    {}

    /**
     * get_name() - Get argument name
     *
     * @return
     *  Name of argument.
     *
     */
    XCL_DRIVER_DLLESPEC
    std::string
    get_name() const;

    /**
     * get_mems() - Get list of device memories from xclbin.
     *
     * @return
     *  A list of xrt::xclbin::mem objects to which this argument
     *  is connected.
     */
    XCL_DRIVER_DLLESPEC
    std::vector<mem>
    get_mems() const;

    /**
     * get_port() - Get port name of this argument
     *
     * @return
     *  Port name
     */
    XCL_DRIVER_DLLESPEC
    std::string
    get_port() const;

    /**
     * get_size() - Argument size in bytes
     *
     * @return
     *   Argument size
     */
    XCL_DRIVER_DLLESPEC
    uint64_t
    get_size() const;

    /**
     * get_offset() - Argument offset
     *
     * @return
     *   Argument offset
     */
    XCL_DRIVER_DLLESPEC
    uint64_t
    get_offset() const;

    /**
     * get_host_type() - Get the argument host type
     *
     * @return
     *   Argument host type
     */
    XCL_DRIVER_DLLESPEC
    std::string
    get_host_type() const;

    /**
     * get_index() - Get the index of this argument
     *
     * @return
     *   Argument index
     */
    XCL_DRIVER_DLLESPEC
    size_t
    get_index() const;
  };

  /*!
   * @class ip
   *
   * @brief
   * xrt::xclbin::ip represents a IP in an xclbin.
   *
   * @details
   * The ip corresponds to an entry in the IP_LAYOUT section of the
   * xclbin.
   */
  class ip_impl;
  class ip : public detail::pimpl<ip_impl>
  {
  public:
    /**
     * @enum control_type -
     *
     * @details
     * See `xclbin.h`
     */
    enum class control_type : uint8_t { hs = 0, chain = 1, none = 2, fa = 5 };

  public:
    ip() = default;

    explicit
    ip(std::shared_ptr<ip_impl> handle)
      : detail::pimpl<ip_impl>(std::move(handle))
    {}

    /**
     * get_name() - Get name of IP
     *
     * @return
     *  IP name.
     */
    XCL_DRIVER_DLLESPEC
    std::string
    get_name() const;

    /**
     * get_control_type() - Get the IP control protocol
     *
     * @return
     *  Control type
     */
    XCL_DRIVER_DLLESPEC
    control_type
    get_control_type() const;

    /**
     * get_num_args() - Number of arguments
     *
     * @return
     *  Number of arguments for this IP per CONNECTIVITY section
     */
    XCL_DRIVER_DLLESPEC
    size_t
    get_num_args() const;

    /**
     * get_args() - Get list of IP arguments
     *
     * @return
     *  A list sorted of xclbin::arg sorted by argument indices
     *
     * An argument may have multiple memory connections
     */
    XCL_DRIVER_DLLESPEC
    std::vector<arg>
    get_args() const;

    /**
     * get_arg() - Get argument at index.
     *
     * @param index
     *  Index of argument
     * @return
     *  The argument a specified index
     *
     * The argument may have multiple memory connections
     */
    XCL_DRIVER_DLLESPEC
    arg
    get_arg(int32_t index) const;

    /**
     * get_base_address() - Get the base address of the cu
     *
     * @return
     *  The base address of the IP
     */
    XCL_DRIVER_DLLESPEC
    uint64_t
    get_base_address() const;

    /**
     * get_size() - Get the address range size of this IP.
     *
     * @return
     *  The size of this IP
     *
     * The address range is a property of the kernel and
     * as such only valid for for kernel compute units.
     *
     * For IPs that are not associated with a kernel, the
     * size return is 0.
     */
    XCL_DRIVER_DLLESPEC
    size_t
    get_size() const;
  };

  /*!
   * class kernel
   *
   * @brief
   * xrt::xclbin::kernel represents a kernel in an xclbin.
   *
   * @details
   * The kernel corresponds to an entry in the XML meta data section
   * of the xclbin combined with meta data from other xclbin sections.
   * The kernel object is implicitly constructed from the xclbin
   * object via APIs.
   */
  class kernel_impl;
  class kernel : public detail::pimpl<kernel_impl>
  {
  public:
    kernel() = default;

    explicit
    kernel(std::shared_ptr<kernel_impl> handle)
      : detail::pimpl<kernel_impl>(std::move(handle))
    {}

    /**
     * get_name() - Get kernel name
     *
     * @return
     *  The name of the kernel
     */
    XCL_DRIVER_DLLESPEC
    std::string
    get_name() const;

    /**
     * get_cus() - Get list of cu from kernel.
     *
     * @return
     *  A list of xrt::xclbin::ip objects corresponding the compute units
     *  for this kernel object.
     */
    XCL_DRIVER_DLLESPEC
    std::vector<ip>
    get_cus() const;

    /**
     * get_cus() - Get list of compute units that matches name
     *
     * @param name
     *  Name to match against, prefixed with kernel name
     * @return
     *  A list of xrt::xclbin::ip objects that are compute units
     *  of this kernel object and matches the specified name.
     *
     * The kernel name can optionally specify which kernel instance(s) to
     * match "kernel:{cu1,cu2,...} syntax.
     */
    XCL_DRIVER_DLLESPEC
    std::vector<ip>
    get_cus(const std::string& name) const;

    /**
     * get_cu() - Get compute unit by name
     *
     * @return
     *  The xct::xclbin::ip object matching the specified name, or error if
     *  not present.
     */
    XCL_DRIVER_DLLESPEC
    ip
    get_cu(const std::string& name) const;

    /**
     * get_num_args() - Number of arguments
     *
     * @return
     *  Number of arguments for this kernel.
     */
    XCL_DRIVER_DLLESPEC
    size_t
    get_num_args() const;

    /**
     * get_args() - Get list of kernel arguments
     *
     * @return
     *  A list sorted of xclbin::arg sorted by argument indices
     *
     * An argument may have multiple memory connections
     */
    XCL_DRIVER_DLLESPEC
    std::vector<arg>
    get_args() const;

    /**
     * get_arg() - Get kernel argument at index.
     *
     * @return
     *  The xrt::xclbin::arg object at specified argument index.
     *
     * The memory connections of an argument is the union of the
     * connections for each compute unit for that particular argument.
     * In other words, for each connection of the argument returned
     * by ``get_arg()`` there is at least one compute unit that has
     * that connection.
     */
    XCL_DRIVER_DLLESPEC
    arg
    get_arg(int32_t index) const;
  };

  /// @cond
  /** undocumented access to aie metadata, subject to change **/
  class aie_partition_impl;
  class aie_partition : detail::pimpl<aie_partition_impl>
  {
  public:
    explicit
    aie_partition(std::shared_ptr<aie_partition_impl> handle)
      : detail::pimpl<aie_partition_impl>(std::move(handle))
    {}

    XCL_DRIVER_DLLESPEC
    uint64_t
    get_inference_fingerprint() const;

    XCL_DRIVER_DLLESPEC
    uint64_t
    get_pre_post_fingerprint() const;

    XCL_DRIVER_DLLESPEC
    uint32_t
    get_operations_per_cycle() const;
  };
  /// @endcond

public:
  /**
   * xclbin() - Construct empty xclbin object
   */
  xclbin() = default;

  /// @cond
  /**
   * xclbin() - Construct from handle
   */
  explicit
  xclbin(std::shared_ptr<xclbin_impl> handle)
    : detail::pimpl<xclbin_impl>(handle)
  {}
  /// @endcond

  /**
   * xclbin() - Constructor from an xclbin filename
   *
   * @param filename
   *  Path to the xclbin file
   *
   * Throws if file not found.
   */
  XCL_DRIVER_DLLESPEC
  explicit
  xclbin(const std::string& filename);

  /**
   * xclbin() - Constructor from raw data
   *
   * @param data
   *  Raw data of xclbin
   *
   * The raw data of the xclbin can be deleted after calling the
   * constructor.
   */
  XCL_DRIVER_DLLESPEC
  explicit
  xclbin(const std::vector<char>& data);

  /**
   * xclbin() - Constructor from raw data
   *
   * @param top
   *  Raw data of xclbin file as axlf*
   *
   * The argument axlf is copied by the constructor.
   */
  XCL_DRIVER_DLLESPEC
  explicit
  xclbin(const axlf* top);

  /**
   * get_kernels() - Get list of kernels from xclbin.
   *
   * @return
   *  A list of xrt::xclbin::kernel from xclbin.
   *
   * Kernels are extracted from embedded XML metadata in the xclbin.
   * A kernel groups one or more compute units. A kernel has arguments
   * from which offset, type, etc can be retrived.
   */
  XCL_DRIVER_DLLESPEC
  std::vector<kernel>
  get_kernels() const;

  /**
   * get_kernel() - Get a kernel by name from xclbin
   *
   * @param name
   *  Name of kernel to get.
   * @return
   *  The matching kernel from the xclbin or error
   *  if no matching kernel is found.
   *
   * A matching kernel is extracted from embedded XML metadata in the
   * xclbin.  A kernel groups one or more compute units. A kernel has
   * arguments from which offset, type, etc can be retrived.
   */
  XCL_DRIVER_DLLESPEC
  kernel
  get_kernel(const std::string& name) const;

  /**
   * get_ips() - Get a list of IPs from the xclbin
   *
   * @return
   *  A list of xrt::xclbin::ip objects from xclbin.
   *
   * The returned xrt::xclbin::ip objects are extracted from the
   * IP_LAYOUT section of the xclbin.
   */
  XCL_DRIVER_DLLESPEC
  std::vector<ip>
  get_ips() const;

  /**
   * get_ip() - Get a specific IP from the xclbin
   *
   * @return
   *  A list of xrt::xclbin::ip objects from xclbin.
   *
   * The returned xrt::xclbin::ip object is extracted from the
   * IP_LAYOUT section of the xclbin.
   */
  XCL_DRIVER_DLLESPEC
  ip
  get_ip(const std::string& name) const;

  /**
   * get_mems() - Get list of memory objects
   *
   * @return
   * A list of xrt::xclbin::mem objects from xclbin
   *
   * The returned xrt::xclbin::mem objects are extracted from
   * the xclbin.
   */
  XCL_DRIVER_DLLESPEC
  std::vector<mem>
  get_mems() const;

  /// @cond
  XCL_DRIVER_DLLESPEC
  std::vector<aie_partition>
  get_aie_partitions() const;
  /// @endcond

  /**
   * get_xsa_name() - Get Xilinx Support Archive (XSA) name of xclbin
   *
   * @return
   *  Name of XSA (vbnv name).
   *
   * An exception is thrown if the data is missing.
   */
  XCL_DRIVER_DLLESPEC
  std::string
  get_xsa_name() const;

  /**
   * get_fpga_device_name() - Get FPGA device name
   *
   * @return
   *  Name of fpga device per XML metadata.
   */
  XCL_DRIVER_DLLESPEC
  std::string
  get_fpga_device_name() const;

  /**
   * get_uuid() - Get the uuid of the xclbin
   *
   * @return
   *  UUID of xclbin
   *
   * An exception is thrown if the data is missing.
   */
  XCL_DRIVER_DLLESPEC
  uuid
  get_uuid() const;

  /**
   * get_target_type() - Get the type of this xclbin
   *
   * @return
   *  Target type, which can be hw, sw_emu, or hw_emu
   */
  XCL_DRIVER_DLLESPEC
  target_type
  get_target_type() const;

  /// @cond
  /**
   * get_axlf() - Get the axlf data of the xclbin
   *
   * @return
   *  The axlf data of the xclbin object
   *
   * An exception is thrown if the data is missing.
   */
  XCL_DRIVER_DLLESPEC
  const axlf*
  get_axlf() const;

  /**
   * get_axlf_section() - Retrieve specified xclbin section
   *
   * @param section
   *  The section to retrieve
   * @return
   *  The specified section if available cast to specified type.
   *  Note, that this is an unsafe cast, behavior is undefined if the
   *  specified SectionType is invalid.
   *
   * The SectionType template parameter is an axlf type from xclbin.h
   * and it much match the type of the section data retrieved.
   *
   * Throws if requested section does not exist in the xclbin.
   */
  template <typename SectionType>
  SectionType
  get_axlf_section(axlf_section_kind section) const
  {
    return reinterpret_cast<SectionType>(get_axlf_section(section).first);
  }
  /// @endcond

private:
  XCL_DRIVER_DLLESPEC
  std::pair<const char*, size_t>
  get_axlf_section(axlf_section_kind section) const;
};
} // namespace xrt

/// @cond
extern "C" {
#endif

/**
 * xrtXclbinAllocFilename() - Allocate a xclbin using xclbin filename
 *
 * @filename:      path to the xclbin file
 * Return:         xrtXclbinHandle on success or NULL with errno set
 */
XCL_DRIVER_DLLESPEC
xrtXclbinHandle
xrtXclbinAllocFilename(const char* filename);


/**
 * xrtXclbinAllocAxlf() - Allocate a xclbin using an axlf
 *
 * @top_axlf:      an axlf
 * Return:         xrtXclbinHandle on success or NULL with errno set
 */
XCL_DRIVER_DLLESPEC
xrtXclbinHandle
xrtXclbinAllocAxlf(const struct axlf* top_axlf);

/**
 * xrtXclbinAllocRawData() - Allocate a xclbin using raw data
 *
 * @data:          raw data buffer of xclbin
 * @size:          size (in bytes) of raw data buffer of xclbin
 * Return:         xrtXclbinHandle on success or NULL with errno set
 */
XCL_DRIVER_DLLESPEC
xrtXclbinHandle
xrtXclbinAllocRawData(const char* data, int size);

/**
 * xrtXclbinFreeHandle() - Deallocate the xclbin handle
 *
 * @xhdl:          xclbin handle
 * Return:         0 on success, -1 on error
 */
XCL_DRIVER_DLLESPEC
int
xrtXclbinFreeHandle(xrtXclbinHandle xhdl);

/**
 * xrtXclbinGetXSAName() - Get Xilinx Support Archive (XSA) Name of xclbin handle
 *
 * @xhdl:       Xclbin handle
 * @name:       Return name of XSA.
 *              If the value is nullptr, the content of this value will not be populated.
 *              Otherwise, the the content of this value will be populated.
 * @size:       size (in bytes) of @name.
 * @ret_size:   Return size (in bytes) of XSA name.
 *              If the value is nullptr, the content of this value will not be populated.
 *              Otherwise, the the content of this value will be populated.
 * Return:  0 on success or appropriate error number
 */
XCL_DRIVER_DLLESPEC
int
xrtXclbinGetXSAName(xrtXclbinHandle xhdl, char* name, int size, int* ret_size);

/**
 * xrtXclbinGetUUID() - Get UUID of xclbin handle
 *
 * @xhdl:     Xclbin handle
 * @ret_uuid: Return xclbin id in this uuid_t struct
 * Return:    0 on success or appropriate error number
 */
XCL_DRIVER_DLLESPEC
int
xrtXclbinGetUUID(xrtXclbinHandle xhdl, xuid_t ret_uuid);

/**
 * xrtXclbinGetNumKernels() - Get number of PL kernels in xclbin
 *
 * @xhdl:   Xclbin handle obtained from an xrtXclbinAlloc function
 * Return:  The number of PL kernels in the xclbin
 *
 * Kernels are extracted from embedded XML metadata in the xclbin.
 * A kernel groups one or more compute units. A kernel has arguments
 * from which offset, type, etc can be retrived.
 */
XCL_DRIVER_DLLESPEC
size_t
xrtXclbinGetNumKernels(xrtXclbinHandle xhdl);

/**
 * xrtXclbinGetNumKernelComputeUnits() - Get number of CUs in xclbin
 *
 * @xhdl:   Xclbin handle obtained from an xrtXclbinAlloc function
 * Return:  The number of compute units
 *
 * Compute units are associated with kernels.  This function returns
 * the total number of compute units as the sum of compute units over
 * all kernels.
 */
XCL_DRIVER_DLLESPEC
size_t
xrtXclbinGetNumKernelComputeUnits(xrtXclbinHandle xhdl);

/**
 * xrtXclbinGetData() - Get the raw data of the xclbin handle
 *
 * @xhdl:       Xclbin handle
 * @data:       Return raw data.
 *              If the value is nullptr, the content of this value will not be populated.
 *              Otherwise, the the content of this value will be populated.
 * @size:       Size (in bytes) of @data
 * @ret_size:   Return size (in bytes) of XSA name.
 *              If the value is nullptr, the content of this value will not be populated.
 *              Otherwise, the the content of this value will be populated.
 * Return:  0 on success or appropriate error number
 */
XCL_DRIVER_DLLESPEC
int
xrtXclbinGetData(xrtXclbinHandle xhdl, char* data, int size, int* ret_size);

/*
 * xrtGetXclbinUUID() - Get UUID of xclbin image running on device
 *
 * @dhdl:   Device handle
 * @out:    Return xclbin id in this uuid_t struct
 * Return:  0 on success or appropriate error number
 */
XCL_DRIVER_DLLESPEC
int
xrtXclbinUUID(xclDeviceHandle dhdl, xuid_t out);

/// @endcond
#ifdef __cplusplus
}
#endif

#endif
