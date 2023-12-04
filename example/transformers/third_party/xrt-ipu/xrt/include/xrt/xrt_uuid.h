/**
 * Copyright (C) 2016-2021 Xilinx, Inc
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may
 * not use this file except in compliance with the License. A copy of the
 * License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#ifndef xrt_uuid_h_
#define xrt_uuid_h_

#ifdef _WIN32
# include "windows/uuid.h"
#else
# include <uuid/uuid.h>
typedef uuid_t xuid_t;
#endif

#ifdef __cplusplus
#include <string>

namespace xrt {

/*!
 * @class uuid
 * 
 * @brief
 * Wrapper class to treat uuid_t as a value type supporting copying
 *
 * @details
 * xrt::uuid is used by many XRT APIs to match an expected xclbin
 * against current device xclbin, or to get the uuid of current loaded
 * shell on the device.
 */
class uuid
{
  xuid_t m_uuid;
public:

  /**
   * uuid() - Construct cleared uuid
   */
  uuid()
  {
    uuid_clear(m_uuid);
  }

  /**
   * uuid() - Construct uuid from a basic bare uuid
   *
   * @param val
   *  The basic uuid to construct this object from
   *  
   * A basic uuid is either a uuid_t on Linux, or a typedef
   * of equivalent basic type of other platforms
   */
  uuid(const xuid_t val)
  {
    uuid_copy(m_uuid,val);
  }

  /**
   * uuid() - Construct uuid from a string representaition
   *
   * @param uuid_str
   *  A string formatted as a uuid string
   *
   * A uuid string is 36 bytes with '-' at 8, 13, 18, and 23
   */
  explicit uuid(const std::string& uuid_str)
  {
    if (uuid_str.empty()) {
      uuid_clear(m_uuid);
      return;
    }
    uuid_parse(uuid_str.c_str(), m_uuid);
  }

  /**
   * uuid() - copy constructor
   *
   * @param rhs
   *   Value to be copied
   */
  uuid(const uuid& rhs)
  {
    uuid_copy(m_uuid,rhs.m_uuid);
  }

  /// @cond
  uuid(uuid&&) = default;
  uuid& operator=(uuid&&) = default;
  /// @endcond

  /**
   * operator=() - assignment
   *
   * @param rhs
   *   Value to be assigned from
   * @return 
   *   Reference to this
   */
  uuid& operator=(const uuid& rhs)
  {
    uuid source(rhs);
    std::swap(*this,source);
    return *this;
  }

  /**
   * get() - Get the underlying basis uuid type value
   *
   * @return
   *  Basic uuid value
   *
   * A basic uuid is either a uuid_t on Linux, or a typedef
   * of equivalent basic type of other platforms
   */
  const xuid_t& get() const
  {
    return m_uuid;
  }

  /**
   * to_string() - Convert to string
   *
   * @return
   *   Lower case string representation of this uuid
   */
  std::string
  to_string() const
  {
    char str[40] = {0};
    uuid_unparse_lower(m_uuid,str);
    return str;
  }

  /**
   * bool() - Conversion operator
   *
   * @return
   *  True if this uuid is not null
   */
  operator bool() const
  {
    return uuid_is_null(m_uuid) == false;
  }

  /**
   * operator==() - Compare to basic uuid
   *
   * @param xuid
   *  Basic uuid to compare against
   * @return
   *  True if equal, false otherwise
   *
   * A basic uuid is either a uuid_t on Linux, or a typedef
   * of equivalent basic type of other platforms
   */
  bool
  operator == (const xuid_t& xuid) const
  {
    return uuid_compare(m_uuid, xuid) == 0;
  }

  /**
   * operator!=() - Compare to basic uuid
   *
   * @param xuid
   *  Basic uuid to compare against
   * @return
   *  False if equal, true otherwise
   *
   * A basic uuid is either a uuid_t on Linux, or a typedef
   * of equivalent basic type of other platforms
   */
  bool
  operator != (const xuid_t& xuid) const
  {
    return uuid_compare(m_uuid, xuid) != 0;
  }

  /**
   * operator==() - Comparison
   *
   * @param rhs
   *  uuid to compare against
   * @return
   *  True if equal, false otherwise
   */
  bool
  operator == (const uuid& rhs) const
  {
    return uuid_compare(m_uuid, rhs.m_uuid) == 0;
  }

  /**
   * operator!=() - Comparison
   *
   * @param rhs
   *  uuid to compare against
   * @return
   *  False if equal, true otherwise
   */
  bool
  operator != (const uuid& rhs) const
  {
    return uuid_compare(m_uuid, rhs.m_uuid) != 0;
  }

  /**
   * operator<() - Comparison
   *
   * @param rhs
   *  uuid to compare against
   * @return
   *  True of this is less that argument uuid, false otherwise
   */
  bool
  operator < (const uuid& rhs) const
  {
    return uuid_compare(m_uuid, rhs.m_uuid) < 0;
  }
};

} // xrt
#endif
#endif
