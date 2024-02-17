/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef __STATS_HPP_
#define __STATS_HPP_

#include <Psapi.h>
#include <iostream>
#include <vector>
#include <windows.h>

using namespace std::literals::chrono_literals;

namespace ryzenai {
namespace stats {
/* Structure to hold all memory related metrics */
struct MemInfo {
  uint64_t commit_memory;
};

/* Structure to hold CPU times from which utilization is computed */
struct ProcTimeInfo {
  uint64_t sys_kernel_time;
  uint64_t sys_user_time;
  uint64_t proc_kernel_time;
  uint64_t proc_user_time;
};

/* Measure CPU Load for the given process */
struct CPULoad {
  CPULoad(int processID) : process_id_(processID) {
    process_handle_ = OpenProcess(PROCESS_QUERY_INFORMATION, FALSE, processID);
    if (process_handle_ == NULL) {
      throw std::runtime_error("Couldn't open the process" +
                               std::to_string(processID));
    }

    // Initialize timers
    get_cpu_load();
  }

  ~CPULoad() { CloseHandle(process_handle_); }

  float get_cpu_load() {
    FILETIME ft_proc_create, ft_proc_exit;
    FILETIME ft_sys_idle;
    FILETIME ft_proc_kernel_curr, ft_proc_user_curr;
    FILETIME ft_sys_kernel_curr, ft_sys_user_curr;

    GetSystemTimes(&ft_sys_idle, &ft_sys_kernel_curr, &ft_sys_user_curr);
    GetProcessTimes(process_handle_, &ft_proc_create, &ft_proc_exit,
                    &ft_proc_kernel_curr, &ft_proc_user_curr);

    ProcTimeInfo curr = {convert_ft_to_uint64(ft_sys_kernel_curr),
                         convert_ft_to_uint64(ft_sys_user_curr),
                         convert_ft_to_uint64(ft_proc_kernel_curr),
                         convert_ft_to_uint64(ft_proc_user_curr)};

    float ratio = compute_load(prev_, curr);
    prev_ = curr;
    return ratio;
  }

  int get_pid() const { return process_id_; }

private:
  int process_id_;
  HANDLE process_handle_ = NULL;
  ProcTimeInfo prev_{0, 0, 0, 0};

  uint64_t convert_ft_to_uint64(FILETIME ft) {
    ULARGE_INTEGER u;
    u.LowPart = ft.dwLowDateTime;
    u.HighPart = ft.dwHighDateTime;
    return u.QuadPart;
  }

  float compute_load(const ProcTimeInfo &begin, const ProcTimeInfo &end) {
    uint64_t sys_kernel_time = end.sys_kernel_time - begin.sys_kernel_time;
    uint64_t sys_user_time = end.sys_user_time - begin.sys_user_time;
    uint64_t proc_kernel_time = end.proc_kernel_time - begin.proc_kernel_time;
    uint64_t proc_user_time = end.proc_user_time - begin.proc_user_time;

    auto sys_time = sys_kernel_time + sys_user_time;
    if (sys_time == 0)
      return 0;

    return static_cast<float>((proc_kernel_time + proc_user_time)) / sys_time;
  }
};

/* Get system level commit memory */
static MemInfo get_sys_commit_mem() {
  PERFORMANCE_INFORMATION perfInfo;
  GetPerformanceInfo(&perfInfo, sizeof(PERFORMACE_INFORMATION));
  MemInfo mem_info;
  mem_info.commit_memory = perfInfo.CommitTotal * perfInfo.PageSize;
  return mem_info;
}
} // namespace stats
} // namespace ryzenai

#endif // __STATS_HPP_