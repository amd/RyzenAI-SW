/*
 * Copyright (C) 2015-2018, Xilinx Inc - All rights reserved
 * Xilinx SDAccel HAL userspace driver APIs
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

#include <stdint.h>

#ifndef _XRT_STREAM_H
#define _XRT_STREAM_H

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t stream_xfer_req_type;

/**
 * stream_opt_type
 *
 * For streaming queues, additional control can be set on a per-queue basis via
 * clSetStreamOpt(opt_type, opt_value)
 *
 * The following options are available:
 * - STREAM_OPT_AIO_MAX_EVENT: maximum # aio event
 *   this option creates a per-queue asynchronous i/o context with
 *   the "opt_value" as the maximum # concurrently i/o operations.
 *
 * The next 3 options will allow accumulation of the io requests before 
 * submitting to the kernel for processing. This accumulation will increase
 * the latency but may increase throughput.
 *
 * -  STREAM_OPT_AIO_BATCH_THRESH_BYTES,  io batching threshold: # bytes
 *    Keep accumulating the i/o request, until the total # of r/w bytes reaches
 *    the threshold of "opt_value" of bytes
 *
 * -  STREAM_OPT_AIO_BATCH_THRESH_PKTS,   io_batching threshold: # request
 *    Keep accumulating the i/o request, until the total # of r/w request
 *    reaches the threshold of "opt_value" of requests
 */
typedef enum stream_opt_type {
    STREAM_OPT_AIO_MAX_EVENT = 1,       /* maximum # aio event */
    STREAM_OPT_AIO_BATCH_THRESH_BYTES,  /* io batching threshold: # bytes */
    STREAM_OPT_AIO_BATCH_THRESH_PKTS,   /* io_batching threshold: # request */

    STREAM_OPT_MAX
} stream_opt_type;

/**
 * cl_stream_xfer_req.
 * For each read or write request, this extra data needs to be sent.
 */

typedef struct stream_xfer_req {
  stream_xfer_req_type flags;
  char*                cdh;
  uint32_t             cdh_len;
  void*                priv_data;
  uint32_t             timeout; //in ms
  char                 reserved[64];
} stream_xfer_req;

/**
 * struct cl_streams_poll_req_completions
 * For each poll completion provide one of this struct.
 * Keep this in sync with xclReqCompletion in xclhal2.h
 */
typedef struct streams_poll_req_completions {
  char                 resv[64]; /* reserved for meta data */
  void                 *priv_data;
  size_t               nbytes;
  int                  err_code;
} streams_poll_req_completions;

#ifdef __cplusplus
}
#endif

#endif
