/******************************************************************************
* Copyright (C) 2023 - 2024 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <xaiengine.h>
#include <xaiefal/common/xaiefal-common.hpp>
#include <xaiefal/common/xaiefal-log.hpp>
#include <xaiefal/rsc/xaiefal-rsc-base.hpp>

namespace xaiefal {
	class XAieDevHandle;

	enum XAieRscBitmapType {
		XAIE_STATIC_RSC,
		XAIE_AVAIL_RSC,
	};

	/**
	 * @class XAieRscMgrBackend
	 * @brief Resource Manager Backend Base class
	 */
	class XAieRscMgrBackend {
	public:
		XAieRscMgrBackend() = delete;
		XAieRscMgrBackend(XAieDevHandle *DevHd):
			AieHd(DevHd) {}

		/**
		 * This function checks for resource availibility and will
		 * reserve the resources if available
		 *
		 * @param reference to a resource
		 *
		 * @return XAIE_OK for success, error code for failure
		 */
		virtual AieRC request(XAieUserRsc& RscReq) {
			(void)RscReq;
			Logger::log(LogLevel::ERROR) << __func__ <<
				" Not supported for base backend " << std::endl;
			return XAIE_ERR;
		}

		/**
		 * This function checks for resource availibility and will
		 * reserve the list resources if available
		 *
		 * @param reference to list of resource
		 *
		 * @return XAIE_OK for success, error code for failure
		 */
		AieRC request(std::vector<XAieUserRsc>& RscReq) {
			AieRC RC;
			for (uint32_t i = 0; i < RscReq.size(); i++) {
				RC = request(RscReq[i]);
				if (RC != XAIE_OK) {
					RscReq.resize(i);
					free(RscReq);
					return RC;
				}
			}
			return XAIE_OK;
		}

		/**
		 * This function checks for resource availibility of a specific
		 * resource and will reserve the resources if available.
		 *
		 * @param reference to a specific resource
		 *
		 * @return XAIE_OK for success, error code for failure
		 */
		virtual AieRC requestAllocated(XAieUserRsc& RscReq) {
			(void)RscReq;
			Logger::log(LogLevel::ERROR) << __func__ <<
				" Not supported for base backend " << std::endl;
			return XAIE_ERR;
		}

		/**
		 * This function checks for resource availibility of a list
		 * of specific resources and will reserve the resources if
		 * available.
		 *
		 * @param reference to list of resource
		 *
		 * @return XAIE_OK for success, error code for failure
		 */
		AieRC requestAllocated(std::vector<XAieUserRsc>& RscReq) {
			AieRC RC;
			for (uint32_t i = 0; i < RscReq.size(); i++) {
				RC = requestAllocated(RscReq[i]);
				if (RC != XAIE_OK) {
					RscReq.resize(i);
					free(RscReq);
					return RC;
				}
			}
			return XAIE_OK;
		}

		/**
		 * This function checks for resource availibility and will
		 * reserve the amount of resources needed contiguously
		 *
		 * @param reference to list of resource
		 *
		 * @return XAIE_OK for success, error code for failure
		 */
		virtual AieRC requestContiguous(std::vector<XAieUserRsc>& RscReq) {
			(void)RscReq;
			Logger::log(LogLevel::ERROR) << __func__ <<
				" Not supported for base backend " << std::endl;
			return XAIE_ERR;
		}

		/**
		 * This function checks for resource availibility of a
		 * broadcast resource and will reserve the resources if
		 * available.
		 *
		 * @param reference to list of resource
		 *
		 * @return XAIE_OK for success, error code for failure
		 */
		virtual AieRC requestBc(std::vector<XAieUserRsc>& RscReq, bool isBcAll) {
			(void)RscReq;
			(void)isBcAll;
			Logger::log(LogLevel::ERROR) << __func__ <<
				" Not supported for base backend " << std::endl;
			return XAIE_ERR;
		}

		/**
		 * This function releases the resource from runtime and static
		 * bitmaps.
		 *
		 * @param reference to resource
		 *
		 * @return XAIE_OK for success, error code for failure
		 */
		virtual AieRC release(XAieUserRsc& RscRel) {
			(void)RscRel;
			Logger::log(LogLevel::ERROR) << __func__ <<
				" Not supported for base backend " << std::endl;
			return XAIE_ERR;
		}

		/**
		 * This function releases the list of resources from
		 * runtime and static bitmaps
		 *
		 * @param reference to list of resources
		 *
		 * @return XAIE_OK for success, error code for failure
		 */
		AieRC release(std::vector<XAieUserRsc>& RscRel) {
			for (uint32_t i = 0; i < RscRel.size(); i++)
				release(RscRel[i]);
			return XAIE_OK;
		}

		/**
		 * This function frees the resource from runtime bitmap
		 *
		 * @param reference to resource
		 *
		 * @return XAIE_OK for success, error code for failure
		 */
		virtual AieRC free(XAieUserRsc& RscFree) {
			(void)RscFree;
			Logger::log(LogLevel::ERROR) << __func__ <<
				" Not supported for base backend " << std::endl;
			return XAIE_ERR;
		}

		/**
		 * This function frees the list of resources from
		 * runtime bitmap
		 *
		 * @param reference to list of resources
		 *
		 * @return XAIE_OK for success, error code for failure
		 */
		AieRC free(std::vector<XAieUserRsc>& RscFree) {
			for (uint32_t i = 0; i < RscFree.size(); i++)
				free(RscFree[i]);
			return XAIE_OK;
		}

		/**
		 * This function requests resource statistics of the
		 * statically allocated resources
		 *
		 * @param reference to list of stats requests
		 * @param Bitmap type request is for
		 *
		 * @return XAIE_OK for success, error code for failure
		 */
		virtual AieRC getRscStats(std::vector<XAieUserRscStat> &vStats,
				XAieRscBitmapType BType) {
			(void)vStats;
			(void)BType;
			Logger::log(LogLevel::ERROR) << __func__ <<
				" Not supported for base backend " << std::endl;
			return XAIE_ERR;
		}

		/**
		 * This function writes the runtime resource bitmaps to
		 * the given ofstream object. NumRscs is updated with number
		 * bitmaps written.
		 *
		 * @param Fs ofstream to write bitmaps too
		 * @param NumRscs is updated with number of bitmaps written
		 *
		 * @return XAIE_OK for success, error code for failure
		 */
		virtual AieRC writeRscBitmaps(std::ofstream &Fs, uint64_t &NumRscs) {
			(void)Fs;
			(void)NumRscs;
			Logger::log(LogLevel::ERROR) << __func__ <<
				" Not supported for selected backend " << std::endl;
			return XAIE_ERR;
		}

		/**
		 * This function reads the resources metadata and loads
		 * it into the static resource bitmaps.
		 *
		 * @param MetaData metadata to load static resource bitmaps
		 * @param NumBitmaps number of resource bitmaps in the metadata
		 *
		 * @return XAIE_OK for success, error code for failure
		 */
		virtual AieRC loadRscBitmaps(const char *MetaData, uint64_t NumBitmaps) {
			(void)MetaData;
			Logger::log(LogLevel::ERROR) << __func__ <<
				" Not supported for selected backend " << std::endl;
			return XAIE_ERR;
		}

		XAie_DevInst *dev() {
			return AieHd->dev();
		}
	protected:
		XAieDevHandle *AieHd;
	}; /* class XAieRscMgrBackend */
} /* namespace xaiefal */
