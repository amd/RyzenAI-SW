/******************************************************************************
* Copyright (C) 2023 - 2024 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#pragma once

#include <fstream>
#include <memory>
#include <vector>
#include <xaiengine.h>


#define BACKEND	XAieRscMgrGeneric
#include <xaiefal/rsc/xaiefal-rscmgr-generic.hpp>

#define XAIE_ERROR_BCAST_ID	0U
#define XAIE_ECC_BCAST_ID	6U
#define XAIE_ECC_PERFCNT_ID 	0U

namespace xaiefal {
	/**
	 * @class XAieRscMgr
	 * @brief Resource Manager class
	 */
	class XAieRscMgr {
	public:
		XAieRscMgr() = delete;
		XAieRscMgr(XAieDevHandle *DevHd):
			AieHd(DevHd) {
			if (!DevHd)
				throw std::invalid_argument("rscmgr: empty device handle");
			Backend = std::make_shared<BACKEND>(DevHd);
		}
		~XAieRscMgr() {}

		/**
		 * This function makes a call to RscMgr Backend to request
		 * a resource
		 *
		 * @param Rsc reference to any resource class inherited
		 * from XAieRsc
		 *
		 * @return XAIE_OK for success, error code for failure
		 */
		AieRC request(XAieRsc& Rsc) {
			AieRC RC = XAIE_OK;
			std::vector<XAieUserRsc> vRequests;

			switch(Rsc.getRscType()) {
			case XAIE_PERFCOUNT:
			case XAIE_USEREVENT:
			case XAIE_TRACECTRL:
			case XAIE_PCEVENT:
			case XAIE_SSEVENT:
			{
				RC = Rsc.getRscs(vRequests);
				if (RC != XAIE_OK) {
					return RC;
				}

				if (Rsc.getPreferredId() != XAIE_RSC_ID_ANY) {
					RC = Backend->requestAllocated(vRequests);
				} else if ((Rsc.getRscType() == XAIE_PCEVENT) &&
						(vRequests.size() > 1)) {
					/* PCRange */
					RC = Backend->requestContiguous(vRequests);
				} else {
					RC = Backend->request(vRequests);
				}

				if (RC == XAIE_OK) {
					RC = Rsc.setRscs(vRequests);
				}
				break;
			}
			case XAIE_BROADCAST:
			{
				uint32_t NumRscs;
				bool BcastAll;

				RC = Rsc.getRscs(vRequests);
				if (RC != XAIE_OK) {
					return RC;
				}

				BcastAll = false;
				if (vRequests.size() == 0) {
					vRequests.resize(1);
					BcastAll = true;
				}

				vRequests[0].RscId = Rsc.getPreferredId();
				RC = Backend->requestBc(vRequests, BcastAll);
				if (RC == XAIE_OK) {
					RC = Rsc.setRscs(vRequests);
				}
				break;
			}
			case XAIE_COMBOEVENT:
			{
				RC = Rsc.getRscs(vRequests);
				if (RC != XAIE_OK) {
					return RC;
				}

				RC = Backend->requestContiguous(vRequests);

				if (RC == XAIE_OK) {
					RC = Rsc.setRscs(vRequests);
				}
				break;
			}
			case XAIE_GROUPEVENT:
			{
				RC = Rsc.getRscs(vRequests);
				if (RC != XAIE_OK) {
					return RC;
				}

				RC = Backend->requestAllocated(vRequests);
				if (RC == XAIE_OK) {
					RC = Rsc.setRscs(vRequests);
				}
				break;
			}
			default:
				return XAIE_ERR;
			}
			return RC;
		}

		/**
		 * This function makes a call to RscMgr Backend to release
		 * a resource
		 *
		 * @param Rsc reference to any resource class inherited
		 * from XAieRsc
		 *
		 * @return XAIE_OK for success, error code for failure
		 *
		 * @note Will remove resource from runtime and static bitmaps
		 */
		AieRC release(XAieRsc& Rsc) {
			AieRC RC = XAIE_OK;
			std::vector<XAieUserRsc> vReleases;

			if (Rsc.getRscType() >= XAIE_MAXRSC) {
				return XAIE_ERR;
			}

			RC = Rsc.getRscs(vReleases);
			if (RC != XAIE_OK) {
				return RC;
			}

			return Backend->release(vReleases);
		}

		/**
		 * This function makes a call to RscMgr Backend to free
		 * a resource
		 *
		 * @param Rsc reference to any resource class inherited
		 * from XAieRsc
		 *
		 * @return XAIE_OK for success, error code for failure
		 *
		 * @note Will only remove resource from runtime bitmaps,
		 * 	 does not affect static bitmaps
		 */
		AieRC free(XAieRsc& Rsc) {
			AieRC RC = XAIE_OK;
			std::vector<XAieUserRsc> vReleases;

			if (Rsc.getRscType() >= XAIE_MAXRSC) {
				return XAIE_ERR;
			}

			RC = Rsc.getRscs(vReleases);
			if (RC != XAIE_OK) {
				return RC;
			}

			return Backend->free(vReleases);
		}

		/**
		 * This function requests resource statistics of the
		 * statically allocated resources
		 *
		 * @return XAIE_OK for success, error code for failure
		 */
		AieRC getStaticRscs(std::vector<XAieUserRscStat> &vStats) {
			return Backend->getRscStats(vStats, XAIE_STATIC_RSC);
		}

		/**
		 * This function requests resource statistics of the
		 * available resources (will check static and runtime
		 * bitmaps).
		 *
		 * @return XAIE_OK for success, error code for failure
		 */
		AieRC getAvailRscs(std::vector<XAieUserRscStat> &vStats) {
			return Backend->getRscStats(vStats, XAIE_AVAIL_RSC);
		}

		/**
		 * This function saves the runtime allocated resources to
		 * the given file name.
		 *
		 * @return XAIE_OK for success, error code for failure
		 */
		AieRC saveAllocatedRscs(const std::string &File) {
			uint64_t FirstBitmapOffset = 0x10;
			uint64_t NumRscsInFile = 0;
			std::ofstream Fs;
			AieRC RC;

			Fs.open(File);
			if (!Fs.is_open()){
				Logger::log(LogLevel::ERROR) <<
					"Could not open file " << File
					<< std::endl;
				return XAIE_ERR;
			}

			Fs.write(reinterpret_cast<char *>(&NumRscsInFile),
					sizeof(NumRscsInFile));
			Fs.write(reinterpret_cast<char *>(&FirstBitmapOffset),
					sizeof(FirstBitmapOffset));
			if (Fs.fail()) {
				Logger::log(LogLevel::ERROR) <<
					"Failed to write rscs to file " << File
					<< std::endl;
				Fs.close();
				return XAIE_ERR;
			}

			RC = Backend->writeRscBitmaps(Fs, NumRscsInFile);
			if (RC != XAIE_OK) {
				Logger::log(LogLevel::ERROR) <<
					"Failed to write rscs to file " << File
					<< std::endl;
				Fs.close();
				return RC;
			}

			Fs.seekp(std::ios_base::beg);
			Fs.write(reinterpret_cast<char *>(&NumRscsInFile),
					sizeof(NumRscsInFile));
			if (Fs.fail()) {
				Logger::log(LogLevel::ERROR) <<
					"Failed to write rscs to file " << File
					<< std::endl;
				Fs.close();
				return XAIE_ERR;
			}

			Fs.close();
			return XAIE_OK;
		}

		AieRC loadStaticRscs(const char *MetaData) {
			uint64_t *MetaHeader = (uint64_t *)MetaData;
			uint64_t NumBitmaps, FirstBitmapOffset;

			if (MetaHeader == NULL) {
				Logger::log(LogLevel::ERROR) <<
					"Invalid resource metadata" << std::endl;
				return XAIE_INVALID_ARGS;
			}

			NumBitmaps = MetaHeader[0U];
			FirstBitmapOffset = MetaHeader[1U];

			if (FirstBitmapOffset < (sizeof(uint64_t) * 2U) ||
				NumBitmaps == 0U) {
					Logger::log(LogLevel::ERROR) <<
						"Invalid metadata header"
						<< std::endl;
					return XAIE_INVALID_ARGS;
			}
			return Backend->loadRscBitmaps(MetaData + FirstBitmapOffset,
					NumBitmaps);
		}

		/**
		 * This function makes a call to RscMgr Backend to reserve
		 * resources necessary for ECC
		 *
		 * @return XAIE_OK for success, error code for failure
		 */
		AieRC reserveEcc() {
			std::vector<XAieUserRsc> vBcastRscs, vPerfRscs;
			std::vector<XAie_LocType> vLocs;
			uint32_t NumRscs, NumTiles;
			AieRC RC = XAIE_OK;

			NumRscs = dev()->AieTileNumRows * dev()->NumCols * 2
				+ (dev()->NumRows - dev()->AieTileNumRows)
				* dev()->NumCols;
			vBcastRscs.resize(NumRscs);

			vBcastRscs[0].RscId = XAIE_ECC_BCAST_ID;
			RC = Backend->requestBc(vBcastRscs, true);
			if (RC != XAIE_OK) {
				Logger::log(LogLevel::WARN) << "Unable to reserve " <<
					"broadcast resources for ECC" << std::endl;
				return RC;
			}

			NumTiles = dev()->NumCols * dev()->NumRows;
			vLocs.resize(NumTiles);
			RC = _XAie_GetUngatedLocsInPartition(dev(), &NumTiles, vLocs.data());
			if (RC != XAIE_OK) {
				Logger::log(LogLevel::WARN) << "Unable to reserve " <<
					"resources for ECC, get ungated tiles failed"
					<< std::endl;
				return RC;
			}
			vLocs.resize(NumTiles);

			for (auto loc : vLocs) {
				uint8_t TileType;
				XAieUserRsc Rsc;

				TileType = _XAie_GetTileTypefromLoc(dev(), loc);
				if (TileType == XAIEGBL_TILE_TYPE_AIETILE) {
					Rsc.Loc = loc;
					Rsc.Mod = XAIE_CORE_MOD;
					Rsc.RscType = XAIE_PERFCOUNT;
					Rsc.RscId = XAIE_ECC_PERFCNT_ID;

					RC = Backend->requestAllocated(Rsc);
					if (RC != XAIE_OK) {
						Logger::log(LogLevel::WARN) <<
							"Unable to reserve " <<
							"perfcounter resources for ECC"
							<< std::endl;
						Backend->release(vBcastRscs);
						Backend->release(vPerfRscs);
						return RC;
					}
					vPerfRscs.push_back(Rsc);
				}
			}
			return RC;
		}

		/**
		 * This function makes a call to RscMgr Backend to reserve
		 * resources necessary for Error Handling
		 *
		 * @return XAIE_OK for success, error code for failure
		 */
		AieRC reserveErrorHandling() {
			std::vector<XAieUserRsc> vRscs, vShimRscs;
			const XAie_L1IntrMod *L1IntrMod;
			uint32_t NumRscs;
			AieRC RC = XAIE_OK;

			NumRscs = dev()->AieTileNumRows * dev()->NumCols * 2
				+ (dev()->NumRows - dev()->AieTileNumRows)
				* dev()->NumCols;
			vRscs.resize(NumRscs);

			vRscs[0].RscId = XAIE_ERROR_BCAST_ID;
			RC = Backend->requestBc(vRscs, true);
			if (RC != XAIE_OK) {
				Logger::log(LogLevel::WARN) << "Unable to reserve " <<
					"broadcast resources for Error Handling"
					<< std::endl;
				return RC;
			}

			L1IntrMod = dev()->DevProp.DevMod[XAIEGBL_TILE_TYPE_SHIMPL].L1IntrMod;
			for (uint32_t i = 1; i < L1IntrMod->MaxErrorBcIdsRvd; i++) {
				for (uint8_t j = 0; j < dev()->NumCols; j++) {
					XAieUserRsc rsc;
					rsc.Loc = XAie_TileLoc(j, dev()->ShimRow);
					rsc.Mod = XAIE_PL_MOD;
					rsc.RscType = XAIE_BROADCAST;
					rsc.RscId = i;
					vShimRscs.push_back(rsc);
				}
			}
			RC = Backend->requestAllocated(vShimRscs);
			if (RC != XAIE_OK) {
				Logger::log(LogLevel::WARN) << "Unable to reserve " <<
					"shim broadcast resources for Error Handling"
					<< std::endl;
				return RC;
			}
			return XAIE_OK;
		}

		XAie_DevInst *dev() {
			return AieHd->dev();
		}

	protected:
		XAieDevHandle *AieHd; /**< AI engine device instance */
		std::shared_ptr<XAieRscMgrBackend> Backend; /**< Resource manager backend */
	}; /* class XAieRscMgr */

	/**
	 * This function saves all runtime allocated resources
	 * metadata to given filename
	 *
	 * @param File filename string
	 * @return XAIE_OK for success, error code for failure
	 */
	inline AieRC XAieDev::saveAllocatedRscsToFile(const std::string &File) {
		return AieHandle->rscMgr()->saveAllocatedRscs(File);
	}

	/**
	 * This loads resource metadata into the static resource
	 * bitmaps
	 *
	 * @param MetaData pointer to resource metadata
	 * @return XAIE_OK for success, error code for failure
	 */
	inline AieRC XAieDev::loadStaticRscsFromMem(const char *MetaData) {
		return AieHandle->rscMgr()->loadStaticRscs(MetaData);
	}

	/**
	 * This function makes a call to RscMgr Backend to reserve
	 * resources necessary for ECC
	 *
	 * @return XAIE_OK for success, error code for failure
	 */
	inline AieRC XAieDev::reserveEcc() {
		return AieHandle->rscMgr()->reserveEcc();
	}

	/**
	 * This function makes a call to RscMgr Backend to reserve
	 * resources necessary for Error Handling
	 *
	 * @return XAIE_OK for success, error code for failure
	 */
	inline AieRC XAieDev::reserveErrorHandling() {
		return AieHandle->rscMgr()->reserveErrorHandling();
	}
} /* namespace xaiefal */
