/******************************************************************************
* Copyright (C) 2023 - 2024 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#pragma once

#include <bitset>
#include <memory>
#include <vector>
#include <xaiengine.h>
#include <xaiefal/common/xaiefal-common.hpp>
#include <xaiefal/common/xaiefal-log.hpp>
#include <xaiefal/rsc/xaiefal-rscmgr-backend-base.hpp>

#define XAIE_TRACE_PER_MOD 1U
#define XAIE_COMBO_PER_MOD 4U

#define XAIE_RSC_HEADER_TTYPE_SHIFT	0U
#define XAIE_RSC_HEADER_TTYPE_MASK	0xF
#define XAIE_RSC_HEADER_MOD_SHIFT	4U
#define XAIE_RSC_HEADER_MOD_MASK	0xF
#define XAIE_RSC_HEADER_RTYPE_SHIFT	8U
#define XAIE_RSC_HEADER_RTYPE_MASK	0xFF
#define XAIE_RSC_HEADER_SIZE_SHIFT	16U
#define XAIE_RSC_HEADER_SIZE_MASK	0xFFFFFFFF

namespace xaiefal {
	/**
	 * @typedef XAieRscBitmap
	 * @brief Vector of 32 bit bitsets used to
	 * 	  represent resource bitmaps.
	 */
	typedef std::vector<std::bitset<32>> XAieRscBitmap;
	/**
	 * @typedef XAieRscMaxRsc
	 * @brief Type to hold max resources per module given tile type
	 */
	typedef std::vector<std::pair<XAie_ModuleType, uint32_t>> XAieRscMaxRsc;
	/**
	 * @struct AieRscMap
	 * @brief Structure containing resource bitmaps and
	 *	  relevant information.
	 */
	struct AieRscMap {
		XAieRscBitmap *Bitmaps[XAIE_MAXRSC];
		XAieRscMaxRsc MaxRscs[XAIE_MAXRSC];
	};
	/**
	 * @class XAieRscMgrGeneric
	 * @brief Resource Manager Backend Generic Class
	 */
	class XAieRscMgrGeneric : public XAieRscMgrBackend {
	public:
		XAieRscMgrGeneric() = delete;
		XAieRscMgrGeneric(XAieDevHandle *DevHd):
			XAieRscMgrBackend(DevHd) {
			std::string errStr = "Could not create resource manager";
			if (rscBitmapsInit() != XAIE_OK) {
				throw std::runtime_error(errStr);
			}
		}
		~XAieRscMgrGeneric() {
			for(uint8_t TType = 0U; TType < (uint8_t)XAIEGBL_TILE_TYPE_MAX; TType++) {
				if (TType == XAIEGBL_TILE_TYPE_SHIMNOC)
					continue;

				for (uint8_t RType = 0U; RType < (uint8_t)XAIE_MAXRSC; RType++) {
					auto MaxRsc = RscMaps[TType].MaxRscs[RType];
					uint32_t NumRscs = 0;

					for (auto x: MaxRsc)
						NumRscs += x.second;

					if (NumRscs == 0U)
						continue;

					delete RscMaps[TType].Bitmaps[RType];
				}
			}
		}

		/**
		 * This function checks for resource availibility and will
		 * reserve the resources if available
		 *
		 * @param reference to a resource
		 *
		 * @return XAIE_OK for success, error code for failure
		 */
		AieRC request(XAieUserRsc& RscReq) {
			uint8_t TileType = _XAie_GetTileTypefromLoc(dev(), RscReq.Loc);
			uint32_t StartBit, StaticOffset, MaxRscId;

			if ((_XAie_CheckModule(dev(), RscReq.Loc, RscReq.Mod) != XAIE_OK) ||
					TileType >= XAIEGBL_TILE_TYPE_MAX) {
				Logger::log(LogLevel::WARN) << __func__ <<
					" Invalid Location/Module for request" << std::endl;
				return XAIE_INVALID_ARGS;
			}

			StaticOffset = getStaticOff(RscReq.Loc, RscReq.Mod, RscReq.RscType);
			StartBit = getStartBit(RscReq.Loc, RscReq.Mod, RscReq.RscType);
			MaxRscId = getMaxRsc(RscReq.Loc, RscReq.Mod, RscReq.RscType);
			auto Bitmap = RscMaps[TileType].Bitmaps[RscReq.RscType];

			for (uint32_t i = 0; i < MaxRscId; i++) {
				uint32_t sIndex, rIndex, rBit, sBit;

				sIndex = (StartBit + StaticOffset) / 32U;
				sBit = (StartBit + StaticOffset) % 32U;
				rIndex = StartBit / 32U;
				rBit = StartBit % 32U;
				/**
				 * Check Static and runtime bitmaps find
				 * resource available in both
				 */
				if (!(Bitmap->at(sIndex).test(sBit) |
					Bitmap->at(rIndex).test(rBit))) {
						Bitmap->at(rIndex).set(rBit);
						RscReq.RscId = i;
						return XAIE_OK;
				}
				StartBit++;
			}
			return XAIE_ERR;
		}

		/**
		 * This function checks for resource availibility of a specific
		 * resource and will reserve the resources if available.
		 *
		 * @param reference to a specific resource
		 *
		 * @return XAIE_OK for success, error code for failure
		 */
		AieRC requestAllocated(XAieUserRsc& RscReq) {
			uint8_t TileType = _XAie_GetTileTypefromLoc(dev(), RscReq.Loc);
			uint32_t StaticOff, rBit, sBit, rIndex, sIndex;

			if ((_XAie_CheckModule(dev(), RscReq.Loc, RscReq.Mod) != XAIE_OK) ||
					TileType >= XAIEGBL_TILE_TYPE_MAX) {
				Logger::log(LogLevel::WARN) << __func__ <<
					" Invalid Location/Module for request" << std::endl;
				return XAIE_INVALID_ARGS;
			}

			if (RscReq.RscId > getMaxRsc(RscReq.Loc, RscReq.Mod, RscReq.RscType)) {
				Logger::log(LogLevel::WARN) << __func__ <<
					" Invalid resource id for request" << std::endl;
				return XAIE_INVALID_ARGS;
			}

			rBit = RscReq.RscId + getStartBit(RscReq.Loc, RscReq.Mod, RscReq.RscType);
			StaticOff = getStaticOff(RscReq.Loc, RscReq.Mod, RscReq.RscType);
			auto Bitmap = RscMaps[TileType].Bitmaps[RscReq.RscType];

			sIndex = (rBit + StaticOff) / 32U;
			sBit = (rBit + StaticOff) % 32U;
			rIndex = rBit / 32U;
			rBit %= 32U;

			if (!(Bitmap->at(sIndex).test(sBit) |
				Bitmap->at(rIndex).test(rBit))) {
					Bitmap->at(rIndex).set(rBit);
					return XAIE_OK;
			}
			return XAIE_ERR;
		}

		/**
		 * This function checks for resource availibility and will
		 * reserve the amount of resources needed contiguously
		 *
		 * @param reference to list of resource
		 *
		 * @return XAIE_OK for success, error code for failure
		 */
		AieRC requestContiguous(std::vector<XAieUserRsc>& RscReq) {
			uint32_t StaticOff, StartBit, MaxRscId, NumContigRscs;
			uint8_t TileType;

			if (RscReq.size() <= 0) {
				return XAIE_ERR;
			}

			TileType = _XAie_GetTileTypefromLoc(dev(), RscReq[0].Loc);
			if ((_XAie_CheckModule(dev(), RscReq[0].Loc, RscReq[0].Mod) != XAIE_OK) ||
					TileType >= XAIEGBL_TILE_TYPE_MAX) {
				Logger::log(LogLevel::WARN) << __func__ <<
					" Invalid Location/Module for request" << std::endl;
				return XAIE_INVALID_ARGS;
			}

			StaticOff = getStaticOff(RscReq[0].Loc, RscReq[0].Mod, RscReq[0].RscType);
			StartBit = getStartBit(RscReq[0].Loc, RscReq[0].Mod, RscReq[0].RscType);
			MaxRscId = getMaxRsc(RscReq[0].Loc, RscReq[0].Mod, RscReq[0].RscType);
			NumContigRscs = RscReq.size();
			auto Bitmap = RscMaps[TileType].Bitmaps[RscReq[0].RscType];

			for (uint32_t i = 0; i < MaxRscId; i++) {
				uint32_t sIndex, rIndex, sBit, rBit;
				uint8_t j, k;

				sIndex = (StartBit + StaticOff) / 32U;
				sBit = (StartBit + StaticOff) % 32U;
				rIndex = StartBit / 32U;
				rBit = StartBit % 32U;

				if (!(Bitmap->at(sIndex).test(sBit) |
					Bitmap->at(rIndex).test(rBit))) {
					for (j = 1; j < NumContigRscs; j++) {
						if ((sBit + j) >= 32U)
							sIndex++;
						if ((rBit + j) >= 32U)
							rIndex++;

						if (Bitmap->at(sIndex).test((sBit + j) % 32U) |
							Bitmap->at(rIndex).test((rBit + j) % 32U)) {
							break;
						}
					}
					if (j == NumContigRscs) {
						for (k = 0; k < j; k++) {
							rIndex = (StartBit + k) / 32U;
							rBit = (StartBit + k) % 32U;
							Bitmap->at(rIndex).set(rBit);
							RscReq[k].RscId = i + k;
						}
						return XAIE_OK;
					}
				}
				StartBit++;
			}
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
		AieRC requestBc(std::vector<XAieUserRsc>& RscReq, bool isBcAll) {
			uint32_t preferredId, MaxRscId, CommonId;

			preferredId = RscReq[0].RscId;
			if (isBcAll) {
				if (setBcAllRscs(RscReq, preferredId) != XAIE_OK)
					return XAIE_ERR;
			}

			MaxRscId = getMaxRsc(RscReq[0].Loc, RscReq[0].Mod, RscReq[0].RscType);
			if ((preferredId != XAIE_RSC_ID_ANY) &&
					(preferredId >= MaxRscId)) {
				Logger::log(LogLevel::WARN) << __func__ <<
					" Invalid resource id for request" << std::endl;
				return XAIE_INVALID_ARGS;
			}

			CommonId = getCommonRscId(RscReq);
			if (CommonId == XAIE_RSC_ID_ANY) {
				Logger::log(LogLevel::WARN) << __func__ <<
					" Unable to find common broadcast channel"
					<< std::endl;
				return XAIE_ERR;
			}

			for (uint32_t i = 0; i < RscReq.size(); i++) {
				uint8_t TType = _XAie_GetTileTypefromLoc(dev(), RscReq[i].Loc);
				auto Bitmap = RscMaps[TType].Bitmaps[RscReq[i].RscType];
				uint32_t StartBit, rBit, rIndex;

				RscReq[i].RscId = CommonId;
				StartBit = CommonId + getStartBit(RscReq[i].Loc, RscReq[i].Mod,
						RscReq[i].RscType);
				rIndex = StartBit / 32U;
				rBit = StartBit % 32U;

				Bitmap->at(rIndex).set(rBit);
			}
			return XAIE_OK;
		}

		/**
		 * This function releases the resource from runtime and static
		 * bitmaps.
		 *
		 * @param reference to resource
		 *
		 * @return XAIE_OK for success, error code for failure
		 */
		AieRC release(XAieUserRsc& RscRel) {
			uint8_t TileType = _XAie_GetTileTypefromLoc(dev(), RscRel.Loc);
			uint32_t sBit, rBit, sIndex, rIndex, StaticOff;

			if ((_XAie_CheckModule(dev(), RscRel.Loc, RscRel.Mod) != XAIE_OK) ||
					TileType >= XAIEGBL_TILE_TYPE_MAX) {
				Logger::log(LogLevel::WARN) << __func__ <<
					" Invalid Location/Module for release" << std::endl;
				return XAIE_INVALID_ARGS;
			}

			if (RscRel.RscId >= getMaxRsc(RscRel.Loc, RscRel.Mod, RscRel.RscType)) {
				Logger::log(LogLevel::WARN) << __func__ <<
					" Invalid resource id for release" << std::endl;
				return XAIE_INVALID_ARGS;
			}

			rBit = RscRel.RscId + getStartBit(RscRel.Loc, RscRel.Mod, RscRel.RscType);
			StaticOff = getStaticOff(RscRel.Loc, RscRel.Mod, RscRel.RscType);
			auto Bitmap = RscMaps[TileType].Bitmaps[RscRel.RscType];

			sIndex = (rBit + StaticOff) / 32U;
			sBit = (rBit + StaticOff) % 32U;
			rIndex = rBit / 32U;
			rBit %= 32U;

			Bitmap->at(rIndex).reset(rBit);
			Bitmap->at(sIndex).reset(sBit);
			return XAIE_OK;
		}

		/**
		 * This function frees a resource from just the
		 * runtime bitmap.
		 *
		 * @param reference to resource
		 *
		 * @return XAIE_OK for success, error code for failure
		 */
		AieRC free(XAieUserRsc& RscFree) {
			uint8_t TileType = _XAie_GetTileTypefromLoc(dev(), RscFree.Loc);
			uint32_t rBit, rIndex;

			if ((_XAie_CheckModule(dev(), RscFree.Loc, RscFree.Mod) != XAIE_OK) ||
					TileType >= XAIEGBL_TILE_TYPE_MAX) {
				Logger::log(LogLevel::WARN) << __func__ <<
					" Invalid Location/Module for free" << std::endl;
				return XAIE_INVALID_ARGS;
			}

			if (RscFree.RscId >= getMaxRsc(RscFree.Loc, RscFree.Mod, RscFree.RscType)) {
				Logger::log(LogLevel::WARN) << __func__ <<
					" Invalid resource id for free" << std::endl;
				return XAIE_INVALID_ARGS;
			}

			rBit = RscFree.RscId + getStartBit(RscFree.Loc, RscFree.Mod, RscFree.RscType);
			auto Bitmap = RscMaps[TileType].Bitmaps[RscFree.RscType];

			rIndex = rBit / 32U;
			rBit %= 32U;

			Bitmap->at(rIndex).reset(rBit);
			return XAIE_OK;
		}

		/**
		 * This function requests resource statistics for
		 * static or available resources
		 *
		 * @return XAIE_OK for success, error code for failure
		 */
		AieRC getRscStats(std::vector<XAieUserRscStat> &vStats,
				XAieRscBitmapType BType) {
			for (uint32_t i = 0; i < vStats.size(); i++) {
				uint32_t StartBit, StaticOff, MaxRscId, NumRscs = 0;
				uint32_t sBit, sIndex, rBit, rIndex;
				XAie_ModuleType M = vStats[i].Mod;
				XAieRscType T = vStats[i].RscType;
				XAie_LocType L = vStats[i].Loc;
				uint8_t TileType = _XAie_GetTileTypefromLoc(dev(), L);

				if ((_XAie_CheckModule(dev(), L, M) != XAIE_OK) ||
						TileType >= XAIEGBL_TILE_TYPE_MAX) {
					Logger::log(LogLevel::WARN) << __func__ <<
						" Invalid Location/Module for stat request"
						<< std::endl;
					return XAIE_INVALID_ARGS;
				}
				if (T >= XAIE_MAXRSC) {
					Logger::log(LogLevel::WARN) << __func__ <<
						" Invalid resource type for stat request"
						<< std::endl;
					return XAIE_INVALID_ARGS;
				}

				MaxRscId = getMaxRsc(L, M, T);
				StartBit = getStartBit(L, M, T);
				StaticOff = getStaticOff(L, M, T);
				auto Bitmap = RscMaps[TileType].Bitmaps[T];

				sIndex = (StartBit + StaticOff) / 32U;
				sBit = (StartBit + StaticOff) % 32U;
				rIndex = StartBit / 32U;
				rBit = StartBit % 32U;

				for (uint8_t j = 0; j < MaxRscId; j++) {
					if (sBit >= 32U) {
						sIndex++;
						sBit %= 32U;
					}
					if (rBit >= 32U) {
						rIndex++;
						rBit %= 32U;
					}
					if (BType == XAIE_STATIC_RSC) {
						if (Bitmap->at(sIndex).test(sBit))
							NumRscs++;
					} else if (BType == XAIE_AVAIL_RSC) {
						if (!(Bitmap->at(sIndex).test(sBit) |
								Bitmap->at(rIndex).test(rBit)))
							NumRscs++;
					}

					sBit++;
					rBit++;
				}
				vStats[i].NumRscs = NumRscs;
			}
			return XAIE_OK;
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
		AieRC writeRscBitmaps(std::ofstream &Fs, uint64_t &NumRscs) {
			uint8_t TType, RType;
			NumRscs = 0U;

			for (TType = 0; TType < XAIEGBL_TILE_TYPE_MAX; TType++) {
				uint32_t NumRows, NumMods;

				if (TType == XAIEGBL_TILE_TYPE_SHIMNOC)
					continue;

				NumMods = dev()->DevProp.DevMod[TType].NumModules;
				NumRows = _XAie_GetNumRows(dev(), TType);

				for (RType = 0; RType < XAIE_MAXRSC; RType++) {
					auto MaxRsc = &RscMaps[TType].MaxRscs[RType];
					auto Bitmap = RscMaps[TType].Bitmaps[RType];

					for (uint8_t i = 0; i < NumMods; i++) {
						uint32_t Size, ModOff = 0U;
						uint64_t RscHeader;
						XAie_ModuleType Mod;

						if (MaxRsc->at(i).second == 0U)
							continue;

						Mod = estimateModfromIndex(TType, i);
						Size = roundUp(NumRows * dev()->NumCols *
							MaxRsc->at(i).second, 32U);
						Size /= 32U;

						RscHeader = createRscHeader(TType,
								static_cast<XAieRscType>(RType),
								Mod, Size);
						Fs.write(reinterpret_cast<char *>(&RscHeader),
								sizeof(RscHeader));
						if (Fs.fail())
							return XAIE_ERR;

						if (Mod == XAIE_CORE_MOD) {
							ModOff = (MaxRsc->at(XAIE_MEM_MOD).second *
								dev()->NumCols * NumRows * 2U) / 32U;
						}

						for (uint32_t Word = 0U; Word < Size; Word += 2U) {
							uint32_t Index = Word + ModOff;
							uint64_t Payload = 0U;

							/*
							 * If bitmap size is odd number of 32 bit
							 * bitmaps, pad with extra zeros to create
							 * 64 bit payloads
							 */
							if (Word != Size - 1U) {
								Payload |= (uint64_t)Bitmap->at(Index +
										1).to_ulong() << 32U;
								Payload |= Bitmap->at(Index).to_ulong();
							} else {
								Payload = Bitmap->at(Index).to_ulong();
							}
							Fs.write(reinterpret_cast<char *>(&Payload),
									sizeof(Payload));
							if (Fs.fail())
								return XAIE_ERR;
						}
						NumRscs++;
					}
				}
			}
			return XAIE_OK;
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
		AieRC loadRscBitmaps(const char *MetaData, uint64_t NumBitmaps) {
			uint64_t *Data = (uint64_t *)MetaData;

			for (uint32_t i = 0; i < NumBitmaps; i++) {
				uint32_t BitmapSize, Bitmap64Size, BitmapOffset = 0U;
				uint32_t RscSize, NumRows, NumRscs;
				const uint64_t *Bits64;
				XAie_ModuleType Mod;
				XAieRscType RscType;
				uint64_t RscHeader;
				uint8_t TileType;

				RscHeader = Data[0U];
				TileType = (RscHeader >> XAIE_RSC_HEADER_TTYPE_SHIFT)
						& XAIE_RSC_HEADER_TTYPE_MASK;
				Mod = static_cast<XAie_ModuleType>(
						(RscHeader >> XAIE_RSC_HEADER_MOD_SHIFT)
						& XAIE_RSC_HEADER_MOD_MASK);
				RscType = static_cast<XAieRscType>(
						(RscHeader >> XAIE_RSC_HEADER_RTYPE_SHIFT)
						& XAIE_RSC_HEADER_RTYPE_MASK);
				RscSize = (RscHeader >> XAIE_RSC_HEADER_SIZE_SHIFT)
						& XAIE_RSC_HEADER_SIZE_MASK;

				if (RscSize == 0U) {
					Logger::log(LogLevel::ERROR) <<
						"Invalid resource length in bitmap"
						<< std::endl;
					return XAIE_INVALID_ARGS;
				}

				if (TileType == XAIEGBL_TILE_TYPE_SHIMNOC ||
						TileType >= XAIEGBL_TILE_TYPE_MAX) {
					Logger::log(LogLevel::ERROR) <<
						"Invalid tile type in bitmap"
						<< std::endl;
					return XAIE_INVALID_ARGS;
				}

				NumRows = _XAie_GetNumRows(dev(), TileType);
				NumRscs = getTotalRscs(TileType, Mod, RscType);
				BitmapSize = NumRows * dev()->NumCols * NumRscs;
				Bitmap64Size = roundUp(BitmapSize, 64U) / 64U;

				if (RscSize != Bitmap64Size) {
					Logger::log(LogLevel::ERROR) <<
						"Invalid resource length in bitmap"
						<< std::endl;
					return XAIE_INVALID_ARGS;
				}

				if (Mod == XAIE_CORE_MOD) {
					uint32_t MemModRscs;

					MemModRscs = getTotalRscs(TileType, XAIE_MEM_MOD,
							RscType);
					BitmapOffset = MemModRscs * dev()->NumCols *
						NumRows * 2U;
				}
				/* Get static bitmap offset */
				BitmapOffset += BitmapSize;
				BitmapOffset /= 32U;
				Bits64 = &Data[1U];

				auto StaticBitmap = RscMaps[TileType].Bitmaps[RscType];
				for (uint32_t j = 0; j < RscSize; j++) {
					uint64_t Val64 = Bits64[j];

					for (uint8_t k = 0; k < 64U; k++) {
						uint32_t Pos;

						Pos = k + (j * 64U);
						if (Pos > BitmapSize)
							break;

						if (Val64 & 1LU) {
							uint32_t Index;

							Index = BitmapOffset + (Pos / 32U);
							StaticBitmap->at(Index).set(Pos % 32U);
						}
						Val64 >>= 1U;
					}
				}

				Data = (uint64_t *)((const char *)Data + sizeof(RscHeader)
						+ (RscSize * sizeof(uint64_t)));
			}
			return XAIE_OK;
		}

	private:
		AieRscMap RscMaps[XAIEGBL_TILE_TYPE_MAX]; /**< Resource mappings */

		/**
		 * This function initializes the resource bitmaps for
		 * all tile types and resource types.
		 *
		 * @return XAIE_OK for success, XAIE_ERR on error.
		 */
		AieRC rscBitmapsInit() {
			uint8_t TType, RType;

			for (TType = 0; TType < XAIEGBL_TILE_TYPE_MAX; TType++) {
				uint32_t NumRows;

				if (TType == XAIEGBL_TILE_TYPE_SHIMNOC)
					continue;

				NumRows = _XAie_GetNumRows(dev(), TType);
				for (RType = 0; RType < (uint8_t)XAIE_MAXRSC; RType++) {
					auto MaxRsc = &RscMaps[TType].MaxRscs[RType];
					uint32_t BitmapSize, TotalRscs = 0;

					if (TType == XAIEGBL_TILE_TYPE_AIETILE) {
						uint32_t numRscs = 0;

						/* Mem Module */
						numRscs = getTotalRscs(TType, XAIE_MEM_MOD,
								static_cast<XAieRscType>(RType));
						MaxRsc->push_back(std::make_pair(XAIE_MEM_MOD, numRscs));
						TotalRscs += numRscs;
						/* Core Module */
						numRscs = getTotalRscs(TType, XAIE_CORE_MOD,
								static_cast<XAieRscType>(RType));
						MaxRsc->push_back(std::make_pair(XAIE_CORE_MOD, numRscs));
						TotalRscs += numRscs;
					} else if (TType == XAIEGBL_TILE_TYPE_MEMTILE) {
						uint32_t numRscs = 0;

						numRscs = getTotalRscs(TType, XAIE_MEM_MOD,
								static_cast<XAieRscType>(RType));
						MaxRsc->push_back(std::make_pair(XAIE_MEM_MOD, numRscs));
						TotalRscs += numRscs;
					} else {
						uint32_t numRscs = 0;

						numRscs = getTotalRscs(TType, XAIE_PL_MOD,
								static_cast<XAieRscType>(RType));
						MaxRsc->push_back(std::make_pair(XAIE_PL_MOD, numRscs));
						TotalRscs += numRscs;
					}

					/**
					 * Calculate number of bits needed for
					 * static and runtime bitmaps, hence
					 * multiply by 2U.
					 */
					BitmapSize = roundUp(NumRows * dev()->NumCols * TotalRscs * 2U, 32U);
					if (BitmapSize == 0U)
						continue;

					BitmapSize /= 32U;
					RscMaps[TType].Bitmaps[RType] = new XAieRscBitmap(BitmapSize + 1);
				}
			}
			RscMaps[XAIEGBL_TILE_TYPE_SHIMNOC] = RscMaps[XAIEGBL_TILE_TYPE_SHIMPL];
			return XAIE_OK;
		}

		/**
		 * This function returns the static resource bitmap offset
		 * for a given tile type, resource type, and module.
		 *
		 * @param TileType: Tile type of query
		 * @param Mod: Module type of query
		 * @param RscType: Resource type of query
		 *
		 * @return StartBit position
		 */
		uint32_t getStaticOff(XAie_LocType Loc, XAie_ModuleType Mod, XAieRscType Type) {
			uint8_t TileType = _XAie_GetTileTypefromLoc(dev(), Loc);
			uint32_t NumCols, NumRows, MaxRscId;

			NumRows = _XAie_GetNumRows(dev(), TileType);
			MaxRscId = getMaxRsc(Loc, Mod, Type);
			NumCols = dev()->NumCols;

			return NumRows * NumCols * MaxRscId;
		}

		/**
		 * This function returns the start bit position in
		 * the resource bitmaps for a given tile type,
		 * resource type, and module.
		 *
		 * @param TileType: Tile type of query
		 * @param Mod: Module type of query
		 * @param RscType: Resource type of query
		 *
		 * @return StartBit position
		 */
		uint32_t getStartBit(XAie_LocType Loc, XAie_ModuleType Mod, XAieRscType Type) {
			uint8_t TileType = _XAie_GetTileTypefromLoc(dev(), Loc);
			uint32_t NumCols, NumRows, StartRow;
			uint32_t StartBit, ModOffset = 0U;

			StartRow = _XAie_GetStartRow(dev(), TileType);
			NumRows = _XAie_GetNumRows(dev(), TileType);
			NumCols = dev()->NumCols;

			if (Mod == XAIE_CORE_MOD)
				ModOffset = getMaxRsc(Loc, XAIE_MEM_MOD, Type) *
					NumCols * NumRows * 2U;

			StartBit = ModOffset + (getMaxRsc(Loc, Mod, Type) *
				(Loc.Col * NumRows + (Loc.Row - StartRow)));

			return StartBit;
		}

		/**
		 * This function returns the number total number of
		 * resources for a given tile type, resource type, and
		 * module from the RscMap structure.
		 *
		 * @param TileType: Tile type of query
		 * @param Mod: Module type of query
		 * @param RscType: Resource type of query
		 *
		 * @return Max number of resources, zero of not found
		 */
		uint32_t getMaxRsc(XAie_LocType Loc, XAie_ModuleType Mod, XAieRscType Type) {
			uint8_t TileType = _XAie_GetTileTypefromLoc(dev(), Loc);

			auto vMaxRscs = RscMaps[TileType].MaxRscs[Type];
			for (auto x : vMaxRscs)
				if (x.first == Mod)
					return x.second;
			return 0;
		}

		/**
		 * This will find a common broadcast channel in
		 * static and runtime bitmaps for all resource
		 * requests in vRscs
		 *
		 * @param vRscs: list of resources
		 *
		 * @return Common broadcast channel Id, RSC_ID_ANY
		 * 	   for failure.
		 */
		uint32_t getCommonRscId(std::vector<XAieUserRsc> vRscs) {
			std::bitset<32> RscStatus;
			uint32_t MaxRscId;

			for (auto rsc : vRscs) {
				uint8_t TType = _XAie_GetTileTypefromLoc(dev(), rsc.Loc);
				uint32_t sIndex, rIndex, sBit, rBit;
				uint32_t StartBit, StaticOff, Mask;

				if ((_XAie_CheckModule(dev(), rsc.Loc, rsc.Mod) != XAIE_OK) ||
						TType >= XAIEGBL_TILE_TYPE_MAX) {
					Logger::log(LogLevel::WARN) << __func__ <<
						" Invalid Location/Module for request" << std::endl;
					return XAIE_RSC_ID_ANY;
				}

				MaxRscId = getMaxRsc(rsc.Loc, rsc.Mod, rsc.RscType);
				if (MaxRscId > 32U) {
					Logger::log(LogLevel::ERROR) << __func__ <<
						" Max resource ID larger than bitmap size"
						<< std::endl;
					return XAIE_RSC_ID_ANY;
				}

				StaticOff = getStaticOff(rsc.Loc, rsc.Mod, rsc.RscType);
				StartBit = getStartBit(rsc.Loc, rsc.Mod, rsc.RscType);
				auto Bitmap = RscMaps[TType].Bitmaps[rsc.RscType];
				Mask = (1 << MaxRscId) - 1;

				sIndex = (StartBit + StaticOff) / 32U;
				sBit = (StartBit + StaticOff) % 32U;
				rIndex = StartBit / 32U;
				rBit = StartBit % 32U;

				/* Check static and runtime bitmaps */
				RscStatus |= (Bitmap->at(sIndex) >> sBit) |
					(Bitmap->at(rIndex) >> rBit);
				RscStatus &= Mask;

				if ((sBit + MaxRscId) > 32U) {
					uint32_t remBits = (sBit + MaxRscId) % 32U;
					std::bitset<32> Temp;

					Mask = (1 << remBits) - 1 ;
					Temp |= Bitmap->at(sIndex + 1);
					Temp &= Mask;
					Temp << (MaxRscId - remBits);
					RscStatus |= Temp;
				}
				if ((rBit + MaxRscId) > 32U) {
					uint32_t remBits = (rBit + MaxRscId) % 32U;
					std::bitset<32> Temp;

					Mask = (1 << remBits) - 1 ;
					Temp |= Bitmap->at(rIndex + 1);
					Temp &= Mask;
					Temp << (MaxRscId - remBits);
					RscStatus |= Temp;
				}
			}

			if (vRscs[0].RscId == XAIE_RSC_ID_ANY) {
				for (uint8_t i = 0; i < MaxRscId; i++)
					if (!RscStatus.test(i))
						return i;
			} else {
				if (RscStatus.test(vRscs[0].RscId))
					return XAIE_RSC_ID_ANY;
				else
					return vRscs[0].RscId;
			}
			return XAIE_RSC_ID_ANY;
		}

		/**
		 * This function sets the vector of resource requests
		 * to broadcast to whole partition.
		 *
		 * @param vRscs: list of resources
		 * @param preferredId: preferred broadcast id
		 *
		 * @return XAIE_OK on success, error code for failure
		 */
		AieRC setBcAllRscs(std::vector<XAieUserRsc> &vRscs, uint32_t preferredId) {
			std::vector<XAie_LocType> Locs;
			uint32_t NumTiles;
			XAieUserRsc Rsc;
			AieRC RC;

			vRscs.clear();
			NumTiles = dev()->NumCols * dev()->NumRows;
			Locs.resize(NumTiles);
			RC = _XAie_GetUngatedLocsInPartition(dev(),
					&NumTiles, Locs.data());
			if (RC != XAIE_OK) {
				Logger::log(LogLevel::WARN) <<
					"Unable to get ungated locs" <<
					std::endl;
				return RC;
			}
			Locs.resize(NumTiles);

			Rsc.RscType = XAIE_BROADCAST;
			Rsc.RscId = preferredId;
			for (auto loc : Locs) {
				uint8_t TType;

				TType = _XAie_GetTileTypefromLoc(dev(), loc);
				Rsc.Loc = loc;
				if (TType == XAIEGBL_TILE_TYPE_AIETILE) {
					Rsc.Mod = XAIE_MEM_MOD;
					vRscs.push_back(Rsc);
					Rsc.Mod = XAIE_CORE_MOD;
					vRscs.push_back(Rsc);
				} else if (TType == XAIEGBL_TILE_TYPE_MEMTILE) {
					Rsc.Mod = XAIE_MEM_MOD;
					vRscs.push_back(Rsc);
				} else {
					Rsc.Mod = XAIE_PL_MOD;
					vRscs.push_back(Rsc);
				}
			}
			return XAIE_OK;
		}

		/**
		 * This function returns the number of resources in a given
		 * module, tile type, and resource type.
		 *
		 * @param TileType: Tile type of query
		 * @param Mod: Module type of query
		 * @param RscType: Resource type of query
		 *
		 * @return Total number of resources
		 */
		uint32_t getTotalRscs(uint8_t TileType, XAie_ModuleType Mod, XAieRscType RscType) {
			switch(RscType) {
			case XAIE_PERFCOUNT:
			{
				const XAie_PerfMod *PerfMod;

				if (Mod == XAIE_PL_MOD)
					PerfMod = &dev()->DevProp.DevMod[TileType].PerfMod[0U];
				else
					PerfMod = &dev()->DevProp.DevMod[TileType].PerfMod[Mod];

				if(!PerfMod)
					return 0U;

				return PerfMod->MaxCounterVal;
			}
			case XAIE_USEREVENT:
			{
				const XAie_EvntMod *EventMod;

				if (Mod == XAIE_PL_MOD)
					EventMod = &dev()->DevProp.DevMod[TileType].EvntMod[0U];
				else
					EventMod = &dev()->DevProp.DevMod[TileType].EvntMod[Mod];

				if(!EventMod)
					return 0U;

				return EventMod->NumUserEvents;
			}
			case XAIE_TRACECTRL:
			{
				if (_XAie_GetNumRows(dev(), TileType) > 0U)
					return XAIE_TRACE_PER_MOD;
				else
					return 0U;
			}
			case XAIE_PCEVENT:
			{
				const XAie_EvntMod *EventMod;

				if (Mod == XAIE_PL_MOD)
					EventMod = &dev()->DevProp.DevMod[TileType].EvntMod[0U];
				else
					EventMod = &dev()->DevProp.DevMod[TileType].EvntMod[Mod];

				if(!EventMod)
					return 0U;

				return EventMod->NumPCEvents;
			}
			case XAIE_SSEVENT:
			{
				const XAie_EvntMod *EventMod;

				if (Mod == XAIE_PL_MOD)
					EventMod = &dev()->DevProp.DevMod[TileType].EvntMod[0U];
				else
					EventMod = &dev()->DevProp.DevMod[TileType].EvntMod[Mod];

				if(!EventMod)
					return 0U;

				return EventMod->NumStrmPortSelectIds;

			}
			case XAIE_BROADCAST:
			{
				const XAie_EvntMod *EventMod;

				if (Mod == XAIE_PL_MOD)
					EventMod = &dev()->DevProp.DevMod[TileType].EvntMod[0U];
				else
					EventMod = &dev()->DevProp.DevMod[TileType].EvntMod[Mod];

				if(!EventMod)
					return 0U;

				return EventMod->NumBroadcastIds;

			}
			case XAIE_COMBOEVENT:
			{
				if (_XAie_GetNumRows(dev(), TileType) > 0U)
					return XAIE_COMBO_PER_MOD;
				else
					return 0U;
			}
			case XAIE_GROUPEVENT:
			{
				const XAie_EvntMod *EventMod;

				if (Mod == XAIE_PL_MOD)
					EventMod = &dev()->DevProp.DevMod[TileType].EvntMod[0U];
				else
					EventMod = &dev()->DevProp.DevMod[TileType].EvntMod[Mod];

				if(!EventMod)
					return 0U;

				return EventMod->NumGroupEvents;

			}
			default:
				return 0U;
			}
		}

		/**
		 * This function returns a resource header given a
		 * tile type resource type module and bitmap size
		 *
		 * @param Tile type of bitmap
		 * @param Resource type of bitmap
		 * @param Module type of bitmap
		 * @param Size of bitmap in 32 bits
		 *
		 * @return resource header
		 */
		uint64_t createRscHeader(uint8_t TType, XAieRscType RType,
				XAie_ModuleType Mod, uint32_t Size) {
			uint64_t Ret = 0;

			Size = (Size % 2U) ? (Size + 1U) : Size;
			Size /= 2U;

			Ret |= (static_cast<uint64_t>(TType) &
					XAIE_RSC_HEADER_TTYPE_MASK) <<
					XAIE_RSC_HEADER_TTYPE_SHIFT;
			Ret |= (static_cast<uint64_t>(RType) &
					XAIE_RSC_HEADER_RTYPE_MASK) <<
					XAIE_RSC_HEADER_RTYPE_SHIFT;
			Ret |= (static_cast<uint64_t>(Mod) &
					XAIE_RSC_HEADER_MOD_MASK) <<
					XAIE_RSC_HEADER_MOD_SHIFT;
			Ret |= (static_cast<uint64_t>(Size) &
					XAIE_RSC_HEADER_SIZE_MASK) <<
					XAIE_RSC_HEADER_SIZE_SHIFT;
			return Ret;
		}
		/**
		 * This function returns estimated Module type based
		 * on index value
		 *
		 * @return Module type, XAIE_ANY_MOD for failure
		 */
		XAie_ModuleType estimateModfromIndex(uint8_t TileType, uint32_t Index) {
			XAie_ModuleType M;

			switch(TileType) {
			case XAIEGBL_TILE_TYPE_AIETILE:
			{
				if (Index == 0U) {
					M = XAIE_MEM_MOD;
				} else {
					M = XAIE_CORE_MOD;
				}
				break;
			}
			case XAIEGBL_TILE_TYPE_MEMTILE:
			{
				if (Index == 0U) {
					M = XAIE_MEM_MOD;
				}
				break;
			}
			case XAIEGBL_TILE_TYPE_SHIMNOC:
			case XAIEGBL_TILE_TYPE_SHIMPL:
			{
				if (Index == 0U) {
					M = XAIE_PL_MOD;
				}
				break;
			}
			default:
				M = static_cast<XAie_ModuleType>(XAIE_MOD_ANY);
			}
			return M;
		}
		/**
		 * This function rounds the value to nearest multiple of
		 * aligned.
		 *
		 * @param Val: Value to round
		 * @param Aligned: Multiple to round up to
		 *
		 * @return Aligned value
		 */
		uint32_t roundUp(uint32_t Val, uint32_t Aligned) {
			return (Aligned * ((Val + (Aligned - 1U)) / Aligned));
		}
	}; /* class XAieRscMgrGeneric */
} /* namespace xaiefal */
