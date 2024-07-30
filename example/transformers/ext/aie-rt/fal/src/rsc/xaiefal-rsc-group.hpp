// Copyright(C) 2020 - 2021 by Xilinx, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
/**
 * @param file xaiefal-rsc-base.hpp
 * Base classes for AI engine resources management
 */

#include <fstream>
#include <functional>
#include <map>
#include <vector>
#include <xaiengine.h>
#include <xaiefal/common/xaiefal-common.hpp>
#include <xaiefal/common/xaiefal-log.hpp>

#pragma once

namespace xaiefal {
	class XAieRsc;
	class XAieRscGetRscsWrapper;
	class XAieDevHandle;

	/**
	 * @enum XAieRscType
	 * @brief Enums representing all requestable resources
	 */
	enum XAieRscType {
		XAIE_PERFCOUNT = 0U,
		XAIE_USEREVENT,
		XAIE_TRACECTRL,
		XAIE_PCEVENT,
		XAIE_SSEVENT,
		XAIE_BROADCAST,
		XAIE_COMBOEVENT,
		XAIE_GROUPEVENT,
		XAIE_MAXRSC,
		XAIE_TRACEEVENT = 0x100U
	};

	/**
	 * @struct XAieUserRscStat
	 * @brief Data needed to make resource statistics request
	 */
	struct XAieUserRscStat {
		/**< Location of Request */
		XAie_LocType Loc {
			XAie_TileLoc(XAIE_LOC_ANY, XAIE_LOC_ANY)
		};
		/**< Module of Request */
		XAie_ModuleType Mod {
			static_cast<XAie_ModuleType>(XAIE_MOD_ANY)
		};
		/**< Resource Type of Request */
		XAieRscType RscType {
			static_cast<XAieRscType>(XAIE_RSC_TYPE_ANY)
		};
		/**< Number of requests */
		uint32_t NumRscs {0U};
	};

	/**
	 * @struct XAieRscStat
	 * @brief struct of resource usage statistics
	 */
	struct XAieRscStat {
		XAieRscStat(const std::string &Name = ""):
			GroupName(Name) {}
		std::string GroupName; /**< name of the group of resources */
		std::map<std::tuple<uint8_t, uint8_t, uint32_t, uint32_t>,
			uint32_t> Rscs; /**< number of resources info:
					       * key: col, row, mod type, rsc type>
					       * value: numer of resources
					       */
		/**
		 * This function adds resource information to the resources structure.
		 * @param Loc tile location
		 * @param Mod module type
		 * @param RscType resource type
		 * @param NumRscs number of resources of the resource type
		 */
		void addRscStat(XAie_LocType Loc, XAie_ModuleType Mod,
				XAieRscType RscType, uint32_t NumRscs) {
			(void)Mod;
			(void)RscType;
			(void)NumRscs;

			std::tuple<uint8_t, uint8_t, uint32_t, uint32_t> rKey {
				Loc.Col, Loc.Row, static_cast<uint32_t>(Mod),
				static_cast<uint32_t>(RscType)};

			if (Rscs.find(rKey) != Rscs.end()) {
				Rscs[rKey] += NumRscs;
			} else {
				Rscs[rKey] = NumRscs;
			}
		}

		/**
		 * This function returns number of resource of a type of resources
		 * of a module of a tile in this resource stat structure.
		 * @param Loc tile location
		 * @param Mod module type
		 * @param RscType resource type
		 *
		 * @return number of resources of the type of resources of a module
		 *	of a tile.
		 */
		uint32_t getNumRsc(XAie_LocType Loc, XAie_ModuleType Mod,
				XAieRscType RscType) {
			uint32_t NumRscs = 0;
			std::tuple<uint8_t, uint8_t, uint32_t, uint32_t> rKey {
				Loc.Col, Loc.Row, static_cast<uint32_t>(Mod),
				static_cast<uint32_t>(RscType)};

			if (Rscs.find(rKey) != Rscs.end()) {
				NumRscs =  Rscs[rKey];
			}

			return NumRscs;
		}

		/**
		 * This function checkes if the resources stat map has resources
		 *
		 * @return true if the resources stat map is not empty, otherwise
		 *	false.
		 */
		bool hasRsc() const {
			return (Rscs.size() == 0) ? false : true;
		}

		/**
		 * This function shows the resources information of this resource group
		 */
		void show() const {
			Logger::log(LogLevel::INFO) << GroupName << ":" << std::endl;
			for (auto const& r : Rscs) {
				std::string Str = "\t(" +
					std::to_string(static_cast<uint32_t>(std::get<0>(r.first))) +
					", " +
					std::to_string(static_cast<uint32_t>(std::get<1>(r.first))) +
					"): ";

				switch (std::get<2>(r.first)) {
				case XAIE_MEM_MOD:
					Str += "Mem";
					break;
				case XAIE_CORE_MOD:
					Str += "Core";
					break;
				case XAIE_PL_MOD:
					Str += "SHIM";
					break;
				default:
					Str += "Unknown";
					break;
				}
				Str += ": ";

				switch (std::get<3>(r.first)) {
				case static_cast<uint32_t>(XAIE_PERFCOUNT):
					Str += "Perfcount ";
					break;
				case static_cast<uint32_t>(XAIE_USEREVENT):
					Str += "UserEvents ";
					break;
				case static_cast<uint32_t>(XAIE_TRACECTRL):
					Str += "TraceCntr ";
					break;
				case static_cast<uint32_t>(XAIE_PCEVENT):
					Str += "PCEvents ";
					break;
				case static_cast<uint32_t>(XAIE_SSEVENT):
					Str += "SSEventsPorts ";
					break;
				case static_cast<uint32_t>(XAIE_BROADCAST):
					Str += "BC ";
					break;
				case static_cast<uint32_t>(XAIE_COMBOEVENT):
					Str += "ComboEvents ";
					break;
				case static_cast<uint32_t>(XAIE_GROUPEVENT):
					Str += "GroupEvents ";
					break;
				case static_cast<uint32_t>(XAIE_TRACEEVENT):
					Str += "TraceEvents ";
					break;
				default:
					Str += "Unknown ";
					break;
				}
				Str += ": " + std::to_string(r.second);
				std::cout << Str << std::endl;
			}
		}
	};

	/**
	 * @class XAieRscGroupBase
	 * @brief base class to resources functional group
	 */
	class XAieRscGroupBase {
	public:
		XAieRscGroupBase(std::shared_ptr<XAieDevHandle> DevHd,
				const std::string &Name = ""):
			FuncName(Name), AieHd(DevHd) {}
		XAieRscGroupBase() {}
		~XAieRscGroupBase() {}

		/**
		 * This function is to return resources statics of this function
		 * group.
		 *
		 * @return resoruce group statics object.
		 */
		XAieRscStat getRscStat() const {
			return getRscStat(XAie_TileLoc(XAIE_LOC_ANY, XAIE_LOC_ANY));
		}
		/**
		 * This function is to return resources statics of this function
		 * group of a specific tile.
		 *
		 * @Loc tile location
		 *
		 * @return resoruce group statics object.
		 */
		XAieRscStat getRscStat(XAie_LocType Loc) const {
			return _getRscStatTile(Loc, static_cast<XAie_ModuleType>(XAIE_MOD_ANY),
					static_cast<XAieRscType>(XAIE_RSC_TYPE_ANY), XAIE_RSC_ID_ANY);
		}
		/**
		 * This function is to return resources statics of this function
		 * group of a specific tile of a specific resource type.
		 *
		 * @Loc tile location
		 * @RscType resource type
		 *
		 * @return resoruce group statics object.
		 */
		XAieRscStat getRscStat(XAie_LocType Loc, XAieRscType RscType) const {
			return _getRscStatTile(Loc, static_cast<XAie_ModuleType>(XAIE_MOD_ANY),
					RscType, XAIE_RSC_ID_ANY);
		}
		/**
		 * This function is to return resources statics of this function
		 * group of a specific tile, specific module.
		 *
		 * @Loc tile location
		 * @Mod module type
		 *
		 * @return resoruce group statics object.
		 */
		XAieRscStat getRscStat(XAie_LocType Loc, XAie_ModuleType Mod) const {
			return getRscStat(Loc, Mod, static_cast<XAieRscType>(XAIE_RSC_TYPE_ANY));
		}
		/**
		 * This function is to return resources statics of this function
		 * group of a specific tile, specific module, specific resource.
		 *
		 * @Loc tile location
		 * @Mod module type
		 * @RscType resource type
		 *
		 * @return resoruce group statics object.
		 */
		XAieRscStat getRscStat(XAie_LocType Loc, XAie_ModuleType Mod,
			XAieRscType RscType) const {
			return getRscStat(Loc, Mod, RscType, XAIE_RSC_ID_ANY);
		}

		/**
		 * This function is to return resources statics of this function
		 * group of a specific tile, specific module, specific resource.
		 *
		 * @Loc tile location
		 * @Mod module type
		 * @RscType resource type
		 * @RscId resource Id
		 *
		 * @return resoruce group statics object.
		 */
		XAieRscStat getRscStat(XAie_LocType Loc, XAie_ModuleType Mod,
			XAieRscType RscType, uint32_t RscId) const {
			return _getRscStatTile(Loc, Mod, RscType, RscId);
		}

		/**
		 * This function is to return resources statics of this function
		 * group of specific tiles, specific resource.
		 *
		 * @vLocs tile locations
		 * @RscType resource type
		 *
		 * @return resoruce group statics object.
		 */
		XAieRscStat getRscStat(const std::vector<XAie_LocType> &vLocs,
			XAieRscType RscType) const {
			return _getRscStat(vLocs, static_cast<XAie_ModuleType>(XAIE_MOD_ANY),
					RscType, XAIE_RSC_ID_ANY);
		}

		/**
		 * This function is to return resources statics of this function
		 * group of specific tiles, specific module, specific resource, specific
		 * resource id.
		 *
		 * @vLocs tile locations
		 * @RscType resource type
		 * @RscId resource Id
		 *
		 * @return resoruce group statics object.
		 */
		XAieRscStat getRscStat(const std::vector<XAie_LocType> &vLocs,
			XAieRscType RscType, uint32_t RscId) const {
			return _getRscStat(vLocs, static_cast<XAie_ModuleType>(XAIE_MOD_ANY),
					RscType, RscId);
		}

		virtual AieRC addRsc(std::shared_ptr<XAieRsc> R) {
			(void)R;
			throw std::invalid_argument("Add Rsc not supported, rsc group: " + FuncName);
			return XAIE_ERR;
		}
	protected:
		std::string FuncName; /**< function name of this group */
		std::weak_ptr<XAieDevHandle> AieHd; /**< AI engine device instance */

		/**
		 * This function is to return resources statics of this function
		 * group.
		 *
		 * @vLocs tile locations, (0xFF, 0xFF) for any tile
		 * @Mod module type, 0xFFFFFFFF for any module
		 * @RscType, resource type, 0xFFFFFFFF for any resource type
		 * @RscId, resource Id, 0xFFFFFFFF for any resource id
		 *
		 * @return resoruce group statics object.
		 */
		virtual XAieRscStat _getRscStat(const std::vector<XAie_LocType> &vLocs,
				XAie_ModuleType Mod, XAieRscType RscType,
				uint32_t RscId) const {
			XAieRscStat RscStat(FuncName);

			(void)vLocs;
			(void)Mod;
			(void)RscType;
			(void)RscId;

			throw std::invalid_argument("Get rsc stat not supported, rsc group: " + FuncName);
			return RscStat;
		}
		XAieRscStat _getRscStatTile(XAie_LocType Loc, XAie_ModuleType Mod,
			XAieRscType RscType, uint32_t RscId) const {
			std::vector<XAie_LocType> vLocs;

			vLocs.push_back(Loc);
			return _getRscStat(vLocs, Mod, RscType, RscId);
		}

		/**
		 * This function is to create tile vectors for all tiles of
		 * an AIE partition device.
		 *
		 * @param DevInst AI engine device instance pointer
		 * @return vector of tiles locations of all tiles of the
		 *	   AIE partition device.
		 */
		static inline std::vector<XAie_LocType> _getAllTilesLocs(
				XAie_DevInst *DevInst) {
			std::vector<XAie_LocType> vLocs(DevInst->NumCols *
					DevInst->NumRows);
			uint8_t C = 0, R = 0;

			for (std::vector<XAie_LocType>::iterator it = vLocs.begin() ;
				it != vLocs.end(); ++it) {
				*it = XAie_TileLoc(C, R);

				R++;
				if (R == DevInst->NumRows) {
					R = 0;
					C++;
				}
			}

			return vLocs;
		}

		/**
		 * This function is to create resources statistics array to pass
		 * to lower level driver to get the resources statistics.
		 *
		 * @param DevInst AI engine device instance pointer
		 * @param vLocs tile locations, (0xFF, 0xFF) for any tile
		 * @param Mod module type, 0xFFFFFFFF for any module
		 * @param RscType, resource type, 0xFFFFFFFF for any resource type
		 *
		 * @return resource statistics vector which can be passed to
		 *	   lower level driver.
		 */
		static inline std::vector<XAieUserRscStat> _createDrvRscStats(
				XAie_DevInst *DevInst,
				const std::vector<XAie_LocType> &vLocs,
				XAie_ModuleType Mod, XAieRscType RscType) {
			uint32_t NumStats = 0;
			uint32_t MaxNumRscs = static_cast<uint32_t>(XAIE_MAXRSC);
			uint32_t ShimModNumRscs = 0, CoreModNumRscs = 0;
			uint32_t CoreMemModNumRscs = 0, MemModNumRscs = 0;
			uint32_t NumTiles = 0, NumCoreTiles = 0, NumShimTiles = 0;
			uint32_t i;

			if (RscType != XAIE_RSC_TYPE_ANY && RscType >= XAIE_MAXRSC) {
				throw std::invalid_argument("Invalid Rsc Type to get rsc stat");
			}

			// Get number of resources per module
			if (Mod == static_cast<XAie_ModuleType>(XAIE_MOD_ANY) ||
					Mod == XAIE_PL_MOD) {
				if (RscType == static_cast<XAieRscType>(XAIE_RSC_TYPE_ANY)) {
					ShimModNumRscs = MaxNumRscs - 1;
				} else if (RscType == XAIE_PCEVENT) {
					ShimModNumRscs = 0;
				} else {
					ShimModNumRscs = 1;
				}
			}
			if (Mod == static_cast<XAie_ModuleType>(XAIE_MOD_ANY) ||
					Mod == XAIE_CORE_MOD) {
				if (RscType == static_cast<XAieRscType>(XAIE_RSC_TYPE_ANY)) {
					CoreModNumRscs = MaxNumRscs;
				} else {
					CoreModNumRscs = 1;
				}
			}
			if (Mod == static_cast<XAie_ModuleType>(XAIE_MOD_ANY) ||
					Mod == XAIE_MEM_MOD) {
				if (RscType == static_cast<XAieRscType>(XAIE_RSC_TYPE_ANY)) {
					MemModNumRscs = MaxNumRscs - 1;
					CoreMemModNumRscs = MaxNumRscs - 2;
				} else if (RscType == XAIE_PCEVENT) {
					MemModNumRscs = 0;
					CoreMemModNumRscs = 0;
				} else {
					MemModNumRscs = 1;
					if (RscType == XAIE_SSEVENT) {
						CoreMemModNumRscs = 0;
					} else {
						CoreMemModNumRscs = 1;
					}
				}
			}

			for (auto L: vLocs) {
				if (L.Row == XAIE_LOC_ANY ||
					L.Col == XAIE_LOC_ANY) {
					throw std::invalid_argument("Invalid tile location to get rsc stat");
				}
				NumTiles++;
				if (L.Row == 0) {
					NumShimTiles++;
				} else if (L.Row >= DevInst->AieTileRowStart) {
					NumCoreTiles++;
				}
			}

			// Calculate number of resource statistics
			NumStats += NumShimTiles * ShimModNumRscs;
			NumStats += NumCoreTiles * CoreModNumRscs;
			NumStats += NumCoreTiles * CoreMemModNumRscs;
			NumStats += (NumTiles - NumShimTiles - NumCoreTiles) * MemModNumRscs;

			std::vector<XAieUserRscStat> RscsStats(NumStats);

			// Fill in Rsc stats parameters
			i = 0;
			for (auto L: vLocs) {
				if (L.Row == 0) {
					// Shim Tiles
					if (ShimModNumRscs == 0) {
						continue;
					}
					if (ShimModNumRscs == 1) {
						RscsStats[i].Loc = L;
						RscsStats[i].Mod = XAIE_PL_MOD;
						RscsStats[i].RscType = RscType;
						RscsStats[i].NumRscs = 0;
						i++;
						continue;
					}
					for (uint8_t Rsc = static_cast<uint8_t>(XAIE_PERFCOUNT);
						Rsc < static_cast<uint8_t>(XAIE_MAXRSC); Rsc++) {
						if (Rsc == static_cast<uint8_t>(XAIE_PCEVENT)) {
							continue;
						}
						RscsStats[i].Loc = L;
						RscsStats[i].Mod = XAIE_PL_MOD;
						RscsStats[i].RscType = static_cast<XAieRscType>(Rsc);
						RscsStats[i].NumRscs = 0;
						i++;
					}
				} else if (L.Row >= DevInst->AieTileRowStart) {
					// Core tiles, core modules
					if (CoreModNumRscs == 1) {
						RscsStats[i].Loc = L;
						RscsStats[i].Mod = XAIE_CORE_MOD;
						RscsStats[i].RscType = RscType;
						RscsStats[i].NumRscs = 0;
						i++;
					} else if (CoreModNumRscs > 1) {
						for (uint8_t Rsc = static_cast<uint8_t>(XAIE_PERFCOUNT);
							Rsc < static_cast<uint8_t>(XAIE_MAXRSC); Rsc++) {
							RscsStats[i].Loc = L;
							RscsStats[i].Mod = XAIE_CORE_MOD;
							RscsStats[i].RscType = static_cast<XAieRscType>(Rsc);
							RscsStats[i].NumRscs = 0;
							i++;
						}
					}

					// Core tiles, mem modules
					if (CoreMemModNumRscs == 0) {
						continue;
					}
					if (CoreMemModNumRscs == 1) {
						RscsStats[i].Loc = L;
						RscsStats[i].Mod = XAIE_MEM_MOD;
						RscsStats[i].RscType = RscType;
						RscsStats[i].NumRscs = 0;
						i++;
						continue;
					}
					for (uint8_t Rsc = static_cast<uint8_t>(XAIE_PERFCOUNT);
						Rsc < static_cast<uint8_t>(XAIE_MAXRSC); Rsc++) {
						if (Rsc == static_cast<uint8_t>(XAIE_PCEVENT) ||
							Rsc == static_cast<uint8_t>(XAIE_SSEVENT)) {
							continue;
						}
						RscsStats[i].Loc = L;
						RscsStats[i].Mod = XAIE_MEM_MOD;
						RscsStats[i].RscType = static_cast<XAieRscType>(Rsc);
						RscsStats[i].NumRscs = 0;
						i++;
					}
				} else {
					// Other tiles
					if (MemModNumRscs == 0) {
						continue;
					}
					if (MemModNumRscs == 1) {
						RscsStats[i].Loc = L;
						RscsStats[i].Mod = XAIE_MEM_MOD;
						RscsStats[i].RscType = RscType;
						RscsStats[i].NumRscs = 0;
						i++;
						continue;
					}
					for (uint8_t Rsc = static_cast<uint8_t>(XAIE_PERFCOUNT);
						Rsc < static_cast<uint8_t>(XAIE_MAXRSC); Rsc++) {
						if (Rsc == static_cast<uint8_t>(XAIE_PCEVENT)) {
							continue;
						}
						RscsStats[i].Loc = L;
						RscsStats[i].Mod = XAIE_MEM_MOD;
						RscsStats[i].RscType = static_cast<XAieRscType>(Rsc);
						RscsStats[i].NumRscs = 0;
						i++;
					}
				}
			}

			return RscsStats;
		}
	};
}
