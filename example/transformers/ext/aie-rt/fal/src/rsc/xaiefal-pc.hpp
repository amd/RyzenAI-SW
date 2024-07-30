// Copyright(C) 2020 - 2021 by Xilinx, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <fstream>
#include <functional>
#include <string.h>
#include <vector>
#include <xaiengine.h>

#include <xaiefal/rsc/xaiefal-rsc-base.hpp>
#include <xaiefal/rsc/xaiefal-rscmgr.hpp>

#pragma once

namespace xaiefal {
	/**
	 * @class XAiePCEvent
	 * @brief AI engine PC event resource class
	 */
	class XAiePCEvent: public XAieSingleTileRsc {
	public:
		XAiePCEvent() = delete;
		XAiePCEvent(std::shared_ptr<XAieDevHandle> DevHd,
			XAie_LocType L):
			XAieSingleTileRsc(DevHd, L, XAIE_CORE_MOD, XAIE_PCEVENT),
			PcAddr(0) {
			State.Initialized = 1;
		}
		XAiePCEvent(XAieDev &Dev, XAie_LocType L):
			XAiePCEvent(Dev.getDevHandle(), L) {};
		/**
		 * This function updates PC address of the PC event.
		 *
		 * @param Addr PC address
		 * @return XAIE_OK for success, error code for failure.
		 */
		AieRC updatePcAddr(uint32_t Addr) {
			PcAddr = Addr;
			if (State.Running == 1) {
				XAie_EventPCDisable(dev(), Loc, vRscs[0].RscId);
				XAie_EventPCEnable(dev(), Loc, vRscs[0].RscId, PcAddr);
			} else {
				State.Configured = 1;
			}
			return XAIE_OK;
		}
		/**
		 * This function returns PC address of the PC event.
		 *
		 * @return PC address of the PC event
		 */
		uint32_t getPcAddr() const {
			return PcAddr;
		}
		/**
		 * This function returns PC event.
		 * It needs to be called after reserve() succeeds.
		 *
		 * @param E return the PC range event.
		 * @return XAIE_OK for success, error code for failure
		 */
		AieRC getEvent(XAie_Events &E) const {
			AieRC RC = XAIE_OK;
			if (State.Reserved == 0) {
				Logger::log(LogLevel::ERROR) << "PC Event " << __func__ << " (" <<
					static_cast<uint32_t>(Loc.Col) << "," << static_cast<uint32_t>(Loc.Row) << ")" <<
					" resource not resesrved." << std::endl;
				RC = XAIE_ERR;
			} else {
				E = XAIE_EVENT_PC_0_CORE;
				E = (XAie_Events)(static_cast<uint32_t>(E) + vRscs[0].RscId);
			}
			return RC;
		}
	protected:
		uint32_t PcAddr; /**< PC address */
	private:
		AieRC _reserve() {
			AieRC RC;

			if (_XAie_GetTileTypefromLoc(dev(), Loc) != XAIEGBL_TILE_TYPE_AIETILE) {
				Logger::log(LogLevel::ERROR) << "PC event " << __func__ << " (" <<
					static_cast<uint32_t>(Loc.Col) << "," << static_cast<uint32_t>(Loc.Row) << ")" <<
					" tile is not core tile." << std::endl;
				RC = XAIE_ERR;
			} else {
				XAieUserRsc Rsc;

				Rsc.Loc = Loc;
				Rsc.Mod = Mod;
				Rsc.RscType = Type;
				Rsc.RscId = preferredId;
				vRscs.push_back(Rsc);
				RC = AieHd->rscMgr()->request(*this);
				if (RC != XAIE_OK) {
					Logger::log(LogLevel::WARN) << "PC event " << __func__ << " (" <<
						static_cast<uint32_t>(Loc.Col) << "," << static_cast<uint32_t>(Loc.Row) << ")" <<
						" no available resource.\n";
				}
			}
			return RC;
		}
		AieRC _release() {
			AieRC RC;

			RC = AieHd->rscMgr()->release(*this);
			vRscs.clear();
			return RC;
		}
		AieRC _start() {
			return XAie_EventPCEnable(dev(), Loc, vRscs[0].RscId, PcAddr);
		}
		AieRC _stop() {
			return XAie_EventPCDisable(dev(), Loc, vRscs[0].RscId);
		}
	};

	/**
	 * @class XAiePCRange
	 * @brief AI engine PC addresses range resource class
	 */
	class XAiePCRange: public XAieSingleTileRsc {
	public:
		XAiePCRange() = delete;
		XAiePCRange(std::shared_ptr<XAieDevHandle> DevHd,
			XAie_LocType L):
			XAieSingleTileRsc(DevHd, L, XAIE_PCEVENT) {
			for (int i = 0;
				i < (int)(sizeof(PcAddrs)/sizeof(PcAddrs[0]));
				i++) {
				PcAddrs[i] = 0;
			}
			State.Initialized = 1;
		}
		XAiePCRange(XAieDev &Dev, XAie_LocType L):
			XAiePCRange(Dev.getDevHandle(), L) {};
		/**
		 * This function updates PC addresses of the range.
		 *
		 * @param Addr0 starting address
		 * @param Addr1 ending address
		 * @return XAIE_OK for success, error code for failure.
		 */
		AieRC updatePcAddr(uint32_t Addr0, uint32_t Addr1) {
			PcAddrs[0] = Addr0;
			PcAddrs[1] = Addr1;
			if (State.Running == 1) {
				XAie_EventPCDisable(dev(), Loc, vRscs[0].RscId);
				XAie_EventPCDisable(dev(), Loc, vRscs[1].RscId);
				XAie_EventPCEnable(dev(), Loc, vRscs[0].RscId, PcAddrs[0]);
				XAie_EventPCEnable(dev(), Loc, vRscs[1].RscId, PcAddrs[1]);
			} else {
				State.Configured = 1;
			}
			return XAIE_OK;
		}
		/** This function returns PC addresses of the range.
		 *
		 * @param Addr0 returns the starting address
		 * @param Addr1 returns the ending address
		 */
		void getPcAddr(uint32_t &Addr0, uint32_t &Addr1) const {
			Addr0 = PcAddrs[0];
			Addr1 = PcAddrs[1];
		}
		/**
		 * This function returns PC range event.
		 * It needs to be called after reserve() succeeds.
		 *
		 * @param E return the PC range event.
		 * @return XAIE_OK for success, error code for failure
		 */
		AieRC getEvent(XAie_Events &E) const {
			AieRC RC = XAIE_OK;
			if (State.Reserved == 0) {
				Logger::log(LogLevel::ERROR) << "PC range " << __func__ << " (" <<
					static_cast<uint32_t>(Loc.Col) << "," << static_cast<uint32_t>(Loc.Row) << ")" <<
					" resource not resesrved." << std::endl;
				RC = XAIE_ERR;
			} else if (vRscs[0].RscId == 0) {
				E = XAIE_EVENT_PC_RANGE_0_1_CORE;
			} else {
				E = XAIE_EVENT_PC_RANGE_2_3_CORE;
			}
			return RC;
		}
	protected:
		uint32_t PcAddrs[2]; /**< starting and end PC addresses */
	private:
		AieRC _reserve() {
			AieRC RC;

			if (_XAie_GetTileTypefromLoc(dev(), Loc) != XAIEGBL_TILE_TYPE_AIETILE) {
				Logger::log(LogLevel::ERROR) << "PC range " << __func__ << " (" <<
					static_cast<uint32_t>(Loc.Col) << "," << static_cast<uint32_t>(Loc.Row) << ")" <<
					" not core tile." << std::endl;
				RC = XAIE_ERR;
			} else {
				XAieUserRsc Rsc;

				Rsc.Loc = Loc;
				Rsc.Mod = Mod;
				Rsc.RscType = Type;
				Rsc.RscId = preferredId;

				vRscs.insert(vRscs.end(), {Rsc, Rsc});
				RC = AieHd->rscMgr()->request(*this);
			}

			if (RC != XAIE_OK) {
				Logger::log(LogLevel::ERROR) << "PC range " << __func__ << " (" <<
					static_cast<uint32_t>(Loc.Col) << "," << static_cast<uint32_t>(Loc.Row) << ")" <<
					" resource not availalble." << std::endl;
				vRscs.clear();
			}
			return RC;
		}
		AieRC _release() {
			AieRC RC;

			RC = AieHd->rscMgr()->release(*this);
			vRscs.clear();
			return RC;
		}
		AieRC _start() {
			AieRC RC;

			RC = XAie_EventPCEnable(dev(), Loc, vRscs[0].RscId, PcAddrs[0]);
			if (RC == XAIE_OK) {
				RC = XAie_EventPCEnable(dev(), Loc, vRscs[1].RscId, PcAddrs[1]);
			}
			if (RC != XAIE_OK) {
				Logger::log(LogLevel::ERROR) << "PC range " << __func__ << " (" <<
					static_cast<uint32_t>(Loc.Col) << "," << static_cast<uint32_t>(Loc.Row) << ")" <<
					" failed to start." << std::endl;
			}
			return RC;
		}
		AieRC _stop() {
			int iRC;
			AieRC RC;

			iRC = (int)XAie_EventPCDisable(dev(), Loc, vRscs[0].RscId);
			iRC |= (int)XAie_EventPCDisable(dev(), Loc, vRscs[1].RscId);

			if (iRC != (int)XAIE_OK) {
				Logger::log(LogLevel::ERROR) << "PC range " << __func__ << " (" <<
					static_cast<uint32_t>(Loc.Col) << "," << static_cast<uint32_t>(Loc.Row) << ")" <<
					" failed to stop." << std::endl;
				RC = XAIE_ERR;
			} else {
				RC = XAIE_OK;
			}
			return RC;
		}
		void _getReservedRscs(std::vector<XAieUserRsc> &vR) const {
			vR.insert(vR.end(), vRscs.begin(), vRscs.end());
		}
	};
}
