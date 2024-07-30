/*
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef __ADF_DBG_HPP__
#define __ADF_DBG_HPP__

#include <adf/adf_types.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <iostream>
#include <adf/adf_api_message.h>
#include <sstream>
#include <iomanip>

namespace aiectrl{

    template< typename T >
    std::string int_to_hex( T i )
    {
      std::stringstream stream;
      stream << "0x" 
            //<< std::setfill ('0') << std::setw(sizeof(T)*2) 
            << std::hex << i;
      return stream.str();
    }

    inline err_code errorMsg(err_code code, std::string msg)
    {
        std::cerr << msg << std::endl;
        return code;
    }

    inline AieRC errorMsg(AieRC code, std::string msg)
    {
        std::cerr << msg << std::endl;
        return code;
    }

    inline void debugMsg(std::string msg)
    {
    #ifdef UADF_DBG
        std::cout << msg << std::endl;
    #endif

    }

    inline void infoMsg(std::string msg)
    {
        std::cout << msg << std::endl;
    }

    inline void dump_transaction_aximm ( XAie_TxnInst * txn, std::ostream & sbuf )
    {
      for (unsigned i = 0; i < txn->NumCmds; i++)
      {
            XAie_TxnCmd cmd = txn->CmdBuf[i];
            
            sbuf << "AXI MM W addr=" << int_to_hex ( cmd.RegOff + 0x20000000000 )
                 << " data=" << int_to_hex(cmd.Value)
                 << " len=4"
                 << " col=" << std::to_string( (cmd.RegOff >> 25) & 0x3F )
                 << " row=" << std::to_string((cmd.RegOff >> 20) & 0x1F )
                 << " tile_addr=" << int_to_hex (cmd.RegOff & 0xFFFFF )
                 << std::endl;
      }
    }

    inline void dump_transaction( XAie_TxnInst * txn, std::ostream & sbuf)
    {
        sbuf << "Tid: " << txn->Tid << " Flags: " << txn->Flags << " NumCmd: " << txn->NumCmds << " MaxCmds: " <<
                     txn->MaxCmds << std::endl;

        for (unsigned i = 0; i < txn->NumCmds; i++)
        {
            XAie_TxnCmd cmd = txn->CmdBuf[i];
            sbuf << "Cmd " << i 
                      << " Op: " <<  cmd.Opcode 
                      << " Mask: " << cmd.Mask 
                      << " RegOff: " << cmd.RegOff
                      << " Val: " << cmd.Value
                      << " Ptr: " << cmd.DataPtr
                      << " Sz: " << cmd.Size
                      << std::endl;
        }
    }

    inline void dump_transaction( XAie_TxnInst * txn, const std::string & fileName )
    {
        std::ofstream fs(fileName, std::ofstream::out);
        dump_transaction(txn, fs);
        fs.close();
    }       

    inline void dump_transaction_aximm ( XAie_TxnInst * txn)
    {
      dump_transaction_aximm( txn, std::cout);
    }

    inline void dump_transaction( XAie_TxnInst * txn )
    {
        dump_transaction( txn, std::cout);
    }

    inline void dump_gmio_cfg ( GMIOConfig cfg)
    {
        std::cout << "id: " << cfg.id 
                  << " name: " << cfg.name
                  << " logicalName: " << cfg.logicalName
                  << " type: " << cfg.type
                  << " shimColumn: " << cfg.shimColumn
                  << " channelNum: " << cfg.channelNum
                  << " streamId: " << cfg.streamId
                  << " burstLength: " << cfg.burstLength
                  << std::endl;
    }

};

#endif
