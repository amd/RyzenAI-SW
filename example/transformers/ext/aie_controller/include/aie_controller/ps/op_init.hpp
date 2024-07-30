/*
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef __OPINIT_HPP__
#define __OPINIT_HPP__

extern "C"
{
    #include <xaiengine.h>
};

#include "op_types.h"
#include "ucode_gen.hpp"

using namespace std;
namespace aiectrl{


class instr_base{
public:
    instr_base() : op_ptr_(nullptr) {
    }
    virtual ~instr_base() {
    }
    virtual void serialize( void * ptr) const
    {
        //cout << "SERIALIZE numbytes: " << op_ptr_->size_in_bytes << std::endl;
        memcpy ( ptr, op_ptr_, op_ptr_->size_in_bytes);
    }

    virtual unsigned size() const
    {
        return op_ptr_->size_in_bytes;
    }

    virtual string type() const = 0;
protected:
    op_base * op_ptr_;
};

class dbgPrint_op : public instr_base {
public:    
    dbgPrint_op() = delete;

    dbgPrint_op(string str) : dbgPrint_op(str.c_str()) { }

    dbgPrint_op( const char * str ){
        print_op_t * p = new print_op_t();
        p->b.type = e_DBGPRINT_OP;
        p->b.size_in_bytes = sizeof(print_op_t);
        memset(p->msg, '\0', DEBUG_STR_MAX_LEN);
        strncpy ( p->msg, str, DEBUG_STR_MAX_LEN );
        op_ptr_ = (op_base*)p;
    }

    virtual ~dbgPrint_op()
    {
        delete reinterpret_cast<print_op_t*>(op_ptr_);
    }

    virtual string type() const override { return "dbgPrint_op";}
};

class wait_op: public instr_base{
public:
    wait_op() = delete;
    wait_op( const shimTileHandle & shim )
    {
        wait_op_t * wait_ptr = new wait_op_t();
        wait_ptr->b.type = e_WAIT_OP;
        wait_ptr->b.size_in_bytes = sizeof(wait_op_t);
        wait_ptr->tileLoc = shim.getTileLoc();
        wait_ptr->channelNum = shim.getGMIOConfig()->channelNum;
        wait_ptr->dma_direction =  shim.getGMIOConfig()->type == GMIOConfig::gm2aie ? DMA_MM2S : DMA_S2MM;
        op_ptr_ = (op_base*)wait_ptr;
    }

    wait_op( const memTilePortHandle & mem )
    {
        wait_op_t * wait_ptr = new wait_op_t();
        wait_ptr->b.type = e_WAIT_OP;
        wait_ptr->b.size_in_bytes = sizeof(wait_op_t);
        wait_ptr->tileLoc = mem.getTileLoc();
        wait_ptr->channelNum = mem.getDMAChConfig().channel;
        wait_ptr->dma_direction =  (XAie_DmaDirection)mem.getDMAChConfig().S2MMOrMM2S;
        op_ptr_ = (op_base*)wait_ptr;
    }
    
    virtual ~wait_op(){        
        wait_op_t * wait_ptr = reinterpret_cast<wait_op_t*>(op_ptr_);
        delete wait_ptr;
    }
    virtual string type() const override { return "wait_op";}
};



class transaction_op: public instr_base{
public:
    transaction_op() = delete;

    transaction_op( void * txn, bool debug=false)
    {
        XAie_TxnHeader *Hdr = (XAie_TxnHeader *)txn;
        
        if (debug) {
            printf("Header version %d.%d\n", Hdr->Major, Hdr->Minor);
            printf("Device Generation: %d\n", Hdr->DevGen);
            printf("Cols, Rows, NumMemRows : (%d, %d, %d)\n", Hdr->NumCols, Hdr->NumRows, Hdr->NumMemTileRows);
            printf("TransactionSize: %u\n", Hdr->TxnSize);
            printf("NumOps: %u\n", Hdr->NumOps);
        }

        transaction_op_t * tptr = new transaction_op_t();
        tptr->b.type = e_TRANSACTION_OP;
        tptr->b.size_in_bytes = sizeof(transaction_op_t) + Hdr->TxnSize;

        cmdBuf_ = new uint8_t[Hdr->TxnSize];
        memcpy(cmdBuf_, txn, Hdr->TxnSize);
        op_ptr_ = (op_base*)tptr;
        TxnSize = Hdr->TxnSize;
    }

    virtual void serialize( void * ptr) const override
    {
        memcpy ( ptr, op_ptr_, sizeof(transaction_op_t) );
        ptr = (char*) ptr + sizeof(transaction_op_t);
        memcpy ( ptr, cmdBuf_, TxnSize);
    }

    virtual ~transaction_op()
    {
        transaction_op_t * tptr = reinterpret_cast<transaction_op_t*>(op_ptr_);
        delete tptr;
        if (cmdBuf_) delete[] cmdBuf_;
    }
    virtual string type() const override { return "transaction_op";}
private:
    uint8_t* cmdBuf_;
    uint32_t TxnSize;
};

class pendingBDCount_op : public instr_base {
public:
    pendingBDCount_op() = delete;
    pendingBDCount_op(XAie_LocType tileLoc, short channelNum, XAie_DmaDirection dma_direction, u8 pendingBDThres = 0)
    {
        pendingBDCount_op_t * p = new pendingBDCount_op_t();
        p->b.type = e_PENDINGBDCOUNT_OP;
        p->b.size_in_bytes = sizeof(pendingBDCount_op_t);
        p->tileLoc = tileLoc;
        p->channelNum = channelNum;
        p->dma_direction = dma_direction;
        p->pendingBDThres = pendingBDThres;
        op_ptr_ = (op_base*)p;

    }
    virtual ~pendingBDCount_op() {
      delete reinterpret_cast<pendingBDCount_op_t*>(op_ptr_);
    }
    virtual string type() const override { return "PENDINGBDCOUNT_OP"; }
};

class patchBD_op : public instr_base {
  patchBD_op() = delete;
  patchBD_op(u32 action, u64 argidx, u64 argplus) {
    patch_op_t * p = new patch_op_t();
    p->b.type = e_PATCHBD_OP;
    p->b.size_in_bytes = sizeof(patch_op_t);
    p->action = action;
    p->argidx = argidx;
    p->argplus = argplus;
    op_ptr_ = (op_base*)p;
  }
  virtual ~patchBD_op() {
    delete reinterpret_cast<patch_op_t*>(op_ptr_);
  }
  virtual string type() const override { return "PATCHBD_OP"; }
};

};
#endif
