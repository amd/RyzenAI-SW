/*
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef __OP_BUF_HPP__
#define __OP_BUF_HPP__

#include <vector>
#include <ps/op_init.hpp>

namespace aiectrl{

class op_buf
{

    public:
        op_buf(){}
        ~op_buf(){}

        void addOP( const instr_base & instr)
        {
            unsigned ibuf_sz = ibuf_.size();
            //std::cout << "OP TYPE: " << instr.type() << " instr size: " << instr.size() << " ibuf size: " << ibuf_.size() << std::endl;
            ibuf_.resize(ibuf_sz + instr.size() );
            instr.serialize ( (void*)&ibuf_[ibuf_sz] );
            //memcpy ( &ibuf_[ibuf_sz], op_ptr, op_ptr->size_in_bytes);
            //std::cout << "ibuf size: " << ibuf_.size() << std::endl;
        }

        size_t size() const{
            return ibuf_.size();
        }

        const void * data() const{
            return ibuf_.data();
        }
        std::vector<uint8_t> ibuf_;
};

};


#endif