/*
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef __OP_BASE_H__
#define __OP_BASE_H__

#ifdef __cplusplus
extern "C" {
#endif    
#include <xaiengine.h>
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define GENERATE_ENUM(ENUM) e_##ENUM,        


enum op_types {
    OP_LIST(GENERATE_ENUM)
};

typedef struct{
    enum op_types type;
    unsigned int size_in_bytes;
} op_base;



#define GENERATE_FUNC_SIGNATURE(INSTRNAME) int op_##INSTRNAME##_func(XAie_DevInst*, op_base *, u8, const u8 *args);

//overload this macro for other printf functions for different platforms
#define TOGETHERWEADVANCE_printf(...) printf(__VA_ARGS__)

OP_LIST( GENERATE_FUNC_SIGNATURE )

#ifdef __cplusplus
}
#endif

#endif 
