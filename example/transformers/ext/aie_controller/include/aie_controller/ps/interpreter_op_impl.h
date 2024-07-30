/*
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include "op_types.h"
#include <assert.h>

#ifdef __AIESIM__
#define __AIESIM_TCT__
#include "aie_tct_aiesim.h"

using namespace aiesim_tct;

static struct aie_tct expected_tcts[16];
static uint32_t tct_count[XAIE_NUM_COLS] = {0};
static const uint32_t TCT_COUNT_ARR_SIZE = sizeof(tct_count)/sizeof(tct_count[0]);
static uint8_t tct_map[XAIE_NUM_COLS*XAIE_NUM_ROWS*AIE_TCT_MAX_ACTORID] = { 0 };
static const uint32_t MAP_SIZE = sizeof(tct_map)/sizeof(tct_map[0]);

int SubmitSerializedTransaction(XAie_DevInst* dev_inst, uint8_t *ptr, const uint8_t start_col_idx, uint8_t *args)
{
    XAie_TxnHeader txn_header = *((XAie_TxnHeader *)ptr);
    printf("Header version %d.%d\n", txn_header.Major, txn_header.Minor);
    printf("Device Generation: %d\n", txn_header.DevGen);
    printf("Cols, Rows, NumMemRows : (%d, %d, %d)\n", txn_header.NumCols,
         txn_header.NumRows, txn_header.NumMemTileRows);
    printf("TransactionSize: %u\n", txn_header.TxnSize);
    printf("NumOps: %u\n", txn_header.NumOps);
    ptr += sizeof(XAie_TxnHeader);

    for(uint32_t i = 0; i < txn_header.NumOps; i++) {
        XAie_OpHdr *op_header = (XAie_OpHdr *)ptr;

        switch(op_header->Op) {
            case XAIE_IO_WRITE: {
                XAie_Write32Hdr *w_header = (XAie_Write32Hdr *)ptr;
                printf("W: 0x%lx, 0x%x\n",
                        w_header->RegOff + dev_inst->BaseAddr,
                        w_header->Value);
                ess_Write32((u64)w_header->RegOff + (u64)dev_inst->BaseAddr, w_header->Value);
                ptr += w_header->Size;
                break;
            }
            case XAIE_IO_BLOCKWRITE: {
                XAie_BlockWrite32Hdr* bw_header = (XAie_BlockWrite32Hdr *)ptr;
                uint32_t* payload = (uint32_t*)(ptr + sizeof(XAie_BlockWrite32Hdr));
                u32 size = (bw_header->Size - sizeof(*bw_header)) / 4;
                printf("BW: \n");
                for(uint32_t i = 0; i < size; i++) {
                    uint64_t addr = bw_header->RegOff + dev_inst->BaseAddr + i*4U;
                    printf("   0x%lx, 0x%x\n", addr, payload[i]);
                    ess_Write32(addr, payload[i]);
                }
                ptr += bw_header->Size;
                break;
            }
            case XAIE_IO_MASKWRITE: {
                XAie_MaskWrite32Hdr *mw_header = (XAie_MaskWrite32Hdr *)ptr;
                printf("MW: 0x%lx, 0x%x, 0x%x\n",
                        mw_header->RegOff + dev_inst->BaseAddr,
                        mw_header->Mask, mw_header->Value);
                u32 regval = ess_Read32((u64)mw_header->RegOff + (u64)dev_inst->BaseAddr);
                ess_Write32((u64)mw_header->RegOff + (u64)dev_inst->BaseAddr, \
                    (regval & (~(mw_header->Mask))) | (mw_header->Mask & mw_header->Value));
                ptr += mw_header->Size;
                break;
            }
            case XAIE_IO_MASKPOLL: {
                XAie_MaskPoll32Hdr *mp_header = (XAie_MaskPoll32Hdr *)ptr;
                printf("MP: 0x%lx, 0x%x, 0x%x\n",
                        mp_header->RegOff + dev_inst->BaseAddr,
                        mp_header->Mask, mp_header->Value);
                u32 regval = ess_Read32((u64)mp_header->RegOff + (u64)dev_inst->BaseAddr);
                while ((regval & mp_header->Mask) != mp_header->Value) {
                    regval = ess_Read32((u64)mp_header->RegOff + (u64)dev_inst->BaseAddr);
                }
                ptr += mp_header->Size;
                break;
            }
            case XAIE_IO_CUSTOM_OP_BEGIN: {
                XAie_CustomOpHdr *co_header = (XAie_CustomOpHdr *)ptr;
                tct_op_t *iptr = (tct_op_t *)(ptr + sizeof(*co_header));
                printf("CustomOp TCT: %d\n", iptr->word);
                u32 word = iptr->word;
                u8 Col = ((word & 0x00FF0000) >> 16) + start_col_idx;
                u8 Row = ((word & 0x0000FF00) >> 8);
                u8 dir = ((word)& 0x000000FF);
                XAie_DmaDirection Dir = (dir == 0) ? DMA_S2MM : DMA_MM2S;
                u32 config = iptr->config;
                u8 ChNum = ((config & 0xFF000000) >> 24);
                u8 ColNum = ((config & 0x00FF0000) >> 16);
                u8 RowNum = ((config & 0x0000FF00) >> 8);
                printf("SyncTaskCompleteToken: {col, row, chl, dir} = {%d+%d, %d+%d, %d, %d}\n", Col,
                    ColNum, Row, RowNum, ChNum, Dir);
#define DEBUGPRINT 1
                for (u8 c = 0; c < ColNum; c++) {
                    u8 nCol = Col + ColNum - 1 - c;
                    uint32_t num_tct = 0;
                    for (u8 r = 0; r < RowNum; r++) {
                        u8 nRow = Row + r;
                        expected_tcts[num_tct++] = aie_tct_create(Dir*6 + ChNum, nRow, nCol);
                    }

#if STRIX == 1 || STRIX_B0 == 1
                    // tile column only [0,3] --->mapping tct 0~3
                    uint32_t fifo_id = nCol;
#else
                    // tile column only [1,4] --->mapping tct 0~3
                    uint32_t fifo_id = nCol-1;
#endif

#ifdef DEBUGPRINT
                    printf("-------------------------\n");
                    printf("FIFO:%d  ",fifo_id);
                    for (uint32_t k = 0; k < num_tct; k++)
                        printf("%d.%x ", expected_tcts[k].valid, expected_tcts[k].tct);

                    printf("MAP before:%d ", fifo_id);
                    for (uint32_t k = 0; k < MAP_SIZE; k++)
                        if (tct_map[k] !=0)
                            printf("%d.%x ", k, tct_map[k]) ;
                    printf("\n");
#endif // DEBUGPRINT

                    aie_tct_map_wait(tct_map, fifo_id, expected_tcts, num_tct);
                    tct_count[fifo_id] += num_tct;

#ifdef DEBUGPRINT
                    printf("MAP after:%d ", fifo_id);
                    for (uint32_t k = 0; k < MAP_SIZE; k++)
                        if (tct_map[k] !=0)
                            printf("%d.%x ", k, tct_map[k]) ;
                    printf("\n");
#endif // DEBUGPRINT
                }

                printf("TCT_COUNT: ");
                for (uint32_t k = 0; k < TCT_COUNT_ARR_SIZE; k++)
                    printf("%d ", tct_count[k]) ;
                printf("\n");
                printf("SyncTaskCompleteToken over\n");
                ptr += co_header->Size;
                break;
            }
            case XAIE_IO_CUSTOM_OP_BEGIN+1: {
                XAie_CustomOpHdr *hdr = (XAie_CustomOpHdr *)ptr;
                patch_op_t *op = (patch_op_t *)(ptr + sizeof(*hdr));
                printf("CustomOp PatchBD argidx %d\n", op->argidx);
                u64 *argv = (u64*)args;
                printf("CustomOp PatchBD regaddr %lx\n", op->regaddr+dev_inst->BaseAddr);
                u64 tensorAddr = argv[op->argidx] + op->argplus;
                printf("CustomOp PatchBD addr val %llu\n", tensorAddr);
                if (tensorAddr != 0) {
                  // patch if arg provided
                  // 1. write lower 32 bits
                  ess_Write32(op->regaddr+dev_inst->BaseAddr, 
                    tensorAddr & 0xFFFFFFFFC); // unused 2-LSB
                  
                  // 2. read upper 32 bits 
                  u32 regval = ess_Read32(op->regaddr+dev_inst->BaseAddr+4u);
                  // 3. mask-write upper 32 bits
                  u32 uppermask = 0x0000FFFF; 
                  ess_Write32(op->regaddr+dev_inst->BaseAddr+4u, 
                      (regval & (~uppermask)) | (tensorAddr >> 32));
                }

                ptr += hdr->Size;
                break;
            }
            case XAIE_IO_CUSTOM_OP_BEGIN+2: {
                // Dump Registers opcode
                // Do nothing in sim
                break;
            }
            default:
                return -1;
        }
    }
    return 0;
}
#else // Simnow, Silicon

#include "profile_impl.h"

static void __attribute__((section(".app_critical_text"))) Write32Transaction(uint8_t **ptr, const uint64_t base_addr);
static void __attribute__((section(".app_critical_text"))) MaskWriteTransaction(uint8_t **ptr, const uint64_t base_addr);
static void __attribute__((section(".app_critical_text"))) MaskPollTransaction(uint8_t **ptr, const uint64_t base_addr);
static void __attribute__((section(".app_critical_text"))) BlockWrite32Transaction(uint8_t **ptr, const uint64_t base_addr);
static void __attribute__((section(".app_critical_text"))) SyncTaskCompleteToken(uint8_t **ptr, const uint64_t base_addr, const uint8_t start_col_idx);
static int __attribute__((section(".app_critical_text"))) PatchOpTransaction(uint8_t **ptr, const uint64_t base_addr, const uint8_t **args);

static void Write32Transaction(uint8_t **ptr, const uint64_t base_addr) {
    XAie_Write32Hdr* w_header = (XAie_Write32Hdr *)(*ptr);
    *((volatile u32 *)(w_header->RegOff + base_addr)) = w_header->Value;
    *ptr = *ptr + w_header->Size;
}

static void BlockWrite32Transaction(uint8_t **ptr, const uint64_t base_addr) {
    XAie_BlockWrite32Hdr* bw_header = (XAie_BlockWrite32Hdr *)(*ptr);
    u32 reg_addr = bw_header->RegOff + base_addr;
    u32 bw_size = bw_header->Size;
    u32 Size = (bw_size - sizeof(*bw_header)) / 4;
    u32 *Payload =(u32 *) ((*ptr) + sizeof(*bw_header));
    for (u32 i = 0; i < Size; i++) {
        *((volatile u32 *)(reg_addr + i * 4)) = *Payload;
        Payload++;
    }
    *ptr = *ptr + bw_size;
}

static void MaskWriteTransaction(uint8_t **ptr, const uint64_t base_addr) {
    XAie_MaskWrite32Hdr* mw_header = (XAie_MaskWrite32Hdr *)(*ptr);
    volatile u32* reg = (volatile u32*)(mw_header->RegOff + base_addr);
    *reg = (*reg & (~mw_header->Mask)) | (mw_header->Mask & mw_header->Value);
    *ptr = *ptr + mw_header->Size;
}

static void MaskPollTransaction(uint8_t **ptr, const uint64_t base_addr) {
    XAie_MaskPoll32Hdr * mp_header = (XAie_MaskPoll32Hdr *)(*ptr);
    volatile u32* reg = (volatile u32*)(mp_header->RegOff + base_addr);
    printf("MP: 0x%lx, 0x%x, 0x%x\n",
                mp_header->RegOff + base_addr,
                mp_header->Mask, mp_header->Value);
    while ((*reg & mp_header->Mask) != mp_header->Value);
    *ptr = *ptr + mp_header->Size;
}

static void SyncTaskCompleteToken(uint8_t **ptr, const uint64_t base_addr, const uint8_t start_col_idx) {
    (void)base_addr;
    XAie_CustomOpHdr *co_header = (XAie_CustomOpHdr *)(*ptr);
    printf("co_header->Size = %d\n", co_header->Size);
    volatile tct_op_t *iptr = ((volatile tct_op_t *)((*ptr) + sizeof(*co_header)));
    u32 word = iptr->word;
    u8 Col = ((word & 0x00FF0000) >> 16) + start_col_idx;
    u8 Row = ((word & 0x0000FF00) >> 8);
    u8 dir = ((word)& 0x000000FF);
    XAie_DmaDirection Dir = (dir == 0) ? DMA_S2MM : DMA_MM2S;
    u32 config = iptr->config;
    u8 ChNum = ((config & 0xFF000000) >> 24);
    u8 ColNum = ((config & 0x00FF0000) >> 16);
    u8 RowNum = ((config & 0x0000FF00) >> 8);
    printf("SyncTaskCompleteToken: {col, row, chl, dir} = {%d+%d, %d+%d, %d, %d}\n", Col,
           ColNum, Row, RowNum, ChNum, Dir);
    *ptr = *ptr + co_header->Size;
#define DEBUGPRINT 1
    for (u8 c = 0; c < ColNum; c++) {
        u8 nCol = Col + ColNum - 1 - c;
        uint32_t num_tct = 0;
        for (u8 r = 0; r < RowNum; r++) {
            u8 nRow = Row + r;
            expected_tcts[num_tct++] = aie_tct_create(Dir*6 + ChNum, nRow, nCol);
        }

#if STRIX == 1 || STRIX_B0 == 1
        // tile column only [0,3] --->mapping tct 0~3
        uint32_t fifo_id = nCol;
#else
        // tile column only [1,4] --->mapping tct 0~3
        uint32_t fifo_id = nCol-1;
#endif

#ifdef DEBUGPRINT
        printf("-------------------------\n");
        printf("FIFO:%d  ",fifo_id);
        for (uint32_t k = 0; k < num_tct; k++)
        printf("%d.%x ", expected_tcts[k].valid, expected_tcts[k].tct);

        printf("MAP before:%d ", fifo_id);
        for (uint32_t k = 0; k < MAP_SIZE; k++)
        if (tct_map[k] !=0)
            printf("%d.%x ", k, tct_map[k]) ;
        printf("\n");
#endif // DEBUGPRINT

        aie_tct_map_wait(tct_map, fifo_id, expected_tcts, num_tct);
        tct_count[fifo_id] += num_tct;

#ifdef DEBUGPRINT
        printf("MAP after:%d ", fifo_id);
        for (uint32_t k = 0; k < MAP_SIZE; k++)
        if (tct_map[k] !=0)
            printf("%d.%x ", k, tct_map[k]) ;
        printf("\n");
#endif // DEBUGPRINT
    }

    printf("TCT_COUNT: ");
        for (uint32_t k = 0; k < TCT_COUNT_ARR_SIZE; k++)
    printf("%d ", tct_count[k]) ;
    printf("\n");
    printf("SyncTaskCompleteToken over\n");
}

static int PatchOpTransaction(uint8_t **ptr, const uint64_t base_addr, const uint8_t **args) {
  XAie_CustomOpHdr *Hdr = (XAie_CustomOpHdr *)(*ptr);
  u32 size = Hdr->Size;
  volatile patch_op_t *op = ((volatile patch_op_t *)((*ptr) + sizeof(*Hdr)));

  const u64 *argv = (const u64*)(*args);

  if (op->argidx > 4) {
    // out of bound access
    printf("argidx is out of bounds\n");
    return -1;
  }

  u64 tensorAddr = argv[op->argidx] + op->argplus;

  if (tensorAddr != 0) { // patch if arg provided
    // 1. write lower 32 bits
    *((volatile u32 *)(op->regaddr + base_addr)) 
      = tensorAddr & 0xFFFFFFFFC; // unused 2-LSB

    // 2. mask-write upper 32 bits
    u32 uppermask = 0x0000FFFF; 
    volatile u32* reg = (volatile u32*)(op->regaddr + base_addr + 4u);
    *reg = (*reg & (~uppermask)) | (tensorAddr >> 32);
  }

  *ptr = *ptr + size;
  return 0;
}

static void DumpRegisters(uint8_t **ptr, const uint64_t base_addr) {
    XAie_CustomOpHdr *Hdr = (XAie_CustomOpHdr *)(*ptr);
    u32 size = Hdr->Size;
    read_register_op_t *op = ((read_register_op_t *)((*ptr) + sizeof(*Hdr)));

    DumpRegistersImpl(base_addr, op->count, (const uint64_t*)(&(op->data[0].address)));

    *ptr = *ptr + size;
}

static void RecordTimestamp(uint8_t **ptr, const uint64_t base_addr) {
    XAie_CustomOpHdr *Hdr = (XAie_CustomOpHdr *)(*ptr);
    u32 size = Hdr->Size;

    u32 payload_size = size - sizeof(*Hdr);
    if (payload_size != 4) {
        *ptr = *ptr + size;
        return;
    }
    record_timer_op_t* op = (record_timer_op_t *)((*ptr) + sizeof(*Hdr));
    RecordTimestampImpl(base_addr, op->id);
    *ptr = *ptr + size;
}

int __attribute__((section(".app_critical_text"))) SubmitSerializedTransaction(XAie_DevInst* dev_inst, uint8_t *ptr, const uint8_t start_col_idx, const uint8_t *args)
{
    XAie_TxnHeader *txn_header = ((XAie_TxnHeader *)ptr);
#ifdef DEBUGPRINT
    printf("Header version %d.%d\n", txn_header->Major, txn_header->Minor);
    printf("Device Generation: %d\n", txn_header->DevGen);
    printf("Cols, Rows, NumMemRows : (%d, %d, %d)\n", txn_header->NumCols,
         txn_header->NumRows, txn_header->NumMemTileRows);
    printf("TransactionSize: %d\n", txn_header->TxnSize);
    printf("NumOps: %d\n", txn_header->NumOps);
#endif
    ptr += sizeof(XAie_TxnHeader);
    const uint64_t base_addr = (const uint64_t) dev_inst->BaseAddr;
    uint32_t NumOps = txn_header->NumOps;
    int rv = 0;
    XAie_OpHdr* op_header;
    for(uint32_t i = 0; i < NumOps; i++) {
        op_header = (XAie_OpHdr*)ptr;
        switch(op_header->Op) {
            case XAIE_IO_WRITE:
                Write32Transaction(&ptr, base_addr);
                break;
            case XAIE_IO_BLOCKWRITE:
                BlockWrite32Transaction(&ptr, base_addr);
                break;
            case XAIE_IO_MASKWRITE:
                MaskWriteTransaction(&ptr, base_addr);
                break;
            case XAIE_IO_MASKPOLL:
                MaskPollTransaction(&ptr, base_addr);
                break;
            case XAIE_IO_CUSTOM_OP_BEGIN:
                SyncTaskCompleteToken(&ptr, base_addr, start_col_idx);
                break;
            case XAIE_IO_CUSTOM_OP_BEGIN+1:
                rv = PatchOpTransaction(&ptr, base_addr, &args);
                if (rv != 0) {
                    return rv;
                }
                break;
            case XAIE_IO_CUSTOM_OP_BEGIN+2:
                DumpRegisters(&ptr, base_addr);
                break;
            case XAIE_IO_CUSTOM_OP_BEGIN+3:
                RecordTimestamp(&ptr, base_addr);
                break;
            default:
                printf("Incorrect Opcode\n");
                return -1;
        }
    }
    
    return rv;
}
#endif // __AIESIM__

static inline u8 ConvertLogicalToPhysicalDMAChNum(short logical_ch_num)
{
    return (logical_ch_num > 1 ? (logical_ch_num - 2) : logical_ch_num);
}

int op_WAIT_OP_func(XAie_DevInst* dev_inst , op_base * ptr, uint8_t start_col_idx, const u8 *args)
{
    (void)start_col_idx;
    (void)args;
    wait_op_t * instr = (wait_op_t*) ptr;
    // std::cout << "op_wait_op_func " 
    //           << instr->b.type << " " 
    //           << instr->b.size_in_bytes << " " 
    //           << (uint32_t)(instr->tileLoc.Row) << "," << (uint32_t)(instr->tileLoc.Col) << " "
    //           << instr->channelNum << " " 
    //           << instr->dma_direction << " " 
    //           << std::endl;
    
    // Only convert logical to physical for Shim tile channels
    u8 channel_num = instr->tileLoc.Row != 0 ? instr->channelNum : ConvertLogicalToPhysicalDMAChNum(instr->channelNum);
    while (XAie_DmaWaitForDone(dev_inst, instr->tileLoc, channel_num, instr->dma_direction, 0) != XAIE_OK);
    return 0;
}

int op_PENDINGBDCOUNT_OP_func(XAie_DevInst* dev_inst, op_base* ptr, uint8_t start_col_idx, const u8 *args)
{
    (void)start_col_idx;
    (void)args;
    pendingBDCount_op_t* instr = (pendingBDCount_op_t*)ptr;
    u8 num_pending_bds;

    // Only convert logical to physical for Shim tile channels
    u8 channel_num = instr->tileLoc.Row != 0 ? instr->channelNum : ConvertLogicalToPhysicalDMAChNum(instr->channelNum);
    do
    {
        XAie_DmaGetPendingBdCount(dev_inst, instr->tileLoc, channel_num, instr->dma_direction, &num_pending_bds);
    } while (num_pending_bds > instr->pendingBDThres);
    return 0;
}

int op_TRANSACTION_OP_func(XAie_DevInst* dev_inst , op_base * ptr, uint8_t start_col_idx, const u8 *args)
{
    uint8_t* txn = (uint8_t*)((unsigned char*)ptr + sizeof(transaction_op_t));
    return SubmitSerializedTransaction(dev_inst, txn, start_col_idx, args);
}

int op_DBGPRINT_OP_func(XAie_DevInst* dev_inst, op_base* ptr, uint8_t start_col_idx, const u8 *args)
{
    (void)dev_inst;
    (void)start_col_idx;
    (void)args;
    print_op_t* p = (print_op_t*)ptr;
    TOGETHERWEADVANCE_printf("%s\n", p->msg);
    return 0;
}

int op_PATCHBD_OP_func(XAie_DevInst* dev_inst, op_base* ptr, uint8_t start_col_idx, const u8 *args)
{
  (void)dev_inst;
  (void)start_col_idx;
  (void)args;
  (void)ptr;
  // TODO standalone op not implemented. Never support?
  return 0;
}
