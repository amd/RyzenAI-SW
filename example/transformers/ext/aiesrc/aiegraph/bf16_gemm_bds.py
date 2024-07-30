#from bdgenerator import generate_bds, DataTransfer, TransferParams, AieTile, TileType, DmaChannel, DmaDir
import sys, os
sys.path.append(os.getcwd()+'/dmacompiler')

from dmacompiler import DataTransfer, TransferParams, AieTile, TileType, DmaChannel, DmaDir, OverlayShape, compile_dma_config, SyncStrategy

#
# GeMM Shape parameters
#

Mgemm = 1;
Kgemm = 4096;
Ngemm = 4096;
ShareIfmChannel = False 
if Mgemm == 1:
    ShareIfmChannel = True 
else:
    ShareIfmChannel = False 

#
# Static constants (do not modify)
#

NUM_ROWS = 4
NUM_COLS = 4

START_COL = 0
START_ROW = 0

#
# Actual sub vol dim computed 
#
M_SUBV = 8;
K_SUBV = 32;
N_SUBV = 128;
GRP_SIZE = 32;

#
# Max-Sub vol dim that the core will support 
#
MAX_M_SUBV = 8
MAX_K_SUBV = 128
MAX_N_SUBV = 128
MAX_GRP_SIZE = 128

#
# Check feasibility of the GeMM dimensions
#
assert MAX_M_SUBV == M_SUBV
assert MAX_N_SUBV == N_SUBV
assert MAX_K_SUBV >= K_SUBV
assert MAX_GRP_SIZE >= GRP_SIZE
assert MAX_K_SUBV == MAX_GRP_SIZE
assert K_SUBV == GRP_SIZE
assert Kgemm % K_SUBV == 0
assert Ngemm % (NUM_ROWS * NUM_COLS * N_SUBV) == 0

print(f"// Channnel sharing: ", ShareIfmChannel)
print(f"// Mgemm: ", Mgemm)
print(f"// Kgemm: ", Kgemm)
print(f"// Ngemm: ", Ngemm)

print(f"// Max-Sub vol dim that the core will support:")
print(f"// M_SUBV: ", MAX_M_SUBV)
print(f"// K_SUBV: ", MAX_K_SUBV)
print(f"// N_SUBV: ", MAX_N_SUBV)
print(f"// GRP_SIZE: ", MAX_GRP_SIZE)

print(f"// Actual sub vol dim computed:")
print(f"// M_SUBV: ", M_SUBV)
print(f"// K_SUBV: ", K_SUBV)
print(f"// N_SUBV: ", N_SUBV)
print(f"// GRP_SIZE: ", GRP_SIZE)

#
# Compute the L1 allocations for the max supported subvol dims
#
ALIGN = 64
CORE_IN1_SIZE = MAX_M_SUBV * MAX_K_SUBV * 2
CORE_IN2_SIZE = (((MAX_K_SUBV * MAX_N_SUBV // 2
                   + MAX_K_SUBV * MAX_N_SUBV // MAX_GRP_SIZE // 2
                   + MAX_K_SUBV * MAX_N_SUBV // MAX_GRP_SIZE * 2) + ALIGN - 1) // ALIGN) * ALIGN
CORE_OUT_SIZE = MAX_M_SUBV * MAX_N_SUBV * 4

CORE_IN1_WORDS = CORE_IN1_SIZE // 4
CORE_IN2_WORDS = CORE_IN2_SIZE // 4
CORE_OUT_WORDS = CORE_OUT_SIZE // 4

#NOTE: The Data in1 size is the unpadded subvol dimension from L3->L2
# Has to be the minimum of the Mgemm (for token phase) and M_SUBV (for prefill phase)
# The K dim and grp_size is padded to the max supported dim of the L1
unpadded_M_SUBV = (M_SUBV if Mgemm > M_SUBV else Mgemm)
DATA_IN1_SIZE = (unpadded_M_SUBV * K_SUBV * 2)
DATA_IN2_SIZE = (((K_SUBV * N_SUBV // 2
                   + K_SUBV * N_SUBV // GRP_SIZE // 2
                   + K_SUBV * N_SUBV // GRP_SIZE * 2) + ALIGN - 1) // ALIGN) * ALIGN
DATA_IN1_WORDS = DATA_IN1_SIZE // 4
DATA_IN2_WORDS = DATA_IN2_SIZE // 4
DATA_OUT_WORDS = (unpadded_M_SUBV * N_SUBV * 4) // 4

INNER_LOOP = Kgemm // K_SUBV
OUTER_M_LOOP = (Mgemm // M_SUBV if Mgemm > M_SUBV else 1)
OUTER_N_LOOP = Ngemm // (NUM_ROWS * NUM_COLS * N_SUBV)
print(f"// INNER_LOOP: ", INNER_LOOP)
print(f"// OUTER_M_LOOP: ", OUTER_M_LOOP)
print(f"// OUTER_N_LOOP: ", OUTER_N_LOOP)

IN1_REPEAT = (OUTER_M_LOOP * OUTER_N_LOOP * INNER_LOOP) + 1
IN2_REPEAT = (OUTER_M_LOOP * OUTER_N_LOOP * (INNER_LOOP + 1))
OUT_REPEAT = OUTER_M_LOOP * OUTER_N_LOOP

#
# Overlay 
#
shape = OverlayShape(NUM_COLS, NUM_ROWS, START_COL, START_ROW)

#
# BD generators
#

def generate_prefill_bds():
    #NOTE: 1 sub vol of param is streamed into the L2 and L1 at the beginning of the layer
    PARAM_SIZE = CORE_IN1_SIZE
    #NOTE: IFM of M_SUBV * Kgemm is pinned in L2 for reuse of Ngemm/(N_SUBV*NUM_ROWS*NUM_COLS)
    IFM_SIZE = (M_SUBV * Kgemm * 2)
    IFM_S2MM_WORDS = IFM_SIZE // 4
    IFM_MM2S_WORDS = INNER_LOOP * CORE_IN1_WORDS 
    IFM_MM2S_BD_OFFSET = IFM_S2MM_WORDS // 4
    IFM_MM2S_BD_WORDS = IFM_MM2S_WORDS // 4
    IFM_REUSE_RATIO = OUTER_N_LOOP
    WGT_REPEAT = (OUTER_N_LOOP * (INNER_LOOP + 1))

    WGT_SIZE = NUM_ROWS * DATA_IN2_SIZE

    # L2 buffer allocations
    # NOTE: Address calculations have to be in bytes
    # All buffer placements are relative to each other
    PARAM_PING_ADDR = 0 
    IFM_PING_ADDR = 0 + PARAM_PING_ADDR + CORE_IN1_SIZE

    WGT_BUFF_1_ADDR = 0 + IFM_PING_ADDR + IFM_SIZE
    WGT_BUFF_2_ADDR = 0 + WGT_BUFF_1_ADDR + WGT_SIZE

    OFM_PING_ADDR = 0 + WGT_BUFF_2_ADDR + WGT_SIZE
    OFM_PONG_ADDR = 0 + OFM_PING_ADDR + (NUM_ROWS * CORE_OUT_SIZE)
    memtile_transfers = [
        # Param 1 --> 1 core input 1
         DataTransfer(
             [AieTile(TileType.Memtile, col, 0) for col in range(NUM_COLS)],
             CORE_IN1_SIZE, [PARAM_PING_ADDR],
             [1]+[0]*(OUTER_M_LOOP-1),
             [TransferParams(DmaChannel(DmaDir.S2MM, 0), DATA_IN1_WORDS)],
             [TransferParams(DmaChannel(DmaDir.MM2S, 0), CORE_IN1_WORDS)]
         ),
        # IFM 1 --> 1 core input 1
         DataTransfer(
             [AieTile(TileType.Memtile, col, 0) for col in range(NUM_COLS)],
             IFM_SIZE, [IFM_PING_ADDR],
             [1]*OUTER_M_LOOP,
             [TransferParams(DmaChannel(DmaDir.S2MM, 0), IFM_S2MM_WORDS,
                             step=[1, (M_SUBV * K_SUBV // 2), (K_SUBV // 2)],
                             wrap=[(K_SUBV // 2), (Kgemm // K_SUBV)])],
             [
                 TransferParams(DmaChannel(DmaDir.MM2S, 0), IFM_MM2S_BD_WORDS, offset=(0 * IFM_MM2S_BD_OFFSET), step=[1, DATA_IN1_WORDS], wrap=[CORE_IN1_WORDS])
                ,TransferParams(DmaChannel(DmaDir.MM2S, 0), IFM_MM2S_BD_WORDS, offset=(1 * IFM_MM2S_BD_OFFSET), step=[1, DATA_IN1_WORDS], wrap=[CORE_IN1_WORDS])
                ,TransferParams(DmaChannel(DmaDir.MM2S, 0), IFM_MM2S_BD_WORDS, offset=(2 * IFM_MM2S_BD_OFFSET), step=[1, DATA_IN1_WORDS], wrap=[CORE_IN1_WORDS])
                ,TransferParams(DmaChannel(DmaDir.MM2S, 0), IFM_MM2S_BD_WORDS, offset=(3 * IFM_MM2S_BD_OFFSET), step=[1, DATA_IN1_WORDS], wrap=[CORE_IN1_WORDS])
             ],
             reuse_ratio=IFM_REUSE_RATIO
         ),
         # 1 --> 4 core input 2
         DataTransfer(
             [AieTile(TileType.Memtile, col, 0) for col in range(NUM_COLS)],
             WGT_SIZE, [WGT_BUFF_1_ADDR, WGT_BUFF_2_ADDR],
             [WGT_REPEAT]*OUTER_M_LOOP,
             [TransferParams(DmaChannel(DmaDir.S2MM, 1), NUM_ROWS * DATA_IN2_WORDS)],
             [
                 TransferParams(DmaChannel(DmaDir.MM2S, 1), CORE_IN2_WORDS, offset=(0 * DATA_IN2_WORDS)),
                 TransferParams(DmaChannel(DmaDir.MM2S, 2), CORE_IN2_WORDS, offset=(1 * DATA_IN2_WORDS)),
                 TransferParams(DmaChannel(DmaDir.MM2S, 3), CORE_IN2_WORDS, offset=(2 * DATA_IN2_WORDS)),
                 TransferParams(DmaChannel(DmaDir.MM2S, 4), CORE_IN2_WORDS, offset=(3 * DATA_IN2_WORDS))
             ],
             sync_strategy=SyncStrategy.Parallel_1_to_N
         ),
         # 4 --> 1 core output
         DataTransfer(
             [AieTile(TileType.Memtile, col, 0) for col in range(NUM_COLS)],
             NUM_ROWS * CORE_OUT_SIZE, [OFM_PING_ADDR, OFM_PONG_ADDR], 
             [OUTER_N_LOOP]*OUTER_M_LOOP,
             [
                 TransferParams(DmaChannel(DmaDir.S2MM, 2), CORE_OUT_WORDS, offset=(0 * CORE_OUT_WORDS)),
                 TransferParams(DmaChannel(DmaDir.S2MM, 3), CORE_OUT_WORDS, offset=(1 * CORE_OUT_WORDS)),
                 TransferParams(DmaChannel(DmaDir.S2MM, 4), CORE_OUT_WORDS, offset=(2 * CORE_OUT_WORDS)),
                 TransferParams(DmaChannel(DmaDir.S2MM, 5), CORE_OUT_WORDS, offset=(3 * CORE_OUT_WORDS))
             ],
             [
                 TransferParams(DmaChannel(DmaDir.MM2S, 5), NUM_ROWS * DATA_OUT_WORDS, offset=0,
                                step=[1, (M_SUBV * N_SUBV), N_SUBV],
                                wrap=[N_SUBV, NUM_ROWS])
             ],
             sync_strategy=SyncStrategy.Parallel_N_to_1
        )
    ]
    code = compile_dma_config(shape, memtile_transfers, [])
    print(code)

def generate_token_bds():
    print(f"// Generating BDs for token phase ")
    # Compute shared IFM transfer constants 
    # NOTE: Each subvolume, the IFM BD reads a small chunk of the input
    #       and pads it up to the core buffer size with garbage bits.
    #       This is fine because the token GeMM will only read the
    #       valid input bits. The IFM BD only runs once, and relies on
    #       stream backpressure for the core subvolume iterations.
    IFM_SIZE = Mgemm * Kgemm * 2
    IFM_S2MM_WORDS = IFM_SIZE // 4
    # NOTE: IFM MM2S fetch from L2 -> L1 is padded upto the M dim of sub vol
    IFM_MM2S_WORDS = INNER_LOOP * CORE_IN1_WORDS 
    IFM_REUSE_RATIO = OUTER_N_LOOP
    #NOTE: For few of the shapes the MM2S transfer length might be too large to be handled by a single BD
    # The transfer is broken down into a chain of 8 BDs
    IFM_MM2S_BD_OFFSET = IFM_S2MM_WORDS // 4
    if (INNER_LOOP % 4) > 0: # This specifc K/K_SUBV dimension is not divisible by 4, chaining is disabled for this
        IFM_MM2S_BD_WORDS = IFM_MM2S_WORDS
    else:
        IFM_MM2S_BD_WORDS = IFM_MM2S_WORDS // 4
    # NOTE: The memtile buffer length register only has 17 bits
    if IFM_MM2S_BD_WORDS > 2**17:
        raise RuntimeError(f'IFM MM2S transfer length too high !', IFM_MM2S_BD_WORDS)

    # Compute shared WGT transfer constants
    #NOTE: DATA_IN2_SIZE is the actual (unpaded) 1 subvol size that is being computed
    WGT_SIZE = (NUM_ROWS // 2) * DATA_IN2_SIZE
    WGT_S2MM_WORDS = WGT_SIZE // 4
    #NOTE: CORE_IN2_SIZE is the amount of 1 subvol data (MAX supported) the core DMA expects
    WGT_MM2S_WORDS = ((NUM_ROWS // 2) * CORE_IN2_SIZE) // 4

    # L2 buffer allocations
    # NOTE: Address calculations have to be in bytes
    # All buffer placements are relative to each other
    PARAM_PING_ADDR = 0
    IFM_PING_ADDR = 0 + CORE_IN1_SIZE
    WGT_LOWER_BUFF_1_ADDR = 0 + IFM_PING_ADDR + IFM_SIZE
    WGT_UPPER_BUFF_1_ADDR = 0 + WGT_LOWER_BUFF_1_ADDR + WGT_SIZE

    WGT_LOWER_BUFF_2_ADDR = 0 + WGT_UPPER_BUFF_1_ADDR + WGT_SIZE
    WGT_UPPER_BUFF_2_ADDR = 0 + WGT_LOWER_BUFF_2_ADDR + WGT_SIZE

    OFM_PING_ADDR = 0 + WGT_UPPER_BUFF_2_ADDR + WGT_SIZE
    OFM_PONG_ADDR = 0 + OFM_PING_ADDR + (NUM_ROWS * CORE_OUT_SIZE)

    memtile_transfers = [
         # 1 --> 1 params
         DataTransfer(
             [AieTile(TileType.Memtile, col, 0) for col in range(NUM_COLS)],
             CORE_IN1_SIZE, [PARAM_PING_ADDR],
             [1],
             [TransferParams(DmaChannel(DmaDir.S2MM, 0), DATA_IN1_WORDS)],
             [TransferParams(DmaChannel(DmaDir.MM2S, 0), CORE_IN1_WORDS)]
         ),
         # 1 --> 1 ifm
          DataTransfer(
             [AieTile(TileType.Memtile, col, 0) for col in range(NUM_COLS)],
             IFM_SIZE, [IFM_PING_ADDR],
             [1],
             [TransferParams(DmaChannel(DmaDir.S2MM, 0), IFM_S2MM_WORDS)],
             # NOTE: the IFM MM2S data transfer is split into four BDs running on the same
             #       channel to avoid overflowing the buffer length register
             [
                TransferParams(DmaChannel(DmaDir.MM2S, 0), IFM_MM2S_BD_WORDS, offset=(0 * IFM_MM2S_BD_OFFSET), step=[1, DATA_IN1_WORDS], wrap=[CORE_IN1_WORDS])
             ]
             if (INNER_LOOP % 4) > 0 else
             [
                TransferParams(DmaChannel(DmaDir.MM2S, 0), IFM_MM2S_BD_WORDS, offset=(0 * IFM_MM2S_BD_OFFSET), step=[1, DATA_IN1_WORDS], wrap=[CORE_IN1_WORDS])
                ,TransferParams(DmaChannel(DmaDir.MM2S, 0), IFM_MM2S_BD_WORDS, offset=(1 * IFM_MM2S_BD_OFFSET), step=[1, DATA_IN1_WORDS], wrap=[CORE_IN1_WORDS])
                ,TransferParams(DmaChannel(DmaDir.MM2S, 0), IFM_MM2S_BD_WORDS, offset=(2 * IFM_MM2S_BD_OFFSET), step=[1, DATA_IN1_WORDS], wrap=[CORE_IN1_WORDS])
                ,TransferParams(DmaChannel(DmaDir.MM2S, 0), IFM_MM2S_BD_WORDS, offset=(3 * IFM_MM2S_BD_OFFSET), step=[1, DATA_IN1_WORDS], wrap=[CORE_IN1_WORDS])
             ],
             reuse_ratio=IFM_REUSE_RATIO
         ),
         # 1 --> 2 wgts lower
         DataTransfer(
             [AieTile(TileType.Memtile, col, 0) for col in range(NUM_COLS)],
             WGT_SIZE, [WGT_LOWER_BUFF_1_ADDR, WGT_LOWER_BUFF_2_ADDR],
             [IN2_REPEAT],
             [TransferParams(DmaChannel(DmaDir.S2MM, 0), WGT_S2MM_WORDS)],
             [
                 TransferParams(DmaChannel(DmaDir.MM2S, 1), CORE_IN2_WORDS, offset=(0 * DATA_IN2_WORDS)),
                 TransferParams(DmaChannel(DmaDir.MM2S, 2), CORE_IN2_WORDS, offset=(1 * DATA_IN2_WORDS))
             ],
             sync_strategy=SyncStrategy.Parallel_1_to_N
         ),
         # 1 --> 2 wgts upper
         DataTransfer(
             [AieTile(TileType.Memtile, col, 0) for col in range(NUM_COLS)],
             WGT_SIZE, [WGT_UPPER_BUFF_1_ADDR, WGT_UPPER_BUFF_2_ADDR],
             [IN2_REPEAT],
             [TransferParams(DmaChannel(DmaDir.S2MM, 1), WGT_S2MM_WORDS)],
             [
                 TransferParams(DmaChannel(DmaDir.MM2S, 3), CORE_IN2_WORDS, offset=(0 * DATA_IN2_WORDS)),
                 TransferParams(DmaChannel(DmaDir.MM2S, 4), CORE_IN2_WORDS, offset=(1 * DATA_IN2_WORDS))
             ],
             sync_strategy=SyncStrategy.Parallel_1_to_N
         ),
         # 4 --> 1 ofm
         DataTransfer(
             [AieTile(TileType.Memtile, col, 0) for col in range(NUM_COLS)],
             NUM_ROWS * CORE_OUT_SIZE, [OFM_PING_ADDR, OFM_PONG_ADDR], 
             [OUT_REPEAT],
             [
                 TransferParams(DmaChannel(DmaDir.S2MM, 2), CORE_OUT_WORDS, offset=(0 * CORE_OUT_WORDS)),
                 TransferParams(DmaChannel(DmaDir.S2MM, 3), CORE_OUT_WORDS, offset=(1 * CORE_OUT_WORDS)),
                 TransferParams(DmaChannel(DmaDir.S2MM, 4), CORE_OUT_WORDS, offset=(2 * CORE_OUT_WORDS)),
                 TransferParams(DmaChannel(DmaDir.S2MM, 5), CORE_OUT_WORDS, offset=(3 * CORE_OUT_WORDS))
             ],
             [
                 TransferParams(DmaChannel(DmaDir.MM2S, 5), NUM_ROWS * DATA_OUT_WORDS, offset=0,
                             step=[1, N_SUBV, CORE_OUT_WORDS], wrap=[N_SUBV, Mgemm])
             ],
             sync_strategy=SyncStrategy.Parallel_N_to_1
        )
    ]
    code = compile_dma_config(shape, memtile_transfers, [])
    print(code)

def main():
    if not ShareIfmChannel:
        generate_prefill_bds()
    else:
        generate_token_bds()

if __name__ == '__main__':
    main()
