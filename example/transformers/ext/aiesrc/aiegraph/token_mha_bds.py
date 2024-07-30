from bdgenerator import generate_bds, DataTransfer, TransferParams, AieTile, TileType, DmaChannel, DmaDir

NUM_COLS = 4
WORD_SIZE = 4
CORE_IN_SIZE = 8192
CORE_IN_WORDS = CORE_IN_SIZE // WORD_SIZE
CORE_OUT_SIZE = 8192
CORE_OUT_WORDS = CORE_OUT_SIZE // WORD_SIZE
REPEAT_COUNT = 1

def main():
    generate_bds(
        NUM_COLS,
        [DataTransfer(
            [AieTile(TileType.Memtile, col, 0) for col in range(NUM_COLS)], 2 * CORE_IN_SIZE, 2, REPEAT_COUNT,
            [TransferParams(DmaChannel(DmaDir.S2MM, 0), 0, 2 * CORE_IN_WORDS, [1], [])],
            [TransferParams(DmaChannel(DmaDir.MM2S, 1), 0 * CORE_IN_WORDS, CORE_IN_WORDS, [1], []),
             TransferParams(DmaChannel(DmaDir.MM2S, 2), 1 * CORE_IN_WORDS, CORE_IN_WORDS, [1], [])]),
         DataTransfer(
            [AieTile(TileType.Memtile, col, 0) for col in range(NUM_COLS)], 2 * CORE_IN_SIZE, 2, REPEAT_COUNT,
            [TransferParams(DmaChannel(DmaDir.S2MM, 1), 0, 2 * CORE_IN_WORDS, [1], [])],
            [TransferParams(DmaChannel(DmaDir.MM2S, 3), 0 * CORE_IN_WORDS, CORE_IN_WORDS, [1], []),
             TransferParams(DmaChannel(DmaDir.MM2S, 4), 1 * CORE_IN_WORDS, CORE_IN_WORDS, [1], [])]),
         DataTransfer(
            [AieTile(TileType.Memtile, col, 0) for col in range(NUM_COLS)], 4 * CORE_OUT_SIZE, 2, REPEAT_COUNT,
            [TransferParams(DmaChannel(DmaDir.S2MM, 2), 0 * CORE_OUT_WORDS, CORE_OUT_WORDS, [1], []),
             TransferParams(DmaChannel(DmaDir.S2MM, 3), 1 * CORE_OUT_WORDS, CORE_OUT_WORDS, [1], []),
             TransferParams(DmaChannel(DmaDir.S2MM, 4), 2 * CORE_OUT_WORDS, CORE_OUT_WORDS, [1], []),
             TransferParams(DmaChannel(DmaDir.S2MM, 5), 3 * CORE_OUT_WORDS, CORE_OUT_WORDS, [1], [])],
            [TransferParams(DmaChannel(DmaDir.MM2S, 5), 0, 4 * CORE_OUT_WORDS, [1], [])])])

if __name__ == '__main__':
    main()
