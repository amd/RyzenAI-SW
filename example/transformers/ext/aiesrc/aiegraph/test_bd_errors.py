from bdgenerator import DataTransfer, TransferParams, AieTile, TileType, DmaChannel, DmaDir

TRANSFER_SIZE = 256
TRANSFER_WORDS = TRANSFER_SIZE // 4

def main():
    # Check invalid input DMA direction
    try:
        DataTransfer(
            [AieTile(TileType.Memtile, 0, 0)], TRANSFER_SIZE, 1, 1,
            [TransferParams(DmaChannel(DmaDir.MM2S, 0), 0, TRANSFER_WORDS, [1], [])],
            [TransferParams(DmaChannel(DmaDir.MM2S, 0), 0, TRANSFER_WORDS, [1], [])])
    except ValueError as e:
        print(e)
    # Check invalid output DMA direction
    try:
        DataTransfer(
            [AieTile(TileType.Memtile, 0, 0)], TRANSFER_SIZE, 1, 1,
            [TransferParams(DmaChannel(DmaDir.S2MM, 0), 0, TRANSFER_WORDS, [1], [])],
            [TransferParams(DmaChannel(DmaDir.S2MM, 0), 0, TRANSFER_WORDS, [1], [])])
    except ValueError as e:
        print(e)
    # Check out of bounds access
    try:
        DataTransfer(
            [AieTile(TileType.Memtile, 0, 0)], TRANSFER_SIZE - 4, 1, 1,
            [TransferParams(DmaChannel(DmaDir.S2MM, 0), 0, TRANSFER_WORDS, [1], [])],
            [TransferParams(DmaChannel(DmaDir.MM2S, 0), 0, TRANSFER_WORDS, [1], [])])
    except ValueError as e:
        print(e)
    # Check over-writing the same address
    try:
        DataTransfer(
            [AieTile(TileType.Memtile, 0, 0)], TRANSFER_SIZE, 1, 1,
            [TransferParams(DmaChannel(DmaDir.S2MM, 0), 0, TRANSFER_WORDS, [1], []),
             TransferParams(DmaChannel(DmaDir.S2MM, 1), 0, TRANSFER_WORDS, [1], [])],
            [TransferParams(DmaChannel(DmaDir.MM2S, 0), 0, TRANSFER_WORDS, [1], [])])
    except ValueError as e:
        print(e)
    # Check reading an uninitialized address
    try:
        DataTransfer(
            [AieTile(TileType.Memtile, 0, 0)], TRANSFER_SIZE, 1, 1,
            [TransferParams(DmaChannel(DmaDir.S2MM, 0), 0, TRANSFER_WORDS - 1, [1], [])],
            [TransferParams(DmaChannel(DmaDir.MM2S, 0), 0, TRANSFER_WORDS, [1], [])])
    except ValueError as e:
        print(e)
    # Check leaving a buffer location uninitialized
    try:
        DataTransfer(
            [AieTile(TileType.Memtile, 0, 0)], TRANSFER_SIZE, 1, 1,
            [TransferParams(DmaChannel(DmaDir.S2MM, 0), 0, TRANSFER_WORDS - 1, [1], [])],
            [TransferParams(DmaChannel(DmaDir.MM2S, 0), 0, TRANSFER_WORDS - 1, [1], [])])
    except ValueError as e:
        print(e)
    # Check writing but not reading a buffer location
    try:
        DataTransfer(
            [AieTile(TileType.Memtile, 0, 0)], TRANSFER_SIZE, 1, 1,
            [TransferParams(DmaChannel(DmaDir.S2MM, 0), 0, TRANSFER_WORDS, [1], [])],
            [TransferParams(DmaChannel(DmaDir.MM2S, 0), 0, TRANSFER_WORDS - 1, [1], [])])
    except ValueError as e:
        print(e)
    # TODO: These should be located on shim tiles, but that isn't implemented yet...
    # Check that input only data transfer doesn't generate an error
    DataTransfer(
        [AieTile(TileType.Memtile, 0, 0)], TRANSFER_SIZE, 1, 1,
        [TransferParams(DmaChannel(DmaDir.S2MM, 0), 0, TRANSFER_WORDS, [1], [])],
        [])
    # Check that output only data transfer doesn't generate an error
    DataTransfer(
        [AieTile(TileType.Memtile, 0, 0)], TRANSFER_SIZE, 1, 1,
        [],
        [TransferParams(DmaChannel(DmaDir.MM2S, 0), 0, TRANSFER_WORDS, [1], [])])


if __name__ == '__main__':
    main()
