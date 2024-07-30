'''
This module implements a set of routines that take in a high
level description of AIE data transfers and compiles them to
low-level runtime configurations. All buffer descriptor resources
(locks, IDs, memory addresses) are automatically managed.

Each data transfer is specified as a list of input transfer params and
list of output transfer params that write/read from a shared memtile buffer.
Access patterns for the data transfer are specified with the same
semantics as AIE buffer descriptor hardware (32-bit word addressing
with offset, length, step, wrap). This enables predicatable runtime
performance, while automating a majority of hardware resource
allocation and configuration.

The generator runs a full simulation of data transfer address
generation during compilation to detect and report any invalid
memory accesses.

To run the compiler, use the following APIs:

from bdgenerator import generate_bds, DataTransfer, TransferParams, AieTile, TileType, DmaChannel, DmaDir

This module can be combined with a tiling solver to provide fully
automated BD generation.

Author: Spiro Stameson
'''

from typing import List
from enum import Enum
import sys

#
# BD allocations must manage the following resources:
#       --> BD IDs
#       --> DMA Channel IDs
#       --> Memtile address locations
#       --> Lock IDs
#
# BD initialization has the following steps:
#       --> Lock register initialization
#       --> BD configuration
#       --> Enqueue task
#
# The user describes memtile data transfers as follows:
#       --> AIE array columns where transfer will run
#       --> Buffer size
#       --> Number of buffers (for multiple buffering scheme)
#       --> Repeat count
#       --> Input transfer parameters (DMA channel, offset, length, step, wrap)
#       --> Output transfer parameters (DMA channel, offset, length, step, wrap)
#
# The BD allocator manages all other resources and generates
# the runtime initialization code. All resources are allocated
# with a greedy algorithm. Buffer base addresses are rounded
# up to the nearest unused bank to avoid conflicts.
#


################################################################################
#
# User Specified Data Transfer API
#
################################################################################


# Represents a DMA stream direction
class DmaDir(Enum):
    S2MM = 1
    MM2S = 2


# Represents a hardware DMA channel
class DmaChannel:
    MIN_DMA_CHANNEL_ID = 0
    MAX_DMA_CHANNEL_ID = 5
    MIN_MEMTILE_DMA_CHANNEL_ID = 0
    MAX_MEMTILE_DMA_CHANNEL_ID = 5
    MIN_SHIM_DMA_CHANNEL_ID = 0
    MAX_SHIM_DMA_CHANNEL_ID = 1

    def __init__(self, dir: DmaDir, id: int):
        self.dir = dir
        self.id = id

        # Check that DMA direction is valid
        if not isinstance(self.dir, DmaDir):
            raise ValueError(f'Invalid DMA direction, must be DmaDir.S2MM or DmaDir.MM2S!')
        # Check that DMA ID is valid
        if self.id < DmaChannel.MIN_DMA_CHANNEL_ID or self.id > DmaChannel.MAX_DMA_CHANNEL_ID:
            raise ValueError(f'DMA channel ID {self.id} is out of bounds for range ' +
                             f'{DmaChannel.MIN_DMA_CHANNEL_ID} to {DmaChannel.MAX_DMA_CHANNEL_ID}!')

    # Make DmaChannel hashable (used for deadlock detection)

    def __str__(self):
        return self.dir.name + '_' + str(self.id)

    def __eq__(self, other):
        return (self.dir == other.dir and
                self.id == other.id)

    def __hash__(self):
        return hash(str(self))


# Represents a type of AIE tile in the array
class TileType(Enum):
    Memtile = 1
    Shim = 2


# Represents a tile location on the AIE array
class AieTile:
    def __init__(self, type: TileType, col: int, row: int):
        self.type = type
        self.col = col
        self.row = row

        # Check that tile type is valid
        if not isinstance(self.type, TileType):
            raise ValueError(f'Invalid tile type!')
        # Check that col is valid
        if self.col < 0:
            raise ValueError(f'Invalid AieTile column value {self.col} for {self.type.name}!')
        # Check that row is valid
        if self.type in (TileType.Memtile, TileType.Shim):
            if self.row != 0:
                raise ValueError(f'Invalid AieTile row value {self.row} for {self.type.name}!')
        else:
            if self.row < 0:
                raise ValueError(f'Invalid AieTile row value {self.row} for {self.type.name}!')

    # Make AieTile hashable (used in BD allocator)

    def __str__(self):
        return self.type.name.lower() + '_' + str(self.col) + '_' + str(self.row)

    def __eq__(self, other):
        return (self.type == other.type and
                self.col == other.col and
                self.row == other.row)

    def __hash__(self):
        return hash(str(self))


# Represent a DMA location on AIE
class AieDma:
    def __init__(self, tile: AieTile, channel: DmaChannel):
        self.tile = tile
        self.channel = channel

        if self.tile.type == TileType.Memtile:
            is_invalid = (self.channel.id < DmaChannel.MIN_MEMTILE_DMA_CHANNEL_ID or
                          self.channel.id > DmaChannel.MAX_MEMTILE_DMA_CHANNEL_ID)
        else:
            is_invalid = (self.channel.id < DmaChannel.MIN_SHIM_DMA_CHANNEL_ID or
                          self.channel.id > DmaChannel.MAX_SHIM_DMA_CHANNEL_ID)
        if is_invalid:
            raise ValueError(f'Invalid DMA channel {self.channel} for tile {self.tile}!')

    # Make AieDma hashable (used for deadlock detection)

    def __str__(self):
        return str(self.tile) + '_' + str(self.channel)

    def __eq__(self, other):
        return (self.tile == other.tile and
                self.channel == other.channel)

    def __hash__(self):
        return hash(str(self))


# Represent a stream connection between two DMAs
class StreamConnection:
    def __init__(self, input_dma: AieDma, output_dma: AieDma):
        self.input_dma = input_dma
        self.output_dma = output_dma

        valid_dir = (input_dma.channel.dir == DmaDir.MM2S and
                     output_dma.channel.dir == DmaDir.S2MM)
        if not valid_dir:
            raise ValueError('Invalid DMA direction in stream connection')
        valid_tiles = ((input_dma.tile.type == TileType.Shim and
                        output_dma.tile.type == TileType.Memtile) or
                       (input_dma.tile.type == TileType.Memtile and
                        output_dma.tile.type == TileType.Shim))
        if not valid_tiles:
            raise ValueError('Invalid tile locations for stream connection!')


# Represent a user-specified data transfer port
#       NOTE: Transfer sizes are in units of 32-bit word
class TransferParams:
    MAX_LENGTH = 2**32 - 1
    MAX_DIMS = 4
    MAX_MEMTILE_LENGTH = 2**17 - 1
    MAX_MEMTILE_DIMS = 4
    MAX_SHIM_LENGTH = 2**32 - 1
    MAX_SHIM_DIMS = 2

    def __init__(
        self,
        dma_channel: DmaChannel,
        offset: int,
        length: int,
        step: List[int],
        wrap: List[int],
    ):
        self.dma_channel = dma_channel
        self.offset = offset
        self.length = length
        self.step = step
        self.wrap = wrap

        # Check that offset is valid
        if self.offset < 0:
            raise ValueError('Address offset must be non-negative!')
        # Check that transfer length is valid
        if self.length <= 0 or length > TransferParams.MAX_LENGTH:
            raise ValueError(f'Transfer length must be between 1 and {TransferParams.MAX_LENGTH} words!')
        # Check that step and wrap are valid
        if len(self.step) > TransferParams.MAX_DIMS:
            raise ValueError(f'Data transfer addressing supports step in at most {TransferParams.MAX_DIMS} dimensions!')
        if len(self.wrap) > TransferParams.MAX_DIMS - 1:
            raise ValueError(f'Data transfer addressing supports wrap in at most {TransferParams.MAX_DIMS - 1} dimensions!')
        if len(self.step) not in (len(self.wrap), len(self.wrap) + 1):
            raise ValueError('Number of step and wrap dimensions are incompatible!')
        for w in self.wrap:
            if w <= 0:
                raise ValueError('Wrap values must be positive!')
        for s in self.step:
            if s <= 0:
                raise ValueError('Step values must be positive!')


# Simulate a transfer to check for invalid memory accesses
def sim_transfer(
    buffer_size: int,
    input_params: List[TransferParams],
    output_params: List[TransferParams]
):
    # Run the data transfer by writing/reading to a dummy buffer
    # to check that all memory accesses are valid. The two rules are
    #
    #       1) Each buffer word is written to exactly once.
    #       2) Each buffer word is read one or more times
    #          after it has been written to.
    #
    # If the data transfer is input only, then data doesn't need to be read.
    # If the data transfer is output only, than data doesn't need to be written.
    # The input only and output only transfers are used for external DDR, where
    # buffers will be initialized/read by the host CPU.

    DUMMY_INIT_VAL = 0
    DUMMY_WRITE_VAL = 1
    DUMMY_READ_VAL = 2

    WORD_SIZE = 4

    # Run the 4d addressing algorithm
    #       NOTE: This needs to be implemented as an inner function
    #             so that we can break out of the nested for loop by
    #             returning when the transfer is complete.
    def run_4d_transfer(params: TransferParams, dummy_buffer: int):
        assert TransferParams.MAX_DIMS == 4
        # Pad step with trailing 0's
        step = params.step + ([0] * (TransferParams.MAX_DIMS - len(params.step)))
        # Pad wrap with trailing MAX_LENGTH's
        wrap = params.wrap + ([TransferParams.MAX_LENGTH] * (TransferParams.MAX_DIMS - len(params.wrap)))
        # Count number of words in transfer for completion
        counter = 0
        for i in range(wrap[3]):
            for j in range(wrap[2]):
                for k in range(wrap[1]):
                    for l in range(wrap[0]):
                        # Generate 4d address
                        # NOTE: Addressing internally uses units of 32-bit words, but these
                        #       are converted to byte addresses for error messages. This
                        #       matches the convention of aiecompiler and aiesimulator.
                        addr = ((step[3] * i) +
                                (step[2] * j) +
                                (step[1] * k) +
                                (step[0] * l) + params.offset)

                        # Check that addr is within range
                        if addr >= len(dummy_buffer):
                            raise ValueError(f'Data transfer generates address 0x{WORD_SIZE * addr:x} ' +
                                             f'out of bounds for buffer size 0x{WORD_SIZE * len(dummy_buffer):x} ' +
                                             f'at index {[i, j, k, l]}!')
                        # Perform the memory op
                        if params.dma_channel.dir == DmaDir.S2MM:
                            # Check that address has not been initialized
                            if dummy_buffer[addr] != DUMMY_INIT_VAL:
                                raise ValueError(f'Data transfer writes data to address 0x{WORD_SIZE * addr:x} multiple times ' +
                                                 f'at index {[i, j, k, l]}!')
                            dummy_buffer[addr] = DUMMY_WRITE_VAL
                        else:
                            # Check that address has been initialized
                            if dummy_buffer[addr] == DUMMY_INIT_VAL:
                                raise ValueError(f'Data transfer reads uninitialized data from address 0x{WORD_SIZE * addr:x} ' +
                                                 f'at index {[i, j, k, l]}!')
                            dummy_buffer[addr] = DUMMY_READ_VAL

                        # Check for transfer completion
                        counter += 1
                        if counter == params.length:
                            return

    # Initialize dummy buffer for simulation
    if len(input_params) == 0:
        # Mark all words as initialized
        #       NOTE: This is an output only data transfer (for shim MM2S)
        #             so we assume the buffer has been initialized by the host.
        dummy_buffer = [DUMMY_WRITE_VAL] * (buffer_size // WORD_SIZE)
    else:
        # Mark all words as uninitialized
        dummy_buffer = [DUMMY_INIT_VAL] * (buffer_size // WORD_SIZE)

    # Run all input and output transfers
    for t in input_params:
        if t.dma_channel.dir != DmaDir.S2MM:
            raise ValueError('Input transfer must have S2MM DMA direction!')
        run_4d_transfer(t, dummy_buffer)
    for t in output_params:
        if t.dma_channel.dir != DmaDir.MM2S:
            raise ValueError('Output transfer must have MM2S DMA direction!')
        run_4d_transfer(t, dummy_buffer)

    # Check the dummy buffer for invalid memory operations
    if len(output_params) == 0:
        # Check that every buffer has been written to
        #       NOTE: This is an input only data transfer (for shim S2MM).
        for addr in range(len(dummy_buffer)):
            if dummy_buffer[addr] == DUMMY_INIT_VAL:
                raise ValueError(f'Data transfer never initializes word at address 0x{WORD_SIZE * addr:x}!')
    else:
        # Check that every buffer address has been read
        #       NOTE: This is either an output only data transfer (for shim MM2S)
        #             or a memtile data transfer with both input and output.
        for addr in range(len(dummy_buffer)):
            if dummy_buffer[addr] == DUMMY_INIT_VAL:
                raise ValueError(f'Data transfer never initializes word at address 0x{WORD_SIZE * addr:x}!')
            if dummy_buffer[addr] == DUMMY_WRITE_VAL:
                raise ValueError(f'Data transfer initializes word at address 0x{WORD_SIZE * addr:x} ' +
                                 f'but never reads the value!')


#
# DataTransfer attributes are as follows
#       aie_tiles - tiles of AIE array where the data transfer will run
#       buffer_size - size of the buffer to allocate
#       num_buffers - number of buffers in multiple buffering scheme
#                     (i.e. 2 for double buffering, 3 for triple, ect.)
#       repeat_count - number of times to run the transfer
#       input_params - access pattern for input data transfers
#       output_params - access pattern for output data transfer
#
class DataTransfer:
    def __init__(
        self,
        aie_tiles: List[AieTile],
        buffer_size: int,
        num_buffers: int,
        repeat_count: int,
        input_params: List[TransferParams],
        output_params: List[TransferParams],
        disable_memcheck: bool = False
    ):
        self.aie_tiles = aie_tiles
        self.buffer_size = buffer_size
        self.num_buffers = num_buffers
        self.repeat_count = repeat_count
        self.input_params = input_params
        self.output_params = output_params

        if TileType.Shim in [tile.type for tile in self.aie_tiles]:
            raise NotImplementedError('Shim data transfers not yet supported!')

        # Check that AIE tiles are valid
        if len(self.aie_tiles) == 0:
            raise ValueError('Data transfer must be placed on one or more tiles!')
        if len(set(self.aie_tiles)) != len(self.aie_tiles):
            raise ValueError('AIE tiles for data transfer must be unique!')
        if len(set([tile.type for tile in self.aie_tiles])) != 1:
            raise ValueError('AIE tiles for data transfer must all have the same type!')
        # Check that buffer size is valid
        if self.buffer_size <= 0:
            raise ValueError('Data transfer buffer size must be positive!')
        if self.buffer_size % 4 != 0:
            raise ValueError('Data transfer buffer size must be a multiple of 32-bit words!')
        # Check that number of buffers is valid
        if self.num_buffers <= 0:
            raise ValueError('Number of data transfer buffers must be at least one!')
        # Check that repeat count is valid
        if self.repeat_count < 1:
            raise ValueError('Cannot enqueue BDs with repeat count less than one!')
        # Check that data transfer is valid
        try:
            sim_transfer(self.buffer_size, self.input_params, self.output_params)
        except ValueError as e:
            if disable_memcheck:
                print(e, file=sys.stderr)
            else:
                raise e


################################################################################
#
# Internal Memtile BD Allocation and Runtime Code Generation
#
################################################################################


# Manage BD and locking resources for a single memtile
class MemtileAllocator:
    MAX_BD_ID_LO = 23
    MAX_BD_ID_HI = 47
    MAX_LOCK_ID = 63
    MAX_BUFFER_ADDR = 2**19 - 1
    MEM_BANK_SIZE = 256

    def __init__(self, aie_tile: AieTile):
        assert aie_tile.type == TileType.Memtile
        self.aie_tile = aie_tile
        self.bd_id_lo_counter = 0
        self.bd_id_hi_counter = MemtileAllocator.MAX_BD_ID_LO + 1
        self.lock_id_counter = 0
        self.base_addr_counter = 0

    # Allocate an id number for BD configuration
    #       NOTE: BDs on even DMA channels can have IDs from 0 to 23
    #             BDs on odd DMA channels can have IDs from 24 to 47
    def bd_id(self, dma_channel: DmaChannel) -> int:
        if dma_channel.id % 2 == 0:
            id = self.bd_id_lo_counter
            if id > MemtileAllocator.MAX_BD_ID_LO:
                raise RuntimeError(f'Failed to allocate BD ID on {self.aie_tile}, DMA channel {dma_channel}!')
            self.bd_id_lo_counter += 1
        else:
            id = self.bd_id_hi_counter
            if id > MemtileAllocator.MAX_BD_ID_HI:
                raise RuntimeError(f'Failed to allocate BD ID on {self.aie_tile}, DMA channel {dma_channel}!')
            self.bd_id_hi_counter += 1
        return id

    # Allocate an id for lock registers
    def lock_id(self) -> int:
        id = self.lock_id_counter
        if id > MemtileAllocator.MAX_LOCK_ID:
            raise RuntimeError(f'Failed to allocate lock ID on {self.aie_tile}!')
        self.lock_id_counter += 1
        return id

    # Allocate a buffer with buffer_size bytes
    def buffer_addr(self, buffer_size: int) -> int:
        def round_to_multiple(x, m):
            return ((x + (m - 1)) // m) * m
        base_addr = self.base_addr_counter
        if base_addr + buffer_size > MemtileAllocator.MAX_BUFFER_ADDR:
            raise RuntimeError(f'Failed to allocate 0x{buffer_size:x} byte buffer on {self.aie_tile}!')
        # Bump base addr and round up to nearest memtile bank
        #       NOTE: AIE2p memtile banks are interleaved, so we just need
        #             the buffer address to be 256-bit aligned
        self.base_addr_counter = round_to_multiple(self.base_addr_counter + buffer_size,
                                                   MemtileAllocator.MEM_BANK_SIZE)
        return base_addr


# Allocate BD resources for a data transfer port
class MemtileBD:
    def __init__(
        self,
        alloc: MemtileAllocator,
        transfer_params: TransferParams,
        use_next_bd: bool = False,
        next_bd_id: int = 0
    ):
        self.id = alloc.bd_id(transfer_params.dma_channel)
        self.name = 'bd_' + str(alloc.aie_tile) + '_id' + str(self.id)
        self.transfer_params = transfer_params
        self.use_next_bd = use_next_bd
        self.next_bd_id = next_bd_id


# Manage resources for a memtile buffer and generate all runtime control code
class MemtileBuffer:
    CONFIG_HEADER ='''#ifndef BD_CONFIG_H
#define BD_CONFIG_H

#include <adf/adf_api/AIERuntimeControl.h>

void run_memtile_bd_config()
{
    //
    // NOTE: This code is auto-generated, so do not modify.
    //'''
    CONFIG_FOOTER = '''}

#endif // BD_CONFIG_H'''
    LOCAL_ADDR_OFFSET = 0x80000
    LOCAL_LOCK_OFFSET = 64

    def __init__(
        self,
        alloc: MemtileAllocator,
        buffer_size: int,
        input_params: List[TransferParams],
        output_params: List[TransferParams]
    ):
        self.aie_tile = alloc.aie_tile
        self.base_addr = alloc.buffer_addr(buffer_size)
        self.buffer_size = buffer_size
        self.input_bds = [MemtileBD(alloc, t) for t in input_params]
        self.output_bds = [MemtileBD(alloc, t) for t in output_params]
        self.locks = [alloc.lock_id() for _ in range(len(self.input_bds) + len(self.output_bds))]

    # pong should have type MemtileBuffer, but __future__ type
    # annotations aren't supported by python 3.6
    def chain_bds(self, pong):
        for i in range(len(self.input_bds)):
            self.input_bds[i].use_next_bd = True
            self.input_bds[i].next_bd_id = pong.input_bds[i].id
        for i in range(len(self.output_bds)):
            self.output_bds[i].use_next_bd = True
            self.output_bds[i].next_bd_id = pong.output_bds[i].id

    def header_comment(self) -> str:
        code = f'''
    //
    // {len(self.input_bds)} --> {len(self.output_bds)} Memtile Buffer
    // Col   : {self.aie_tile.col}
    // Row   : {self.aie_tile.row}
    // Addr  : 0x{self.base_addr:x}
    // Size  : 0x{self.buffer_size:x}
    //'''
        return code

    def init_locks(self) -> str:
        # Initialize first lock to +1 and all other locks to +0
        code = f'''
    adf::initializeLock(adf::memory_tile, {self.aie_tile.col}, {self.aie_tile.row}, {self.locks[0]}, +1);'''
        for lock in self.locks[1:]:
            code += f'''
    adf::initializeLock(adf::memory_tile, {self.aie_tile.col}, {self.aie_tile.row}, {lock}, +0);'''
        return code

    def config_bds(self) -> str:
        idx = 0
        code = '''
    // Input BDs'''
        for bd in self.input_bds:
            code += self._declare_bd(bd, self.locks[idx], self.locks[(idx + 1) % len(self.locks)])
            code += self._config_bd(bd)
            code += '\n'
            idx += 1
        code += '''
    // Output BDs'''
        for bd in self.output_bds:
            code += self._declare_bd(bd, self.locks[idx], self.locks[(idx + 1) % len(self.locks)])
            code += self._config_bd(bd)
            code += '\n'
            idx += 1
        return code.rstrip()

    def enqueue_bds(self, repeat_count: int) -> str:
        code = '''
    // Input BDs'''
        for bd in self.input_bds:
            code += self._enqueue_bd(bd, repeat_count)
        code += '\n'
        code += '''
    // Output BDs'''
        for bd in self.output_bds:
            code += self._enqueue_bd(bd, repeat_count)
        return code.rstrip()

    def _declare_bd(self, bd: MemtileBD, lock_acq_id: int, lock_rel_id: int) -> str:
        code = f'''
    adf::dma_buffer_descriptor {bd.name};
    {bd.name}.address = ((0x{MemtileBuffer.LOCAL_ADDR_OFFSET:x} + 0x{self.base_addr:x}) / sizeof(int32_t)) + {bd.transfer_params.offset};
    {bd.name}.length = {bd.transfer_params.length};
    {bd.name}.stepsize = {{{", ".join(str(x) for x in bd.transfer_params.step)}}};
    {bd.name}.wrap = {{{", ".join(str(x) for x in bd.transfer_params.wrap)}}};
    {bd.name}.lock_acq_enable = true;
    {bd.name}.lock_acq_value = -1;
    {bd.name}.lock_acq_id = {MemtileBuffer.LOCAL_LOCK_OFFSET} + {lock_acq_id};
    {bd.name}.lock_rel_value = +1;
    {bd.name}.lock_rel_id = {MemtileBuffer.LOCAL_LOCK_OFFSET} + {lock_rel_id};
    {bd.name}.use_next_bd = {str(bd.use_next_bd).lower()};
    {bd.name}.next_bd = {bd.next_bd_id};'''
        return code

    def _config_bd(self, bd: MemtileBD) -> str:
        code = f'''
    adf::configureBufferDescriptor(adf::memory_tile, {self.aie_tile.col}, {self.aie_tile.row}, {bd.id}, {bd.name});'''
        return code

    def _enqueue_bd(self, bd: MemtileBD, repeat_count: int) -> str:
        dir = 'adf::dma_s2mm' if bd.transfer_params.dma_channel.dir == DmaDir.S2MM else \
              'adf::dma_mm2s'
        code = f'''
    adf::enqueueTask(adf::memory_tile, {self.aie_tile.col}, {self.aie_tile.row}, {dir}, {bd.transfer_params.dma_channel.id}, {bd.id}, {repeat_count}, false);'''
        return code


################################################################################
#
# Internal Shim BD Allocation and Runtime Code Generation
#
################################################################################


# Manage shim BD resources
class ShimAllocator:
    MIN_BD_ID = 0
    MAX_BD_ID = 15

    def __init__(self, aie_tile: AieTile):
        assert aie_tile.type == TileType.Shim
        self.aie_tile = aie_tile
        self.bd_id_counter = ShimAllocator.MIN_BD_ID

    def bd_id(self):
        id = self.bd_id_counter
        if id > ShimAllocator.MAX_BD_ID:
            raise RuntimeError(f'Failed to allocate BD ID on {self.aie_tile}!')
        self.bd_id_counter += 1
        return id


# Allocate BD resources for a data transfer port
class ShimBD:
    def __init__(
        self,
        alloc: ShimAllocator,
        transfer_params: TransferParams
    ):
        self.id = alloc.bd_id()
        self.name = 'bd_' + str(alloc.aie_tile) + '_id' + str(self.id)
        self.transfer_params = transfer_params


# Represent an external DRAM buffer
class ExternalBuffer:
    def __init__(
        self,
        alloc: ShimAllocator,
        buffer_size: int,
        input_params: List[TransferParams],
        output_params: List[TransferParams]
    ):
        self.aie_tile = alloc.aie_tile
        self.buffer_size = buffer_size
        self.input_bds = [ShimBD(alloc, t) for t in input_params]
        self.output_bds = [ShimBD(alloc, t) for t in output_params]


################################################################################
#
# User API to launch the BD generator
#
################################################################################


def generate_bds(num_aie_cols: int, transfers: List[DataTransfer]):
    # Create allocators
    allocs = {}
    for col in range(num_aie_cols):
        tile = AieTile(TileType.Memtile, col, 0)
        allocs[tile] = MemtileAllocator(tile)
    # Generate code
    print(MemtileBuffer.CONFIG_HEADER)
    for transfer in transfers:
        for tile in transfer.aie_tiles:
            # Chain BDs for multiple buffering mechanism
            buffer_chain = [MemtileBuffer(allocs[tile], transfer.buffer_size,
                                          transfer.input_params, transfer.output_params)]
            for i in range(transfer.num_buffers - 1):
                b = MemtileBuffer(allocs[tile], transfer.buffer_size,
                                  transfer.input_params, transfer.output_params)
                buffer_chain[i].chain_bds(b)
                buffer_chain.append(b)
            # Generate configuration code
            for b in buffer_chain:
                print(b.header_comment())
                print(b.init_locks())
                print(b.config_bds())
            # Divide transfer repeat_count by num_buffers and round up
            repeat_count = (transfer.repeat_count + transfer.num_buffers - 1) // transfer.num_buffers
            # Generate code to enqueue the first BDs in the chain
            print(buffer_chain[0].enqueue_bds(repeat_count))
    print(MemtileBuffer.CONFIG_FOOTER)
