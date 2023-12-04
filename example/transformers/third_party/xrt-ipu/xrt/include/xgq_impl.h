/*
 *  SPDX-License-Identifier: Apache-2.0
 *  Copyright (C) 2021, Xilinx Inc
 *  Copyright (C) 2022, Advanced Micro Devices, Inc.  All rights reserved.
 */
#ifndef XGQ_IMPL_H
#define XGQ_IMPL_H

/* !!! This header file is for internal project use only and it is subject to removal without notice !!! */
#if defined(__KERNEL__)
# include <linux/types.h>
#else
# include <stdint.h>
# include <stddef.h>
# include <errno.h>
#endif /* __KERNEL__ */

#include "xgq_cmd_common.h"

/*
 * Standard bool type's portability is poor across multiple OSes and HW platforms.
 * Defineing our own true and false values as integer to avoid using bool.
 */
#define XGQ_TRUE	1
#define XGQ_FALSE	0

/*
 * Generic XGQ implementation.
 *
 * Each platform should define its own xgq_XXX_plat.h where the platform
 * specific operations are implemented. Afterwards, XGQ_IMPL should be
 * defined and xgq_impl.h should be included there.
 *
 * Note: This file (xgq_impl.h) should only be included by xgq_XXX_plat.h.
 */

#ifndef XGQ_IMPL
#if !defined(__KERNEL__)
#define ____cacheline_aligned_in_smp
#endif
static inline void xgq_mem_write32(uint64_t hdl, uint64_t addr, uint32_t val) {}
static inline void xgq_reg_write32(uint64_t hdl, uint64_t addr, uint32_t val) {}
static inline uint32_t xgq_mem_read32(uint64_t hdl, uint64_t addr)
{
	return 0xFFFFFFFF;
}
static inline uint32_t xgq_reg_read32(uint64_t hdl, uint64_t addr)
{
	return 0xFFFFFFFF;
}
#endif

#ifndef likely
#define likely(x)	__builtin_expect(!!(x), 1) // NOLINT
#endif
#ifndef unlikely
#define unlikely(x)	__builtin_expect(!!(x), 0) // NOLINT
#endif

/*
 * xgq_read32() and xgq_write32() is used when the address can be register or in memory.
 * The producer pointer is one example today.
 */
static inline uint32_t xgq_read32(uint64_t io_hdl, uint64_t addr, int is_mem)
{
#ifdef XGQ_MEM_REG_ACCESS_DIFFER
	return is_mem ? xgq_mem_read32(io_hdl, addr) : xgq_reg_read32(io_hdl, addr);
#else
	return xgq_reg_read32(io_hdl, addr);
#endif
}
static inline void xgq_write32(uint64_t io_hdl, uint64_t addr, uint32_t val, int is_mem)
{
#ifdef XGQ_MEM_REG_ACCESS_DIFFER
	is_mem ? xgq_mem_write32(io_hdl, addr, val) : xgq_reg_write32(io_hdl, addr, val);
#else
	xgq_reg_write32(io_hdl, addr, val);
#endif
}

/*
 * Currently, this is only used as a workaround for the BRAM read/write collision HW
 * issue on MB ERT, which will cause ERT to read incorrect value from CQ. We only
 * trust the value until we read twice and got the same value.
 */
static inline uint32_t xgq_double_read32(uint64_t io_hdl, uint64_t addr, int is_mem)
{
	uint32_t val[2];
	int i = 0;

	val[1] = xgq_read32(io_hdl, addr, is_mem);
	val[0] = val[1] - 1;
	while (val[0] != val[1])
		val[i++ & 0x1] = xgq_read32(io_hdl, addr, is_mem);
	return val[0];
}

/*
 * One XGQ consists of one submission (SQ) and one completion ring (CQ) buffer shared by one client
 * and one server. Client send request through SQ to server, which processes it and send back
 * response through CQ.
 */
#define XGQ_ALLOC_MAGIC			0x5847513F	/* XGQ? */
#define XGQ_MAJOR			1
#define XGQ_MINOR			0
#define XGQ_MIN_NUM_SLOTS		2
#define XGQ_VERSION			((XGQ_MAJOR<<16)+XGQ_MINOR)
#define GET_XGQ_MAJOR(version)		(version>>16)
#define GET_XGQ_MINOR(version)		(version&0xFFFF)

/*
 * Meta data shared b/w client and server of XGQ
 */
struct xgq_header {
	uint32_t xh_magic; /* Always the first member */
	uint32_t xh_version;

	/* SQ and CQ share the same num of slots. */
	uint32_t xh_slot_num;

	uint32_t xh_sq_offset;
	uint32_t xh_sq_slot_size;
	uint32_t xh_cq_offset;
	/* CQ slot size and format is tied to XGQ version. */

	/*
	 * Consumed pointer for both SQ and CQ are here since they don't generate interrupt,
	 * so no need to be a register.
	 */
	uint32_t xh_sq_consumed;
	uint32_t xh_cq_consumed;

	uint32_t xh_flags;

	/*
	 * On some platforms, there is no dedicated producer pointer register. We can use
	 * below in-mem version to communicate b/w the peers.
	 */
	uint32_t xh_sq_produced;
	uint32_t xh_cq_produced;
};

/* Software representation of a single ring buffer. */
struct xgq_ring {
	struct xgq *xr_xgq; /* pointing back to parent q */
	uint32_t xr_slot_num;
	uint32_t xr_slot_sz;
	uint32_t xr_produced;
	uint32_t xr_consumed;

	uint64_t xr_produced_addr;
	uint64_t xr_consumed_addr;
	uint64_t xr_slot_addr;
};

/* Software representation of a single XGQ. */
#define XGQ_DOUBLE_READ		(1UL << 1) // NOLINT
#define XGQ_IN_MEM_PROD		(1UL << 2) // NOLINT
struct xgq {
	uint64_t xq_io_hdl;
	uint64_t xq_header_addr;
	uint32_t xq_flags;
	struct xgq_ring xq_sq ____cacheline_aligned_in_smp;
	struct xgq_ring xq_cq ____cacheline_aligned_in_smp;
};
#define XGQ_NEED_DOUBLE_READ(xgq)	(((xgq)->xq_flags & XGQ_DOUBLE_READ) != 0)
#define XGQ_IS_IN_MEM_PROD(xgq)		(((xgq)->xq_flags & XGQ_IN_MEM_PROD) != 0)


/*
 * XGQ implementation details and helper routines.
 */

static inline size_t xgq_ring_len(size_t nslots, size_t slotsz)
{
	return sizeof(struct xgq_header) + nslots * (slotsz + sizeof(struct xgq_com_queue_entry));
}

static inline void xgq_copy_to_ring(uint64_t io_hdl, void *buf, uint64_t tgt, size_t len)
{
	size_t i = 0;
	uint32_t *src = (uint32_t *)buf;

	for (i = 0; i < len / 4; i++, tgt += 4)
		xgq_mem_write32(io_hdl, tgt, src[i]);
}

static inline void xgq_copy_from_ring(uint64_t io_hdl, void *buf, uint64_t src, size_t len)
{
	size_t i = 0;
	uint32_t *tgt = (uint32_t *)buf;

	for (i = 0; i < len / 4; i++, src += 4)
		tgt[i] = xgq_mem_read32(io_hdl, src);
}

static inline void xgq_init_ring(struct xgq *xgq, struct xgq_ring *ring,
				 uint64_t produced, uint64_t consumed, uint64_t slots,
				 uint32_t slot_num, uint32_t slot_size)
{
	ring->xr_xgq = xgq;
	ring->xr_produced_addr = produced;
	ring->xr_consumed_addr = consumed;
	ring->xr_slot_addr = slots;
	ring->xr_slot_sz = slot_size;
	ring->xr_slot_num = slot_num;
	ring->xr_produced = ring->xr_consumed = 0;
}

static inline int xgq_ring_full(struct xgq_ring *ring)
{
	return (ring->xr_produced - ring->xr_consumed) >= ring->xr_slot_num;
}

static inline int xgq_ring_empty(struct xgq_ring *ring)
{
	return ring->xr_produced == ring->xr_consumed;
}

static inline void xgq_ring_read_produced(uint64_t io_hdl, struct xgq_ring *ring)
{
#ifdef BRAM_COLLISION_WORKAROUND
	ring->xr_produced = xgq_double_read32(io_hdl, ring->xr_produced_addr,
					      XGQ_IS_IN_MEM_PROD(ring->xr_xgq));
#else
	if (unlikely(XGQ_NEED_DOUBLE_READ(ring->xr_xgq))) {
		ring->xr_produced = xgq_double_read32(io_hdl, ring->xr_produced_addr,
						      XGQ_IS_IN_MEM_PROD(ring->xr_xgq));
	} else {
		ring->xr_produced = xgq_read32(io_hdl, ring->xr_produced_addr,
					       XGQ_IS_IN_MEM_PROD(ring->xr_xgq));
	}
#endif
}

static inline void xgq_ring_write_produced(uint64_t io_hdl, struct xgq_ring *ring)
{
	xgq_write32(io_hdl, ring->xr_produced_addr, ring->xr_produced,
		    XGQ_IS_IN_MEM_PROD(ring->xr_xgq));
}

static inline void xgq_ring_read_consumed(uint64_t io_hdl, struct xgq_ring *ring)
{
#ifdef BRAM_COLLISION_WORKAROUND
	ring->xr_consumed = xgq_double_read32(io_hdl, ring->xr_consumed_addr, XGQ_TRUE);
#else
	if (unlikely(XGQ_NEED_DOUBLE_READ(ring->xr_xgq)))
		ring->xr_consumed = xgq_double_read32(io_hdl, ring->xr_consumed_addr, XGQ_TRUE);
	else
		ring->xr_consumed = xgq_mem_read32(io_hdl, ring->xr_consumed_addr);
#endif
}

static inline void xgq_ring_write_consumed(uint64_t io_hdl, struct xgq_ring *ring)
{
	xgq_mem_write32(io_hdl, ring->xr_consumed_addr, ring->xr_consumed);
}

static inline uint64_t xgq_ring_slot_ptr_produced(struct xgq_ring *ring)
{
	return ring->xr_slot_addr +
		/*
		 * In reality, below multiplication of two 32-bit ints will not overflow.
		 * So, keep it as-is, instead of doing 64-bit mutiplication, which is very
		 * slow on 32-bit CPU, e.g., Microblaze.
		 */
		ring->xr_slot_sz * (ring->xr_produced & (ring->xr_slot_num - 1));
}

static inline uint64_t xgq_ring_slot_ptr_consumed(struct xgq_ring *ring)
{
	return ring->xr_slot_addr +
		/*
		 * In reality, below multiplication of two 32-bit ints will not overflow.
		 * So, keep it as-is, instead of doing 64-bit mutiplication, which is very
		 * slow on 32-bit CPU, e.g., Microblaze.
		 */
		ring->xr_slot_sz * (ring->xr_consumed & (ring->xr_slot_num - 1));
}

static inline int xgq_can_produce(struct xgq *xgq)
{
#ifdef XGQ_SERVER
	struct xgq_ring *ring = &xgq->xq_cq;
#else
	struct xgq_ring *ring = &xgq->xq_sq;
#endif

	if (likely(!xgq_ring_full(ring)))
		return XGQ_TRUE;
	xgq_ring_read_consumed(xgq->xq_io_hdl, ring);
	return !xgq_ring_full(ring);
}

static inline int xgq_can_consume(struct xgq *xgq)
{
#ifdef XGQ_SERVER
	struct xgq_ring *ring = &xgq->xq_sq;
#else
	struct xgq_ring *ring = &xgq->xq_cq;
#endif

	if (likely(!xgq_ring_empty(ring)))
		return XGQ_TRUE;
	xgq_ring_read_produced(xgq->xq_io_hdl, ring);
	return !xgq_ring_empty(ring);
}

/*
 * Fast forward to where we left. Used only during xgq_attach().
 */
static inline void xgq_fast_forward(struct xgq *xgq, struct xgq_ring *ring)
{
	xgq_ring_read_produced(xgq->xq_io_hdl, ring);
	xgq_ring_read_consumed(xgq->xq_io_hdl, ring);
}

/*
 * Set consumed to be the same as produced to ignore any existing commands. And there should not
 * be any left over commands anyway. Used only during xgq_alloc().
 */
static inline void xgq_soft_reset(struct xgq *xgq, struct xgq_ring *ring)
{
	xgq_ring_read_produced(xgq->xq_io_hdl, ring);
	ring->xr_consumed = ring->xr_produced;
	xgq_ring_write_consumed(xgq->xq_io_hdl, ring);
}

static inline void
xgq_init(struct xgq *xgq, uint64_t flags, uint64_t io_hdl, uint64_t ring_addr,
	 size_t n_slots, uint32_t slot_size, uint64_t sq_produced, uint64_t cq_produced)
{
	struct xgq_header hdr = {};
	uint64_t sqprod, cqprod;

	xgq->xq_flags = flags;
#ifdef BRAM_COLLISION_WORKAROUND
	xgq->xq_flags |= XGQ_DOUBLE_READ;
#endif
	xgq->xq_io_hdl = io_hdl;
	xgq->xq_header_addr = ring_addr;

	if (XGQ_IS_IN_MEM_PROD(xgq)) {
		/* Passed-in sq/cq producer pointer will be ignored. */
		sqprod = ring_addr + offsetof(struct xgq_header, xh_sq_produced);
		cqprod = ring_addr + offsetof(struct xgq_header, xh_cq_produced);
	} else {
		sqprod = sq_produced;
		cqprod = cq_produced;
	}
	xgq_init_ring(xgq, &xgq->xq_sq, sqprod,
		      ring_addr + offsetof(struct xgq_header, xh_sq_consumed),
		      ring_addr + sizeof(struct xgq_header), n_slots, slot_size);
	xgq_init_ring(xgq, &xgq->xq_cq, cqprod,
		      ring_addr + offsetof(struct xgq_header, xh_cq_consumed),
		      ring_addr + sizeof(struct xgq_header) + n_slots * slot_size,
		      n_slots, sizeof(struct xgq_com_queue_entry));

	hdr.xh_magic = 0;
	hdr.xh_version = XGQ_VERSION;
	hdr.xh_slot_num = n_slots;
	hdr.xh_sq_offset = xgq->xq_sq.xr_slot_addr - ring_addr;
	hdr.xh_sq_slot_size = slot_size;
	hdr.xh_cq_offset = xgq->xq_cq.xr_slot_addr - ring_addr;
	hdr.xh_sq_consumed = 0;
	hdr.xh_sq_consumed = 0;
	hdr.xh_cq_produced = 0;
	hdr.xh_cq_produced = 0;
	hdr.xh_flags = xgq->xq_flags;
	xgq_copy_to_ring(xgq->xq_io_hdl, &hdr, ring_addr, sizeof(hdr));

	xgq_soft_reset(xgq, &xgq->xq_sq);
	xgq_soft_reset(xgq, &xgq->xq_cq);

	// Write the magic number to confirm the header is fully initialized
	hdr.xh_magic = XGQ_ALLOC_MAGIC;
	xgq_copy_to_ring(xgq->xq_io_hdl, &hdr, ring_addr, sizeof(uint32_t));
}

static inline size_t
xgq_alloc_num_slots(size_t rlen, const uint32_t *slot_size, size_t n_slots)
{
	const uint32_t numbits = (sizeof(uint32_t) * 8);
	uint32_t i = 0;
	uint32_t total_len = 0;
	uint32_t numslots = 1;

	while ((total_len <= rlen) && (numslots < (0x1U << (numbits - 1)))) {
		numslots <<= 1;
		for (i = 0, total_len = 0; i < n_slots; i++)
			total_len += xgq_ring_len(numslots, slot_size[i]);
	}
	return numslots >> 1;
}

/*
 * XGQ APIs.
 *
 * Typical flow:
 *
 * Client -> xgq_alloc() -> xgq_produce() -> fill-up-SQ-entry -> xgq_notify_peer_produced()
 * 	  -> xgq_consume() -> process-CQ-entry -> xgq_notify_peer_consumed()
 *
 * Server -> xgq_attach() -> xgq_consume() -> process-SQ-entry -> xgq_notify_peer_consumed()
 * 	  -> xgq_produce() -> fill-up-CQ-entry -> xgq_notify_peer_produced()
 *
 * You can call xgq_produce() and fill out entries multiple times before you call
 * xgq_notify_peer_produced(), which then will publish all entries at once to peer.
 */

static inline int
xgq_alloc(struct xgq *xgq, uint64_t flags, uint64_t io_hdl, uint64_t ring_addr, size_t *ring_len,
	  uint32_t slot_size, uint64_t sq_produced, uint64_t cq_produced)
{
	uint32_t numslots = 0;
	size_t rlen = *ring_len;

	if (slot_size % sizeof(uint32_t))
		return -EINVAL;

	numslots = xgq_alloc_num_slots(rlen, &slot_size, 1);
	if (numslots < XGQ_MIN_NUM_SLOTS)
		return -E2BIG;

	xgq_init(xgq, flags, io_hdl, ring_addr, numslots, slot_size, sq_produced, cq_produced);
	*ring_len = xgq_ring_len(numslots, slot_size);
	return 0;
}

/*
 * Alloc a group of XGQs on the ring buffer. Producer pointers will be embedded in the header.
 */
static inline int
xgq_group_alloc(struct xgq *a_xgq, size_t n_qs, uint64_t flags, uint64_t io_hdl, uint64_t ring_addr,
		size_t *ring_len, const uint32_t *a_slot_size, const uint32_t max_slots)
{
	size_t i;
	uint32_t numslots = 0;
	size_t rlen = *ring_len;
	uint64_t raddr = ring_addr;

	/* Only support in-mem producer pointer for group xgq alloc. */
	flags |= XGQ_IN_MEM_PROD;

	for (i = 0; i < n_qs; i++) {
		if (a_slot_size[i] % sizeof(uint32_t))
			return -EINVAL;
	}

	numslots = xgq_alloc_num_slots(rlen, a_slot_size, n_qs);
	if (numslots < XGQ_MIN_NUM_SLOTS)
		return -E2BIG;
	if (max_slots && numslots > max_slots)
		numslots = max_slots;

	for (i = 0; i < n_qs; i++) {
		xgq_init(&a_xgq[i], flags, io_hdl, raddr, numslots, a_slot_size[i], 0, 0);
		raddr += xgq_ring_len(numslots, a_slot_size[i]);
	}

	*ring_len = raddr - ring_addr;
	return 0;
}

static inline int xgq_attach(struct xgq *xgq, uint64_t flags, uint64_t io_hdl, uint64_t ring_addr,
			     uint64_t sq_produced, uint64_t cq_produced)
{
	struct xgq_header hdr = {};
	uint32_t nslots;
	uint64_t sqprod, cqprod;

	xgq_copy_from_ring(xgq->xq_io_hdl, &hdr, ring_addr, sizeof(uint32_t));
	// Magic number must show up to confirm the header is fully initialized
	if (hdr.xh_magic != XGQ_ALLOC_MAGIC)
		return -EAGAIN;

	xgq_copy_from_ring(xgq->xq_io_hdl, &hdr, ring_addr, sizeof(struct xgq_header));
	if (GET_XGQ_MAJOR(hdr.xh_version) != XGQ_MAJOR)
		return -EOPNOTSUPP;

	nslots = hdr.xh_slot_num;
	if ((nslots < XGQ_MIN_NUM_SLOTS) || (nslots & (nslots - 1)))
		return -EPROTO;

	xgq->xq_flags = 0;
	xgq->xq_flags |= flags;
	xgq->xq_flags |= (hdr.xh_flags & XGQ_DOUBLE_READ);
	xgq->xq_flags |= (hdr.xh_flags & XGQ_IN_MEM_PROD);

	if (XGQ_IS_IN_MEM_PROD(xgq)) {
		/* Passed-in sq/cq producer pointer will be ignored. */
		sqprod = ring_addr + offsetof(struct xgq_header, xh_sq_produced);
		cqprod = ring_addr + offsetof(struct xgq_header, xh_cq_produced);
	} else {
		sqprod = sq_produced;
		cqprod = cq_produced;
	}
	xgq_init_ring(xgq, &xgq->xq_sq, sqprod,
		      ring_addr + offsetof(struct xgq_header, xh_sq_consumed),
		      ring_addr + hdr.xh_sq_offset,
		      hdr.xh_slot_num, hdr.xh_sq_slot_size);
	xgq_init_ring(xgq, &xgq->xq_cq, cqprod,
		      ring_addr + offsetof(struct xgq_header, xh_cq_consumed),
		      ring_addr + hdr.xh_cq_offset,
		      hdr.xh_slot_num, sizeof(struct xgq_com_queue_entry));

	xgq_fast_forward(xgq, &xgq->xq_sq);
	xgq_fast_forward(xgq, &xgq->xq_cq);
	return 0;
}

static inline int xgq_produce(struct xgq *xgq, uint64_t *slot_addr)
{
#ifdef XGQ_SERVER
	struct xgq_ring *ring = &xgq->xq_cq;
#else
	struct xgq_ring *ring = &xgq->xq_sq;
#endif

	if (unlikely(!xgq_can_produce(xgq)))
		return -ENOSPC;
	*slot_addr = xgq_ring_slot_ptr_produced(ring);
	ring->xr_produced++;
	return 0;
}

static inline int xgq_consume(struct xgq *xgq, uint64_t *slot_addr)
{
#ifdef XGQ_OUT_OF_ORDER_WRITE
	uint32_t val = 0;
#endif
#ifdef XGQ_SERVER
	struct xgq_ring *ring = &xgq->xq_sq;
#else
	struct xgq_ring *ring = &xgq->xq_cq;
#endif

	if (unlikely(!xgq_can_consume(xgq)))
		return -ENOENT;
	*slot_addr = xgq_ring_slot_ptr_consumed(ring);
	ring->xr_consumed++;

#ifdef XGQ_OUT_OF_ORDER_WRITE
	/*
	 * Producer pointer does not gurantee the slot content is up-to-date.
	 * See comments above XGQ_ENTRY_NEW_FLAG_MASK for details.
	 */
	while (!(val & XGQ_ENTRY_NEW_FLAG_MASK))
		val = xgq_mem_read32(xgq->xq_io_hdl, *slot_addr);
	xgq_mem_write32(xgq->xq_io_hdl, *slot_addr, val & ~XGQ_ENTRY_NEW_FLAG_MASK);
#endif

	return 0;
}

static inline void xgq_notify_peer_produced(struct xgq *xgq)
{
#ifdef XGQ_SERVER
	xgq_ring_write_produced(xgq->xq_io_hdl, &xgq->xq_cq);
#else
	xgq_ring_write_produced(xgq->xq_io_hdl, &xgq->xq_sq);
#endif

}

static inline void xgq_notify_peer_consumed(struct xgq *xgq)
{
#ifdef XGQ_SERVER
	xgq_ring_write_consumed(xgq->xq_io_hdl, &xgq->xq_sq);
#else
	xgq_ring_write_consumed(xgq->xq_io_hdl, &xgq->xq_cq);
#endif
}

#endif
