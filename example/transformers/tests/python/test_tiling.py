#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import math
import time

import numpy as np
import pytest

np.random.seed(123)


def run(a, b):
    """this is what happens in AIE for
    kernel_x_shape, kernel_y_shape"""
    return np.matmul(a, b.transpose())


def execute(a, b, debug=False):
    global kernel_x_shape, kernel_y_shape
    ra_orig = a.shape[0]
    rb_orig = b.shape[0]
    if debug:
        print(f"a: {a}")
        print(f"b: {b}")

    def ceil_for_me(x, y):
        """Return nearest value of x that is a multiple of y"""
        return y * math.ceil(x * 1.0 / y)

    # pad to the nearest multiple of the GEMM Size we are offloading to aie
    a_new_shape = (
        ceil_for_me(a.shape[0], kernel_x_shape[0]),
        ceil_for_me(a.shape[1], kernel_x_shape[1]),
    )
    b_new_shape = (
        ceil_for_me(b.shape[0], kernel_y_shape[0]),
        ceil_for_me(b.shape[1], kernel_y_shape[1]),
    )
    a_pad_shape = ((0, a_new_shape[0] - a.shape[0]), (0, a_new_shape[1] - a.shape[1]))
    b_pad_shape = ((0, b_new_shape[0] - b.shape[0]), (0, b_new_shape[1] - b.shape[1]))
    a_new = np.pad(a, a_pad_shape, mode="constant", constant_values=0)
    b_new = np.pad(b, b_pad_shape, mode="constant", constant_values=0)
    if debug:
        print(f"a_new: {a_new}")
        print(f"b_new: {b_new}")

    c_new = np.zeros((a_new.shape[0], b_new.shape[0])).astype(np.int32)
    if debug:
        print(f"After padding: {a_new.shape} {b_new.shape} {c_new.shape}")

    for ra in range(0, a_new.shape[0], kernel_x_shape[0]):
        for rb in range(0, b_new.shape[0], kernel_y_shape[0]):
            for ca in range(0, a_new.shape[1], kernel_x_shape[1]):
                cb = ca
                if debug:
                    print("*" * 20)
                    print(
                        f"indices : ra:{ra}, cb:{cb}, ca:{ca}  a:[{ra}:{ra+ kernel_x_shape[0]}, {ca}:{ca + kernel_x_shape[1]}] b:[{rb}:{rb + kernel_y_shape[0]}, {cb}:{cb+kernel_y_shape[1]}] c:[{ra}:{ra + kernel_x_shape[0]}, {rb}:{rb + kernel_y_shape[0]}] "
                    )
                a_tile = a_new[
                    ra : ra + kernel_x_shape[0], ca : ca + kernel_x_shape[1]
                ].astype(np.int32)
                b_tile = b_new[
                    rb : rb + kernel_y_shape[0], cb : cb + kernel_y_shape[1]
                ].astype(np.int32)
                res = run(a_tile, b_tile)
                c_new[ra : ra + kernel_x_shape[0], rb : rb + kernel_y_shape[0]] += res
                if debug:
                    print(f"a_tile: {a_tile}")
                    print(f"b_tile: {b_tile}")
                    print(f"res: {res}")
                    print(f"c_new[]: {c_new}")
                    input("Enter a key ")
    if debug:
        print(f"c_new: {c_new}")
    c = c_new[:ra_orig, :rb_orig]
    return c


def test_tiling():
    test_case = 1
    global kernel_x_shape, kernel_y_shape
    if test_case == 1:
        kernel_x_shape = (4, 2048)
        kernel_y_shape = (2048, 2048)
        x_shapes = [
            (1, 10),
            (8, 2048),
            (8, 2048),
            (8, 8192),
            (8, 2048),
            (1, 2048),
            (1, 2048),
            (1, 8192),
            (1, 2048),
        ]
        y_shapes = [
            (10, 10),
            (2048, 2048),
            (8192, 2048),
            (2048, 8192),
            (50272, 2048),
            (2048, 2048),
            (8192, 2048),
            (2048, 8192),
            (50272, 2048),
        ]
    else:
        kernel_x_shape = (4, 2)
        kernel_y_shape = (2, 2)
        x_shapes = [(3, 3), (5, 6)]
        y_shapes = [(4, 3), (7, 6)]
        # x_shapes = [(8, 2)]
        # y_shapes = [(8, 2)]

    fail_count = 0
    total = 0
    for x_shape, y_shape in zip(x_shapes, y_shapes):
        print(f"Testing for shapes: {x_shape, y_shape}")
        xq = np.random.randint(-128, 128, x_shape).astype(np.int8)
        yq = np.random.randint(-128, 128, y_shape).astype(np.int8)
        # print(f"xq: {xq}")
        # print(f"yq: {yq}")
        zq_tiling = execute(xq.astype(np.int32), yq.astype(np.int32), debug=False)
        zq_numpy = np.matmul(xq.astype(np.int32), np.transpose(yq.astype(np.int32)))
        try:
            np.testing.assert_array_equal(zq_tiling, zq_numpy)
            print("PASS")
        except BaseException:
            print("FAIL")
            fail_count += 1
        total += 1
    print(f"Tiling PASSED {total-fail_count} out of {total} test configs.")
    assert fail_count == 0
