"""Ordinary inline helper control flow and its composition with model levels."""
from __future__ import annotations
import triton
import triton.language as tl


@triton.jit
def branch_adjust(value, mode):
    if mode < 0:
        return value + 1.0
    if mode == 0:
        return value + 2.0
    return value + 3.0


@triton.jit(model="weft")
def helper_control(X, Y, N: tl.constexpr):
    i = tl.program_id(0)
    value = tl.load(X + i)
    mode = i % 3 - 1
    value = branch_adjust(value, mode)
    value = value + (1 - mode).to(tl.float32)
    tl.store(Y + i, value * 0.25)


@triton.jit
def tuple_scale(value, iteration):
    if iteration == 0:
        local = value * 0.5
        return (local, (1, 0.25))
    # Locals belong to the returning path; their types need not match.
    local = iteration + 1
    return (value * 0.5, (local, 0.25))


@triton.jit
def constant_adjust(mode):
    if mode < 0:
        return 1
    if mode == 0:
        return 2
    return 3


@triton.jit
def static_width(small: tl.constexpr):
    if small:
        return 16
    return 32


@triton.jit
def static_dtype():
    return tl.float32


@triton.jit
def store_adjust(pointer, value, mode):
    tl.store(pointer, value)
    if mode < 0:
        tl.store(pointer, (value + 1.0) * 0.25)
        return
    value += 2.0
    tl.store(pointer, value)
    if mode == 0:
        tl.store(pointer, value * 0.25)
        return
    tl.store(pointer, (value + 1.0) * 0.25)


@triton.jit(model="weft")
def helper_composition(X, Y, N: tl.constexpr, MODE: tl.constexpr):
    if MODE == 3:
        width: tl.constexpr = static_width(True)
        dtype: tl.constexpr = static_dtype()
        tl.static_assert(width == 16)
        tl.static_assert(dtype == tl.float32)
        i = tl.program_id(0) * width + tl.arange(0, width)
        value = tl.load(X + i, i < N, other=0)
        value = branch_adjust(value, tl.program_id(0) % 3 - 1)
        value += (1 - (tl.program_id(0) % 3 - 1)).to(dtype)
        tl.store(Y + i, value * 0.25, i < N)
    else:
        i = tl.program_id(0)
        value = tl.load(X + i)
        if MODE == 0:
            for step in range(3):
                value = branch_adjust(value, step - 1)
            value = (value - 3.0) * 0.25
        elif MODE == 1:
            value += 3.0
            step = 0
            while step < 2:
                scale: tl.constexpr
                value, (step, scale) = tuple_scale(value, step)
                tl.static_assert(scale == 0.25)
        elif MODE == 2:
            mode = i % 3 - 1
            adjustment = constant_adjust(mode)
            value = (value + adjustment.to(tl.float32) + (1 - mode).to(tl.float32)) * 0.25
        elif MODE == 4:
            mode = i % 3 - 1
            store_adjust(Y + i, value + (1 - mode).to(tl.float32), mode)
            return
        else:
            tl.static_assert(False, "unknown helper composition")
        tl.store(Y + i, value)


@triton.jit(model="weft")
def helper_levels(X: tl.View[tl.f32, (N,)], Y: tl.View[tl.f32, (N,)]):
    with tl.level.blocks(N, extent=32) as block:
        with tl.level.subtiles(block, extent=16) as sub:
            panel = tl.stage(tl.load(X[sub]))
            acc = tl.state(tl.f32, [16], init=0)
            for step in range(3):
                acc += branch_adjust(panel, step - 1) - panel
            result = (panel + acc - 3.0) * 0.25
            tl.store(Y[sub], result)


@triton.jit
def mismatched_returns(value, mode):
    if mode > 0:
        return value
    return value.to(tl.int32)


@triton.jit
def missing_return(value, mode):
    if mode > 0:
        return value


@triton.jit
def loop_return(value, mode):
    for i in range(mode):
        if i > 0:
            return value
    return value


@triton.jit
def mismatched_tuple(value, mode):
    if mode > 0:
        return value, value
    return (value,)


@triton.jit(model="weft")
def invalid_helper(X, Y, N: tl.constexpr, MODE: tl.constexpr):
    i = tl.program_id(0)
    value = tl.load(X + i)
    if MODE == 0:
        result = mismatched_returns(value, i)
    elif MODE == 1:
        result = missing_return(value, i)
    elif MODE == 2:
        result = loop_return(value, i)
    else:
        result = mismatched_tuple(value, i)[0]
    tl.store(Y + i, result)
