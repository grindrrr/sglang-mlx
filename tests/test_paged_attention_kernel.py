"""Tests for Metal paged attention kernels.

Validates reshape_and_cache (scatter-write), paged_attention_v1
(attention computation), and cache block-copy operations.

All tests require the compiled Metal extension (_ext) and are skipped
when it is not available.
"""

import math

import mlx.core as mx
import numpy as np
import pytest

# Attempt to import the extension
try:
    from sglang_mlx import _ext

    HAS_EXT = True
except ImportError:
    HAS_EXT = False

# Skip all tests if extension is missing
pytestmark = pytest.mark.skipif(not HAS_EXT, reason="sglang_mlx._ext not found")


# ---------------------------------------------------------------------------
# Reference Implementations (NumPy)
# ---------------------------------------------------------------------------


def ref_reshape_and_cache(
    k: np.ndarray,  # [num_tokens, num_heads, head_dim]
    v: np.ndarray,  # [num_tokens, num_heads, head_dim]
    k_cache: np.ndarray,  # [num_blocks, num_heads, head_dim/x, block_size, x]
    v_cache: np.ndarray,  # [num_blocks, num_heads, head_dim, block_size]
    slot_mapping: np.ndarray,  # [num_tokens]
) -> tuple[np.ndarray, np.ndarray]:
    num_tokens = k.shape[0]
    num_heads = k.shape[1]
    head_dim = k.shape[2]
    block_size = v_cache.shape[3]
    x = k_cache.shape[4]

    for i in range(num_tokens):
        slot = slot_mapping[i]
        if slot < 0:
            continue
        block_idx = slot // block_size
        block_offset = slot % block_size

        # Key: [num_blocks, num_heads, head_dim/x, block_size, x]
        # Store as: k_cache[block_idx, head_idx, d/x, block_offset, d%x]
        for h in range(num_heads):
            for d in range(head_dim):
                k_cache[block_idx, h, d // x, block_offset, d % x] = k[i, h, d]

        # Value: [num_blocks, num_heads, head_dim, block_size]
        # Store as: v_cache[block_idx, head_idx, d, block_offset]
        for h in range(num_heads):
            for d in range(head_dim):
                v_cache[block_idx, h, d, block_offset] = v[i, h, d]

    return k_cache, v_cache


def ref_paged_attention(
    q: np.ndarray,  # [num_seqs, num_heads, head_dim]
    k_cache: np.ndarray,  # [num_blocks, num_heads, head_dim/x, block_size, x]
    v_cache: np.ndarray,  # [num_blocks, num_heads, head_dim, block_size]
    block_tables: np.ndarray,  # [num_seqs, max_blocks]
    context_lens: np.ndarray,  # [num_seqs]
    scale: float,
    num_kv_heads: int,
) -> np.ndarray:
    num_seqs, num_heads, head_dim = q.shape
    block_size = v_cache.shape[3]
    x = k_cache.shape[4]
    num_queries_per_kv = num_heads // num_kv_heads

    out = np.zeros_like(q)

    for i in range(num_seqs):
        seq_len = context_lens[i]
        for h in range(num_heads):
            kv_h = h // num_queries_per_kv
            q_vec = q[i, h]

            # Gather keys and values
            keys = []
            values = []
            for t in range(seq_len):
                block_idx = block_tables[i, t // block_size]
                block_offset = t % block_size

                # Fetch k: [d/x, block_offset, d%x] -> [d]
                k_vec = np.zeros(head_dim)
                for d in range(head_dim):
                    k_vec[d] = k_cache[block_idx, kv_h, d // x, block_offset, d % x]
                keys.append(k_vec)

                # Fetch v: [d, block_offset] -> [d]
                v_vec = v_cache[block_idx, kv_h, :, block_offset]
                values.append(v_vec)

            keys = np.array(keys)  # [seq_len, head_dim]
            values = np.array(values)  # [seq_len, head_dim]

            # SDPA
            logits = np.dot(keys, q_vec) * scale

            # Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)

            # Output
            out[i, h] = np.dot(probs, values)

    return out


# ---------------------------------------------------------------------------
# Test Fixtures & Cases
# ---------------------------------------------------------------------------


class TestPagedAttentionKernels:
    # Common shapes
    N_SEQS = 2
    N_HEADS = 8
    N_KV_HEADS = 4
    HEAD_DIM = 64
    BLOCK_SIZE = 16
    MAX_BLOCKS_PER_SEQ = 4
    SCALE = 1.0 / math.sqrt(64)

    @pytest.fixture
    def setup_data(self):
        # 1. Inputs
        q = mx.random.normal((self.N_SEQS, self.N_HEADS, self.HEAD_DIM))
        k = mx.random.normal((self.N_SEQS * 32, self.N_KV_HEADS, self.HEAD_DIM))
        v = mx.random.normal((self.N_SEQS * 32, self.N_KV_HEADS, self.HEAD_DIM))

        # 2. Cache buffers
        num_total_blocks = 20
        # x = 16 / sizeof(float16) = 8
        x = 8
        k_cache = mx.zeros(
            (num_total_blocks, self.N_KV_HEADS, self.HEAD_DIM // x, self.BLOCK_SIZE, x),
            dtype=mx.float16,
        )
        v_cache = mx.zeros(
            (num_total_blocks, self.N_KV_HEADS, self.HEAD_DIM, self.BLOCK_SIZE),
            dtype=mx.float16,
        )

        # 3. Mappings
        # Sequence lengths: 25, 50
        context_lens = mx.array([25, 50], dtype=mx.int32)
        # Block table: sequential assignment for simplicity
        # Seq 0 uses blocks 0, 1
        # Seq 1 uses blocks 2, 3, 4, 5
        block_tables = mx.array(
            [[0, 1, -1, -1], [2, 3, 4, 5]],
            dtype=mx.int32,
        )

        # Slot mapping for reshape_and_cache
        # We'll map the first 25 tokens to blocks 0, 1 and next 50 to 2, 3, 4, 5
        slots_seq0 = [0 * self.BLOCK_SIZE + i for i in range(25)]
        slots_seq1 = [2 * self.BLOCK_SIZE + i for i in range(50)]
        slot_mapping = mx.array(slots_seq0 + slots_seq1, dtype=mx.int32)

        # Truncate K, V to match total context tokens (75)
        k = k[:75]
        v = v[:75]

        return {
            "q": q,
            "k": k,
            "v": v,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "context_lens": context_lens,
            "block_tables": block_tables,
            "slot_mapping": slot_mapping,
        }

    def test_reshape_and_cache(self, setup_data):
        d = setup_data
        # Run kernel
        k_out, v_out = _ext.reshape_and_cache(
            d["k"], d["v"], d["k_cache"], d["v_cache"], d["slot_mapping"]
        )
        mx.eval(k_out, v_out)

        # Run reference
        k_np = np.array(d["k"])
        v_np = np.array(d["v"])
        k_cache_np = np.array(d["k_cache"])
        v_cache_np = np.array(d["v_cache"])
        slot_mapping_np = np.array(d["slot_mapping"])

        ref_k, ref_v = ref_reshape_and_cache(
            k_np, v_np, k_cache_np, v_cache_np, slot_mapping_np
        )

        np.testing.assert_allclose(np.array(k_out), ref_k, atol=1e-5)
        np.testing.assert_allclose(np.array(v_out), ref_v, atol=1e-5)

    def test_paged_attention_v1(self, setup_data):
        d = setup_data

        # 1. Fill cache first
        k_cache, v_cache = _ext.reshape_and_cache(
            d["k"], d["v"], d["k_cache"], d["v_cache"], d["slot_mapping"]
        )

        # 2. Run V1 kernel
        out = _ext.paged_attention_v1(
            d["q"],
            k_cache,
            v_cache,
            d["block_tables"],
            d["context_lens"],
            self.N_KV_HEADS,
            self.SCALE,
            self.BLOCK_SIZE,
            64,  # max_seq_len
        )
        mx.eval(out)

        # 3. Reference (NumPy)
        ref = ref_paged_attention(
            np.array(d["q"]),
            np.array(k_cache),
            np.array(v_cache),
            np.array(d["block_tables"]),
            np.array(d["context_lens"]),
            self.SCALE,
            self.N_KV_HEADS,
        )

        # Allow small tolerance for float16 accumulation differences
        np.testing.assert_allclose(np.array(out), ref, atol=5e-3)

    @pytest.mark.parametrize("head_dim", [32, 128])
    @pytest.mark.parametrize("block_size", [8, 16])
    def test_various_shapes(self, head_dim, block_size):
        # Mini test for various configurations
        scale = 1.0 / math.sqrt(head_dim)
        q = mx.random.normal((1, 4, head_dim))
        k = mx.random.normal((16, 2, head_dim))
        v = mx.random.normal((16, 2, head_dim))

        k_cache = mx.zeros((4, 2, head_dim // 8, block_size, 8), dtype=mx.float16)
        v_cache = mx.zeros((4, 2, head_dim, block_size), dtype=mx.float16)

        context_lens = mx.array([16], dtype=mx.int32)
        block_tables = mx.array(
            [[0, 1, -1, -1] if block_size == 8 else [0, -1, -1, -1]], dtype=mx.int32
        )
        slot_mapping = mx.array(np.arange(16), dtype=mx.int32)

        k_c, v_c = _ext.reshape_and_cache(k, v, k_cache, v_cache, slot_mapping)

        paged = _ext.paged_attention_v1(
            q, k_c, v_c, block_tables, context_lens, 2, scale, block_size, 16
        )
        mx.eval(paged)

        ref = ref_paged_attention(
            np.array(q),
            np.array(k_c),
            np.array(v_c),
            np.array(block_tables),
            np.array(context_lens),
            scale,
            2,
        )
        np.testing.assert_allclose(np.array(paged), ref, atol=5e-3)

    def test_context_longer_than_legacy_2048_limit(self):
        head_dim = 32
        block_size = 16
        seq_len = 2112
        num_heads = 2
        num_kv_heads = 1
        scale = 1.0 / math.sqrt(head_dim)

        q = mx.random.normal((1, num_heads, head_dim))
        k = mx.random.normal((seq_len, num_kv_heads, head_dim))
        v = mx.random.normal((seq_len, num_kv_heads, head_dim))

        num_blocks = math.ceil(seq_len / block_size)
        k_cache = mx.zeros(
            (num_blocks, num_kv_heads, head_dim // 8, block_size, 8), dtype=mx.float16
        )
        v_cache = mx.zeros(
            (num_blocks, num_kv_heads, head_dim, block_size), dtype=mx.float16
        )
        block_tables = mx.array(np.arange(num_blocks, dtype=np.uint32).reshape(1, -1))
        context_lens = mx.array([seq_len], dtype=mx.uint32)
        slot_mapping = mx.array(np.arange(seq_len, dtype=np.int32))

        k_c, v_c = _ext.reshape_and_cache(k, v, k_cache, v_cache, slot_mapping)
        out = _ext.paged_attention_v1(
            q,
            k_c,
            v_c,
            block_tables,
            context_lens,
            num_kv_heads,
            scale,
            block_size,
            seq_len,
        )
        mx.eval(out)

        ref = ref_paged_attention(
            np.array(q),
            np.array(k_c),
            np.array(v_c),
            np.array(block_tables),
            np.array(context_lens),
            scale,
            num_kv_heads,
        )
        np.testing.assert_allclose(np.array(out), ref, atol=5e-3)

    def test_empty_context_returns_zero(self):
        head_dim = 32
        block_size = 16
        q = mx.random.normal((1, 2, head_dim))
        k_cache = mx.zeros((1, 1, head_dim // 8, block_size, 8), dtype=mx.float16)
        v_cache = mx.zeros((1, 1, head_dim, block_size), dtype=mx.float16)
        block_tables = mx.array([[0]], dtype=mx.int32)
        context_lens = mx.array([0], dtype=mx.int32)

        out = _ext.paged_attention_v1(
            q,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            1,
            1.0 / math.sqrt(head_dim),
            block_size,
            block_size,
        )
        mx.eval(out)

        np.testing.assert_array_equal(np.array(out), np.zeros(out.shape))

    def test_context_over_max_seq_len_returns_zero(self):
        head_dim = 32
        block_size = 16
        q = mx.random.normal((1, 2, head_dim))
        k_cache = mx.zeros((1, 1, head_dim // 8, block_size, 8), dtype=mx.float16)
        v_cache = mx.zeros((1, 1, head_dim, block_size), dtype=mx.float16)
        block_tables = mx.array([[0]], dtype=mx.int32)
        context_lens = mx.array([block_size], dtype=mx.int32)

        out = _ext.paged_attention_v1(
            q,
            k_cache,
            v_cache,
            block_tables,
            context_lens,
            1,
            1.0 / math.sqrt(head_dim),
            block_size,
            block_size // 2,
        )
        mx.eval(out)

        np.testing.assert_array_equal(np.array(out), np.zeros(out.shape))

    def test_rejects_max_seq_len_beyond_block_table_capacity(self):
        head_dim = 32
        block_size = 16
        q = mx.zeros((1, 2, head_dim))
        k_cache = mx.zeros((2, 1, head_dim // 8, block_size, 8), dtype=mx.float16)
        v_cache = mx.zeros((2, 1, head_dim, block_size), dtype=mx.float16)

        with pytest.raises(ValueError, match="block table capacity"):
            _ext.paged_attention_v1(
                q,
                k_cache,
                v_cache,
                mx.array([[0]], dtype=mx.int32),
                mx.array([1], dtype=mx.int32),
                1,
                1.0,
                block_size,
                block_size + 1,
            )

    def test_copy_blocks_multiple_layers(self):
        key_caches = []
        value_caches = []
        key_before = []
        value_before = []
        for layer in range(2):
            key_np = np.arange(24, dtype=np.float32).reshape(4, 2, 3) + layer * 100
            value_np = key_np + 1000
            key_before.append(key_np.copy())
            value_before.append(value_np.copy())
            key_caches.append(mx.array(key_np))
            value_caches.append(mx.array(value_np))

        mapping = mx.array([[0, 2], [1, 3]], dtype=mx.int32)
        new_keys, new_values = _ext.copy_blocks(key_caches, value_caches, mapping)
        mx.eval(*new_keys, *new_values)

        for layer in range(2):
            expected_key = key_before[layer].copy()
            expected_value = value_before[layer].copy()
            expected_key[2] = key_before[layer][0]
            expected_key[3] = key_before[layer][1]
            expected_value[2] = value_before[layer][0]
            expected_value[3] = value_before[layer][1]
            np.testing.assert_array_equal(np.array(new_keys[layer]), expected_key)
            np.testing.assert_array_equal(np.array(new_values[layer]), expected_value)

    def test_swap_blocks_noncontiguous_mapping(self):
        src_np = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
        dst_np = np.full((4, 2, 3), -1, dtype=np.float32)
        mapping = mx.array([[0, 2], [3, 1]], dtype=mx.int32)

        out = _ext.swap_blocks(mx.array(src_np), mx.array(dst_np), mapping)
        mx.eval(out)

        expected = dst_np.copy()
        expected[2] = src_np[0]
        expected[1] = src_np[3]
        np.testing.assert_array_equal(np.array(out), expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
