// This file is ported from EricB/kernels-paged-attention-metal (https://huggingface.co/EricB/kernels-paged-attention-metal)
// Modified for MLX integration

// bindings.cpp — nanobind Python bindings for the paged-attention MLX extension.
//
// mx::array is a header-only type, so each .so gets a separate typeinfo address.
// Nanobind's type registry lookup (which uses typeid) therefore fails when our
// extension tries to cast Python mlx.core.array objects to C++ mx::array.
//
// Workaround:
//   INPUT  — accept nb::object, then use nb::inst_ptr<mx::array>() which reads
//             the C++ value from the nb_inst header without a typeid comparison.
//   OUTPUT — use nb::inst_alloc_zero() on mlx.core.array + nb::inst_move() to
//             build a Python wrapper for a C++ mx::array without typeid.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/optional.h>
#include <mlx/mlx.h>

#include "mlx_ops.h"

namespace nb = nanobind;
namespace mx = mlx::core;

// Python type object for mlx.core.array — set in NB_MODULE init.
static PyObject* s_array_type = nullptr;

// Extract a C++ mx::array reference from a Python mlx.core.array object.
static mx::array& extract_array(nb::handle h) {
    if (!PyObject_IsInstance(h.ptr(), s_array_type))
        throw nb::type_error("Expected mlx.core.array");
    return *nb::inst_ptr<mx::array>(h);
}

// Wrap a C++ mx::array as a Python mlx.core.array object.
//
// We must call array.__new__(array) via Python to allocate the instance.
// This ensures the Python object is registered in MLX's own inst_c2p table
// (inside mlx.core.so's nb_internals), so that mlx.core.so's nb_type_dealloc
// can find and destroy it correctly. Using nb::inst_alloc_zero from our module
// would register in a different inst_c2p, causing "unknown instance" crashes at GC.
static nb::object wrap_array(mx::array arr) {
    nb::object array_class = nb::borrow<nb::object>(s_array_type);
    // __new__ calls mlx.core.so's inst_new_int → registers in MLX's inst_c2p.
    nb::object obj = array_class.attr("__new__")(array_class);
    new (nb::inst_ptr<mx::array>(obj)) mx::array(std::move(arr));
    nb::inst_mark_ready(obj);
    return obj;
}

// Wrap a vector of C++ arrays as a Python list of mlx.core.array objects.
static nb::list wrap_arrays(std::vector<mx::array> arrs) {
    nb::list result;
    for (auto& a : arrs)
        result.append(wrap_array(std::move(a)));
    return result;
}


NB_MODULE(_ext, m) {
    m.doc() = "MLX paged-attention kernels";

    // Cache mlx.core.array Python type for use in extract/wrap helpers.
    nb::object mlx = nb::module_::import_("mlx.core");
    s_array_type = mlx.attr("array").ptr();
    Py_INCREF(s_array_type);

    // ── Paged Attention ───────────────────────────────────────────────────────

    m.def("paged_attention_v1",
        [](nb::object query, nb::object key_cache, nb::object value_cache,
           nb::object block_tables, nb::object seq_lens,
           int num_kv_heads, float scale, int block_size, int max_seq_len,
           nb::object alibi_slopes,
           const std::string& kv_cache_dtype, float k_scale, float v_scale) {
            std::optional<mx::array> alibi;
            if (!alibi_slopes.is_none()) alibi = extract_array(alibi_slopes);
            return wrap_array(paged_attention_v1(
                extract_array(query), extract_array(key_cache),
                extract_array(value_cache), extract_array(block_tables),
                extract_array(seq_lens),
                num_kv_heads, scale, block_size, max_seq_len,
                alibi, kv_cache_dtype, k_scale, v_scale));
        },
        nb::arg("query"), nb::arg("key_cache"), nb::arg("value_cache"),
        nb::arg("block_tables"), nb::arg("seq_lens"),
        nb::arg("num_kv_heads"), nb::arg("scale"),
        nb::arg("block_size"), nb::arg("max_seq_len"),
        nb::arg("alibi_slopes")   = nb::none(),
        nb::arg("kv_cache_dtype") = "float",
        nb::arg("k_scale")        = 1.0f,
        nb::arg("v_scale")        = 1.0f,
        "Single-pass paged attention. Returns out [num_seqs, num_heads, head_size]."
    );

    m.def("paged_attention_v2",
        [](nb::object query, nb::object key_cache, nb::object value_cache,
           nb::object block_tables, nb::object seq_lens,
           int num_kv_heads, float scale, int block_size, int max_seq_len,
           int max_num_partitions,
           nb::object alibi_slopes,
           const std::string& kv_cache_dtype, float k_scale, float v_scale) {
            std::optional<mx::array> alibi;
            if (!alibi_slopes.is_none()) alibi = extract_array(alibi_slopes);
            auto outs = paged_attention_v2(
                extract_array(query), extract_array(key_cache),
                extract_array(value_cache), extract_array(block_tables),
                extract_array(seq_lens),
                num_kv_heads, scale, block_size, max_seq_len, max_num_partitions,
                alibi, kv_cache_dtype, k_scale, v_scale);
            return wrap_arrays(std::move(outs));
        },
        nb::arg("query"), nb::arg("key_cache"), nb::arg("value_cache"),
        nb::arg("block_tables"), nb::arg("seq_lens"),
        nb::arg("num_kv_heads"), nb::arg("scale"),
        nb::arg("block_size"), nb::arg("max_seq_len"),
        nb::arg("max_num_partitions"),
        nb::arg("alibi_slopes")   = nb::none(),
        nb::arg("kv_cache_dtype") = "float",
        nb::arg("k_scale")        = 1.0f,
        nb::arg("v_scale")        = 1.0f,
        "Two-pass paged attention. Returns [out, exp_sums, max_logits, tmp_out]."
    );

    // ── Cache Operations ──────────────────────────────────────────────────────

    m.def("reshape_and_cache",
        [](nb::object key, nb::object value,
           nb::object key_cache, nb::object value_cache,
           nb::object slot_mapping,
           const std::string& kv_cache_dtype, float k_scale, float v_scale) {
            return wrap_arrays(reshape_and_cache(
                extract_array(key), extract_array(value),
                extract_array(key_cache), extract_array(value_cache),
                extract_array(slot_mapping),
                kv_cache_dtype, k_scale, v_scale));
        },
        nb::arg("key"), nb::arg("value"),
        nb::arg("key_cache"), nb::arg("value_cache"),
        nb::arg("slot_mapping"),
        nb::arg("kv_cache_dtype") = "float",
        nb::arg("k_scale")        = 1.0f,
        nb::arg("v_scale")        = 1.0f,
        "Reshape KV tokens into paged cache. Returns [new_key_cache, new_value_cache]."
    );

    m.def("reshape_and_cache_flash",
        [](nb::object key, nb::object value,
           nb::object key_cache, nb::object value_cache,
           nb::object slot_mapping,
           const std::string& kv_cache_dtype, float k_scale, float v_scale) {
            return wrap_arrays(reshape_and_cache_flash(
                extract_array(key), extract_array(value),
                extract_array(key_cache), extract_array(value_cache),
                extract_array(slot_mapping),
                kv_cache_dtype, k_scale, v_scale));
        },
        nb::arg("key"), nb::arg("value"),
        nb::arg("key_cache"), nb::arg("value_cache"),
        nb::arg("slot_mapping"),
        nb::arg("kv_cache_dtype") = "float",
        nb::arg("k_scale")        = 1.0f,
        nb::arg("v_scale")        = 1.0f,
        "Flash-layout cache reshape. Returns [new_key_cache, new_value_cache]."
    );

    m.def("copy_blocks",
        [](std::vector<nb::object> key_caches_py,
           std::vector<nb::object> value_caches_py,
           nb::object block_mapping) {
            std::vector<mx::array> key_caches, value_caches;
            key_caches.reserve(key_caches_py.size());
            value_caches.reserve(value_caches_py.size());
            for (auto& o : key_caches_py)   key_caches.push_back(extract_array(o));
            for (auto& o : value_caches_py) value_caches.push_back(extract_array(o));
            auto [new_k, new_v] = copy_blocks(key_caches, value_caches,
                                               extract_array(block_mapping));
            return nb::make_tuple(wrap_arrays(std::move(new_k)),
                                  wrap_arrays(std::move(new_v)));
        },
        nb::arg("key_caches"), nb::arg("value_caches"), nb::arg("block_mapping"),
        "Copy cache blocks across layers. Returns (new_key_caches, new_value_caches)."
    );

    m.def("swap_blocks",
        [](nb::object src, nb::object dst, nb::object block_mapping) {
            return wrap_array(swap_blocks(
                extract_array(src), extract_array(dst),
                extract_array(block_mapping)));
        },
        nb::arg("src"), nb::arg("dst"), nb::arg("block_mapping"),
        "Copy src blocks into dst (blit). Returns new_dst."
    );

    // ── FP8 Conversion ────────────────────────────────────────────────────────

    m.def("convert_fp8",
        [](nb::object src_cache, float scale,
           const std::string& kv_cache_dtype,
           const std::string& dst_dtype_str) {
            return wrap_array(convert_fp8(
                extract_array(src_cache), scale, kv_cache_dtype, dst_dtype_str));
        },
        nb::arg("src_cache"),
        nb::arg("scale")          = 1.0f,
        nb::arg("kv_cache_dtype") = "fp8",
        nb::arg("dst_dtype_str")  = std::string("float16"),
        "Convert between FP8 (uint8) and float/half/bfloat16. Returns converted array."
    );

    // ── Device ────────────────────────────────────────────────────────────────

    m.def("get_device_attribute",
        &get_device_attribute,
        nb::arg("attribute"), nb::arg("device_id") = 0
    );

    m.def("get_max_shared_memory_per_block_device_attribute",
        &get_max_shared_memory_per_block_device_attribute,
        nb::arg("device_id") = 0
    );
}
