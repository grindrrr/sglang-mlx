#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/optional.h>
#include <mlx/mlx.h>

#include "mlx_ops.h"

namespace nb = nanobind;
namespace mx = mlx::core;

static PyObject* s_array_type = nullptr;

static mx::array& extract_array(nb::handle h) {
    if (!PyObject_IsInstance(h.ptr(), s_array_type))
        throw nb::type_error("Expected mlx.core.array");
    return *nb::inst_ptr<mx::array>(h);
}

static nb::object wrap_array(mx::array arr) {
    nb::object array_class = nb::borrow<nb::object>(s_array_type);
    nb::object obj = array_class.attr("__new__")(array_class);
    new (nb::inst_ptr<mx::array>(obj)) mx::array(std::move(arr));
    nb::inst_mark_ready(obj);
    return obj;
}

static nb::list wrap_arrays(std::vector<mx::array> arrs) {
    nb::list result;
    for (auto& a : arrs)
        result.append(wrap_array(std::move(a)));
    return result;
}


NB_MODULE(_ext, m) {
    m.doc() = "MLX paged-attention kernels";

    nb::object mlx = nb::module_::import_("mlx.core");
    s_array_type = mlx.attr("array").ptr();
    Py_INCREF(s_array_type);

    m.def("paged_attention_v1",
        [](nb::object query, nb::object key_cache, nb::object value_cache,
           nb::object block_tables, nb::object seq_lens,
           int num_kv_heads, float scale, int block_size, int max_seq_len,
           nb::object alibi_slopes) {
            std::optional<mx::array> alibi;
            if (!alibi_slopes.is_none()) alibi = extract_array(alibi_slopes);
            return wrap_array(paged_attention_v1(
                extract_array(query), extract_array(key_cache),
                extract_array(value_cache), extract_array(block_tables),
                extract_array(seq_lens),
                num_kv_heads, scale, block_size, max_seq_len,
                alibi));
        },
        nb::arg("query"), nb::arg("key_cache"), nb::arg("value_cache"),
        nb::arg("block_tables"), nb::arg("seq_lens"),
        nb::arg("num_kv_heads"), nb::arg("scale"),
        nb::arg("block_size"), nb::arg("max_seq_len"),
        nb::arg("alibi_slopes")   = nb::none(),
        "Single-pass paged attention. Returns out [num_seqs, num_heads, head_size]."
    );

    m.def("reshape_and_cache",
        [](nb::object key, nb::object value,
           nb::object key_cache, nb::object value_cache,
           nb::object slot_mapping) {
            return wrap_arrays(reshape_and_cache(
                extract_array(key), extract_array(value),
                extract_array(key_cache), extract_array(value_cache),
                extract_array(slot_mapping)));
        },
        nb::arg("key"), nb::arg("value"),
        nb::arg("key_cache"), nb::arg("value_cache"),
        nb::arg("slot_mapping"),
        "Reshape KV tokens into paged cache. Returns [new_key_cache, new_value_cache]."
    );

    m.def("reshape_and_cache_flash",
        [](nb::object key, nb::object value,
           nb::object key_cache, nb::object value_cache,
           nb::object slot_mapping) {
            return wrap_arrays(reshape_and_cache_flash(
                extract_array(key), extract_array(value),
                extract_array(key_cache), extract_array(value_cache),
                extract_array(slot_mapping)));
        },
        nb::arg("key"), nb::arg("value"),
        nb::arg("key_cache"), nb::arg("value_cache"),
        nb::arg("slot_mapping"),
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

    m.def("get_device_attribute",
        &get_device_attribute,
        nb::arg("attribute"), nb::arg("device_id") = 0
    );

    m.def("get_max_shared_memory_per_block_device_attribute",
        &get_max_shared_memory_per_block_device_attribute,
        nb::arg("device_id") = 0
    );
}
