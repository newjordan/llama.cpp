#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama-kv-cache.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

static void add_cell(llama_kv_cells & cells, uint32_t row, llama_pos pos, std::initializer_list<llama_seq_id> seqs) {
    assert(cells.is_empty(row));
    cells.pos_set(row, pos);
    for (const auto seq : seqs) {
        cells.seq_add(row, seq);
    }
}

static void test_shared_prefix_private_tails_are_ragged() {
    llama_kv_cells cells;
    cells.resize(1024);

    for (uint32_t i = 0; i < 128; ++i) {
        add_cell(cells, i, i, { 3, 7 });
    }
    for (uint32_t i = 0; i < 64; ++i) {
        add_cell(cells, 512 + i, 128 + i, { 3 });
        add_cell(cells, 768 + i, 128 + i, { 7 });
    }

    const auto plan = llama_kv_build_ragged_plan(cells, { 3, 7 }, 256);
    const uint32_t invalid = std::numeric_limits<uint32_t>::max();

    assert(plan.has_shared_prefix);
    assert(plan.reduces_columns);
    assert(plan.union_rows == 256);
    assert(plan.n_kv == 256);
    assert(plan.dense_n_kv == 1024);
    assert(plan.rows.size() == 2*plan.n_kv);

    // Both streams alias the exact same shared prefix rows.
    for (uint32_t i = 0; i < 128; ++i) {
        assert(plan.rows[i] == i);
        assert(plan.rows[plan.n_kv + i] == i);
    }
    // Each stream then sees only its private copy-on-write tail.
    for (uint32_t i = 0; i < 64; ++i) {
        assert(plan.rows[128 + i] == 512 + i);
        assert(plan.rows[plan.n_kv + 128 + i] == 768 + i);
    }
    assert(plan.rows[192] == invalid);
    assert(plan.rows[plan.n_kv + 192] == invalid);
}

static void test_unrelated_sequences_do_not_activate_tree_path() {
    llama_kv_cells cells;
    cells.resize(1024);

    for (uint32_t i = 0; i < 64; ++i) {
        add_cell(cells, i, i, { 1 });
        add_cell(cells, 512 + i, i, { 2 });
    }

    const auto plan = llama_kv_build_ragged_plan(cells, { 1, 2 }, 256);
    assert(!plan.has_shared_prefix);
    assert(!plan.reduces_columns);
}

static void test_shared_dense_family_stays_on_existing_graph() {
    llama_kv_cells cells;
    cells.resize(1024);

    for (uint32_t i = 0; i < 128; ++i) {
        add_cell(cells, i, i, { 1, 2 });
    }
    for (uint32_t i = 0; i < 32; ++i) {
        add_cell(cells, 128 + 2*i, 128 + i, { 1 });
        add_cell(cells, 129 + 2*i, 128 + i, { 2 });
    }

    const auto plan = llama_kv_build_ragged_plan(cells, { 1, 2 }, 256);
    assert(plan.has_shared_prefix);
    assert(plan.n_kv == 256);
    assert(plan.dense_n_kv == 256);
    assert(!plan.reduces_columns);
}

static void test_commit_then_refork_rebuilds_generation_visibility() {
    llama_kv_cells cells;
    cells.resize(2048);

    for (uint32_t i = 0; i < 128; ++i) {
        add_cell(cells, i, i, { 4, 9 });
    }
    for (uint32_t i = 0; i < 32; ++i) {
        add_cell(cells, 512 + i, 128 + i, { 4 });
        add_cell(cells, 1024 + i, 128 + i, { 9 });
    }
    assert(llama_kv_build_ragged_plan(cells, { 4, 9 }, 256).reduces_columns);

    // Commit sequence 4: dropping sequence 9 removes its private rows and
    // turns every shared row back into a singleton winner row.
    for (uint32_t i = 0; i < 128; ++i) {
        assert(!cells.seq_rm(i, 9));
    }
    for (uint32_t i = 0; i < 32; ++i) {
        assert(cells.seq_rm(1024 + i, 9));
    }
    const auto committed = llama_kv_build_ragged_plan(cells, { 4, 9 }, 256);
    assert(!committed.has_shared_prefix);
    assert(!committed.reduces_columns);

    // Re-fork the committed winner into a fresh physical slot/sequence. The
    // entire winner state becomes shared again; new tails remain isolated.
    for (uint32_t i = 0; i < 128; ++i) {
        cells.seq_add(i, 9);
    }
    for (uint32_t i = 0; i < 32; ++i) {
        cells.seq_add(512 + i, 9);
        add_cell(cells, 1536 + i, 160 + i, { 9 });
    }
    const auto reforked = llama_kv_build_ragged_plan(cells, { 4, 9 }, 256);
    assert(reforked.has_shared_prefix);
    assert(reforked.reduces_columns);
    assert(reforked.rows[160] == std::numeric_limits<uint32_t>::max());
    assert(reforked.rows[reforked.n_kv + 160] == 1536);
}

static void test_small_reduction_stays_dense() {
    // Activation-ratio heuristic (default LLAMA_KV_TREE_RAGGED_MIN_REDUCTION
    // = 10): a fork family whose ragged plan saves under 10% of the dense
    // columns must stay on the dense graph — the indexed gather costs more
    // than it saves there (B2 branch-cost evidence).
    llama_kv_cells cells;
    cells.resize(8192);

    for (uint32_t i = 0; i < 6300; ++i) {
        add_cell(cells, i, i, { 3, 7 });
    }
    for (uint32_t i = 0; i < 100; ++i) {
        add_cell(cells, 6300 + i, 6300 + i, { 3 });
        add_cell(cells, 6800 + i, 6300 + i, { 7 });
    }

    const auto plan = llama_kv_build_ragged_plan(cells, { 3, 7 }, 256);

    assert(plan.has_shared_prefix);
    assert(plan.n_kv == 6400);        // GGML_PAD(6300 + 100, 256)
    assert(plan.dense_n_kv == 6912);  // GGML_PAD(6900, 256)
    // 6400/6912 = 92.6% of dense: a 7.4% saving, under the 10% threshold.
    assert(!plan.reduces_columns);
}

int main() {
    test_shared_prefix_private_tails_are_ragged();
    test_unrelated_sequences_do_not_activate_tree_path();
    test_shared_dense_family_stays_on_existing_graph();
    test_commit_then_refork_rebuilds_generation_visibility();
    test_small_reduction_stays_dense();
    std::cout << "Treebeard ragged KV plan tests passed\n";
    return 0;
}
