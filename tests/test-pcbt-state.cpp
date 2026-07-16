// PCBT-1 exit gate: isolated state-machine tests — every legal transition
// covered, every illegal transition rejected, no inference launched.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "server-pcbt.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <set>
#include <utility>

static const pcbt_status ALL[] = {
    pcbt_status::CREATING, pcbt_status::RUNNING, pcbt_status::AWAITING_DECISION,
    pcbt_status::COMMITTING, pcbt_status::COMMITTED, pcbt_status::ABORTING,
    pcbt_status::ABORTED, pcbt_status::EXPIRING, pcbt_status::EXPIRED,
    pcbt_status::FAILED,
};

static void test_transition_matrix() {
    using S = pcbt_status;
    // The complete legal relation per contract §4, written out explicitly so
    // the test is an independent statement of the spec, not a mirror of the
    // implementation.
    std::set<std::pair<S,S>> legal = {
        {S::CREATING, S::RUNNING},
        {S::RUNNING, S::AWAITING_DECISION},
        {S::AWAITING_DECISION, S::COMMITTING},
        {S::COMMITTING, S::COMMITTED},
        {S::CREATING, S::ABORTING}, {S::RUNNING, S::ABORTING}, {S::AWAITING_DECISION, S::ABORTING},
        {S::ABORTING, S::ABORTED},
        {S::CREATING, S::EXPIRING}, {S::RUNNING, S::EXPIRING}, {S::AWAITING_DECISION, S::EXPIRING},
        {S::EXPIRING, S::EXPIRED},
        // FAILED from every pre-commit (non-terminal) state
        {S::CREATING, S::FAILED}, {S::RUNNING, S::FAILED}, {S::AWAITING_DECISION, S::FAILED},
        {S::COMMITTING, S::FAILED}, {S::ABORTING, S::FAILED}, {S::EXPIRING, S::FAILED},
    };
    int checked = 0;
    for (auto from : ALL) {
        for (auto to : ALL) {
            const bool want = legal.count({from, to}) > 0;
            assert(pcbt_transition_legal(from, to) == want);
            // transition() must agree and only mutate when legal
            server_branch_transaction tx;
            tx.status = from;
            const bool did = tx.transition(to);
            assert(did == want);
            assert(tx.status == (want ? to : from));
            ++checked;
        }
    }
    assert(checked == 100);
    // terminal states admit nothing
    for (auto s : {S::COMMITTED, S::ABORTED, S::EXPIRED, S::FAILED}) {
        for (auto to : ALL) {
            assert(!pcbt_transition_legal(s, to));
        }
    }
}

static void test_event_ring_bounds() {
    server_branch_transaction tx;
    tx.max_events = 8;
    for (int i = 0; i < 20; ++i) {
        tx.push_event(1000 + i, "k", "{}");
    }
    assert(tx.events.size() == 8);
    assert(tx.next_seq == 21);
    assert(tx.first_seq == 13);            // truncation is explicit
    assert(tx.events.front().seq == 13);
    assert(tx.events.back().seq == 20);
    uint64_t prev = 0;
    for (const auto & e : tx.events) {     // monotonic
        assert(e.seq > prev);
        prev = e.seq;
    }
}

static void test_candidate_byte_accounting() {
    server_branch_transaction tx;
    tx.budget.max_candidate_bytes = 10;
    tx.branches.resize(2);
    tx.branches[0].key = "a";
    tx.branches[1].key = "b";

    const uint8_t body[8] = {1,2,3,4,5,6,7,8};
    assert(tx.store_candidate_bytes(tx.branches[0], body, 8));
    assert(!tx.branches[0].body_truncated);
    assert(tx.candidate_bytes_stored == 8);

    // second branch only has 2 bytes of aggregate room -> truncated
    assert(!tx.store_candidate_bytes(tx.branches[1], body, 8));
    assert(tx.branches[1].body_truncated);
    assert(tx.branches[1].output_bytes.size() == 2);
    assert(tx.candidate_bytes_stored == 10);
}

static void test_decision_readiness() {
    server_branch_transaction tx;
    tx.branches.resize(3);
    for (auto & b : tx.branches) b.phase = pcbt_branch_phase::QUEUED;
    assert(!tx.ready_for_decision());

    tx.branches[0].phase = pcbt_branch_phase::COMPLETED;
    tx.branches[1].phase = pcbt_branch_phase::FAILED;
    assert(!tx.ready_for_decision());       // one branch still queued

    tx.branches[2].phase = pcbt_branch_phase::CANCELED;
    assert(tx.ready_for_decision());        // terminal + >=1 candidate
    assert(!tx.all_branches_failed_or_canceled());

    tx.branches[0].phase = pcbt_branch_phase::FAILED;
    assert(!tx.ready_for_decision());       // no candidate -> FAILED path
    assert(tx.all_branches_failed_or_canceled());
}

static void test_cleanup_idempotent() {
    server_branch_transaction tx;
    tx.budget.max_candidate_bytes = 64;
    tx.branches.resize(2);
    tx.branches[0].key = "a"; tx.branches[0].slot_id = 3;
    tx.branches[0].phase = pcbt_branch_phase::RUNNING;
    tx.branches[1].key = "b"; tx.branches[1].slot_id = 5;
    tx.branches[1].phase = pcbt_branch_phase::COMPLETED;
    const uint8_t body[4] = {9,9,9,9};
    tx.store_candidate_bytes(tx.branches[1], body, 4);

    tx.cleanup_members();
    assert(tx.branches[0].phase == pcbt_branch_phase::CANCELED);  // fenced
    assert(tx.branches[1].phase == pcbt_branch_phase::COMPLETED); // terminal preserved
    assert(tx.branches[0].slot_id == -1 && tx.branches[1].slot_id == -1);
    assert(tx.candidate_bytes_stored == 0);

    tx.cleanup_members();                                          // idempotent
    assert(tx.branches[0].phase == pcbt_branch_phase::CANCELED);
    assert(tx.candidate_bytes_stored == 0);
}

static void test_registry_idempotency() {
    pcbt_registry reg;
    reg.max_transactions = 2;
    bool conflict = false;

    auto * t1 = reg.create("run-1", "sha256:aa", conflict);
    assert(t1 && !conflict && t1->id == 1);

    auto * t1b = reg.create("run-1", "sha256:aa", conflict);   // exact retry
    assert(t1b == t1 && !conflict);

    auto * bad = reg.create("run-1", "sha256:bb", conflict);   // changed body
    assert(!bad && conflict);

    auto * t2 = reg.create("run-2", "sha256:cc", conflict);
    assert(t2 && !conflict && t2->id == 2);

    auto * t3 = reg.create("run-3", "sha256:dd", conflict);    // capacity
    assert(!t3 && !conflict);

    assert(reg.find(1) == t1 && reg.find(2) == t2 && reg.find(99) == nullptr);
}

int main() {
    test_transition_matrix();
    test_event_ring_bounds();
    test_candidate_byte_accounting();
    test_decision_readiness();
    test_cleanup_idempotent();
    test_registry_idempotency();
    std::cout << "PCBT state-machine tests passed\n";
    return 0;
}
