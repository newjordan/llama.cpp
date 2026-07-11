#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

struct server_snapshot_manifest_error : std::runtime_error {
    bool unavailable;

    server_snapshot_manifest_error(std::string message, bool unavailable = false)
        : std::runtime_error(std::move(message)), unavailable(unavailable) {}
};

struct server_snapshot_manifest_ref {
    std::string owner;
    std::string digest;
    std::string retention_class;
    uint64_t revision = 0;
};

struct server_snapshot_publish_intent {
    std::string owner;
    std::string digest;
    std::string retention_class;
    uint64_t revision = 0;
};

struct server_snapshot_publish_begin_result {
    server_snapshot_publish_intent intent;
    server_snapshot_manifest_ref existing_ref;
    bool already_retained = false;
    bool deduplicated = false;
    uint64_t revision = 0;
    uint64_t file_bytes = 0;
};

struct server_snapshot_publish_advance_intent {
    std::string owner;
    std::string digest;
    std::string retention_class;
    std::string head_name;
    std::string expected_digest;
    uint64_t expected_generation = 0;
    uint64_t revision = 0;
};

struct server_snapshot_manifest_result {
    server_snapshot_manifest_ref ref;
    bool deduplicated = false;
    uint64_t revision = 0;
    uint64_t file_bytes = 0;
};

struct server_snapshot_manifest_compact_result {
    uint64_t revision = 0;
    uint64_t refs = 0;
    uint64_t records_before = 0;
    uint64_t records_after = 0;
    uint64_t bytes_before = 0;
    uint64_t bytes_after = 0;
};

struct server_snapshot_cache_eviction_candidate {
    std::string digest;
    uint64_t newest_ref_revision = 0;
    uint64_t refs = 0;
};

struct server_snapshot_cache_eviction_result {
    std::string digest;
    uint64_t released_refs = 0;
    uint64_t revision = 0;
    uint64_t file_bytes = 0;
};

struct server_snapshot_logical_head {
    std::string name;
    std::string digest;
    std::string parent_digest;
    uint64_t generation = 0;
    uint64_t revision = 0;
};

struct server_snapshot_logical_head_result {
    server_snapshot_logical_head head;
    bool deduplicated = false;
    uint64_t revision = 0;
    uint64_t file_bytes = 0;
};

struct server_snapshot_logical_head_tombstone {
    server_snapshot_logical_head head;
    uint64_t deletion_revision = 0;
};

struct server_snapshot_publish_advance_begin_result {
    server_snapshot_publish_advance_intent intent;
    server_snapshot_manifest_ref ref;
    server_snapshot_logical_head head;
    bool already_committed = false;
    bool deduplicated = false;
    uint64_t revision = 0;
    uint64_t file_bytes = 0;
};

struct server_snapshot_publish_advance_result {
    server_snapshot_manifest_ref ref;
    server_snapshot_logical_head head;
    bool deduplicated = false;
    uint64_t revision = 0;
    uint64_t file_bytes = 0;
};

class server_snapshot_manifest {
public:
    server_snapshot_manifest(
            std::string namespace_path,
            std::string compatibility_id,
            uint64_t max_bytes);

    const std::map<std::string, server_snapshot_manifest_ref> & refs() const;
    const std::map<std::string, server_snapshot_publish_intent> & publish_intents() const;
    const std::map<std::string, server_snapshot_publish_advance_intent> &
        publish_advance_intents() const;
    const std::set<std::string> & managed_digests() const;
    const std::map<std::string, server_snapshot_logical_head> & logical_heads() const;
    uint64_t ref_count(const std::string & digest) const;
    uint64_t publish_intent_count(const std::string & digest) const;
    uint64_t revision() const;
    uint64_t file_bytes() const;
    uint64_t byte_budget() const;
    uint64_t high_water_bytes() const;
    uint64_t record_count() const;
    uint64_t recovered_temp_files() const;
    uint64_t recovered_tail_bytes() const;
    uint64_t compactions() const;

    server_snapshot_manifest_result retain(
            const std::string & owner,
            const std::string & digest,
            const std::string & retention_class);
    server_snapshot_manifest_result release(
            const std::string & owner,
            const std::string & digest);
    server_snapshot_publish_begin_result begin_publish(
            const std::string & owner,
            const std::string & digest,
            const std::string & retention_class);
    server_snapshot_manifest_result commit_publish(
            const std::string & owner,
            const std::string & digest);
    server_snapshot_manifest_result abort_publish(
            const std::string & owner,
            const std::string & digest);
    server_snapshot_publish_advance_begin_result begin_publish_advance(
            const std::string & owner,
            const std::string & digest,
            const std::string & retention_class,
            const std::string & head_name,
            uint64_t expected_generation,
            const std::string & expected_digest);
    server_snapshot_publish_advance_result commit_publish_advance(
            const std::string & owner,
            const std::string & digest);
    server_snapshot_publish_advance_result abort_publish_advance(
            const std::string & owner,
            const std::string & digest);
    std::vector<server_snapshot_cache_eviction_candidate> cache_eviction_candidates() const;
    server_snapshot_cache_eviction_result evict_cache(const std::string & digest);
    void forget_managed(const std::string & digest);
    server_snapshot_logical_head_result create_head(
            const std::string & name,
            const std::string & digest);
    server_snapshot_logical_head_result advance_head(
            const std::string & name,
            uint64_t expected_generation,
            const std::string & expected_digest,
            const std::string & digest);
    server_snapshot_logical_head_result delete_head(
            const std::string & name,
            uint64_t expected_generation,
            const std::string & expected_digest);
    server_snapshot_manifest_compact_result compact();

private:
    enum class record_type : uint8_t {
        retain = 1,
        release = 2,
        checkpoint = 3,
        begin_publish = 4,
        commit_publish = 5,
        abort_publish = 6,
        evict_cache = 7,
        forget_managed = 8,
        create_head = 9,
        advance_head = 10,
        delete_head = 11,
        begin_publish_advance = 12,
        commit_publish_advance = 13,
        abort_publish_advance = 14,
    };

    std::string namespace_path_;
    std::string path_;
    std::string compatibility_id_;
    uint64_t max_bytes_ = 0;
    uint64_t revision_ = 0;
    uint64_t file_bytes_ = 0;
    uint64_t high_water_bytes_ = 0;
    uint64_t record_count_ = 0;
    uint64_t recovered_temp_files_ = 0;
    uint64_t recovered_tail_bytes_ = 0;
    uint64_t compactions_ = 0;
    uint64_t temp_sequence_ = 0;
    std::map<std::string, server_snapshot_manifest_ref> refs_;
    std::map<std::string, server_snapshot_publish_intent> publish_intents_;
    std::map<std::string, server_snapshot_publish_advance_intent> publish_advance_intents_;
    std::set<std::string> managed_digests_;
    std::map<std::string, server_snapshot_logical_head> logical_heads_;
    std::map<std::string, server_snapshot_logical_head_tombstone> logical_head_tombstones_;

    static std::string ref_key(const std::string & owner, const std::string & digest);
    static void validate_owner(const std::string & owner);
    static void validate_digest(const std::string & digest);
    static void validate_retention_class(const std::string & retention_class);
    static void validate_head_name(const std::string & name);

    std::vector<uint8_t> encode_header() const;
    std::vector<uint8_t> encode_mutation(
            record_type type,
            uint64_t revision,
            const std::string & owner,
            const std::string & digest,
            const std::string & retention_class) const;
    std::vector<uint8_t> encode_checkpoint(uint64_t revision) const;
    std::vector<uint8_t> encode_head_mutation(
            record_type type,
            uint64_t revision,
            const std::string & name,
            const std::string & digest,
            const std::string & parent_digest,
            uint64_t generation) const;
    std::vector<uint8_t> encode_publish_advance_mutation(
            record_type type,
            uint64_t revision,
            const server_snapshot_publish_advance_intent & intent) const;
    void validate_publish_terminal_space(const server_snapshot_publish_intent & intent);
    void validate_publish_advance_terminal_space(
            const server_snapshot_publish_advance_intent & intent);
    void apply_record(const std::vector<uint8_t> & record);
    void discover();
    void append_record(const std::vector<uint8_t> & record);
    void write_compacted(const std::vector<uint8_t> & checkpoint);
    void sync_namespace_directory() const;
};
