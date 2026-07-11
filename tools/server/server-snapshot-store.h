#pragma once

#include "llama.h"

#include <cstdint>
#include <functional>
#include <map>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

struct server_snapshot_store_error : std::runtime_error {
    bool unavailable;

    server_snapshot_store_error(std::string message, bool unavailable = false)
        : std::runtime_error(std::move(message)), unavailable(unavailable) {}
};

struct server_snapshot_store_entry {
    std::string digest;
    std::string path;
    uint64_t n_tokens = 0;
    uint64_t state_bytes = 0;
    uint64_t payload_bytes = 0;
    uint64_t file_bytes = 0;
};

struct server_snapshot_store_payload {
    server_snapshot_store_entry entry;
    std::vector<llama_token> tokens;
    std::vector<uint8_t> state;
};

struct server_snapshot_store_spill_result {
    server_snapshot_store_entry entry;
    bool deduplicated = false;
};

class server_snapshot_store {
public:
    using digest_fn = std::function<std::string(const std::vector<llama_token> &, const std::vector<uint8_t> &)>;

    server_snapshot_store(
            std::string base_path,
            std::string compatibility_id,
            uint64_t max_disk_bytes,
            digest_fn compute_digest);

    std::map<std::string, server_snapshot_store_entry> entries() const;
    bool find_entry(const std::string & digest, server_snapshot_store_entry & entry) const;
    uint64_t projected_file_bytes(uint64_t n_tokens, uint64_t state_bytes) const;
    const std::string & compatibility_id() const;
    const std::string & namespace_id() const;
    const std::string & namespace_path() const;

    uint64_t disk_bytes() const;
    uint64_t disk_budget_bytes() const;
    uint64_t disk_high_water_bytes() const;
    uint64_t recovered_temp_files() const;
    uint64_t ignored_corrupt_files() const;
    uint64_t runtime_integrity_failures() const;
    uint64_t orphaned_disk_bytes() const;

    server_snapshot_store_spill_result spill(
            const std::string & digest,
            const std::vector<llama_token> & tokens,
            const std::vector<uint8_t> & state);
    server_snapshot_store_payload load(
            const std::string & digest,
            uint64_t max_payload_bytes = std::numeric_limits<uint64_t>::max()) const;
    server_snapshot_store_entry erase(const std::string & digest);

private:
    std::string base_path_;
    std::string compatibility_id_;
    std::string namespace_id_;
    std::string namespace_path_;
    uint64_t max_disk_bytes_ = 0;
    uint64_t disk_bytes_ = 0;
    uint64_t disk_high_water_bytes_ = 0;
    uint64_t recovered_temp_files_ = 0;
    uint64_t ignored_corrupt_files_ = 0;
    mutable uint64_t runtime_integrity_failures_ = 0;
    uint64_t orphaned_disk_bytes_ = 0;
    uint64_t temp_sequence_ = 0;
    digest_fn compute_digest_;
    std::map<std::string, server_snapshot_store_entry> entries_;
    mutable std::mutex mutex_;

    static bool valid_digest(const std::string & digest);
    static std::string digest_hex(const std::string & digest);
    static std::string compatibility_namespace(const std::string & compatibility_id);
    static uint64_t object_file_bytes(size_t compatibility_size, size_t digest_size,
            uint64_t n_tokens, uint64_t state_bytes);

    server_snapshot_store_entry inspect_file(const std::string & path, const std::string & expected_digest) const;
    server_snapshot_store_payload load_unlocked(const std::string & digest, uint64_t max_payload_bytes) const;
    void discover();
    void sync_namespace_directory() const;
};
