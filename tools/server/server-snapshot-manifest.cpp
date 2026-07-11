#include "server-snapshot-manifest.h"

extern "C" {
#include "sha256.h"
}

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <tuple>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

constexpr std::array<uint8_t, 8> MANIFEST_MAGIC = {'T', 'S', 'T', 'M', 'A', 'N', '0', '1'};
constexpr uint32_t MANIFEST_VERSION = 1;
constexpr uint32_t RECORD_MAGIC = 0x31464e4d; // "MNF1" in little endian
constexpr size_t HEADER_FIXED_BYTES = 16;
constexpr size_t RECORD_FIXED_BYTES = 40;
constexpr size_t RECORD_CHECKSUM_BYTES = SHA256_DIGEST_SIZE;
constexpr size_t RECORD_MIN_BYTES = RECORD_FIXED_BYTES + RECORD_CHECKSUM_BYTES;
constexpr size_t CHECKPOINT_REF_FIXED_BYTES = 20;
constexpr size_t CHECKPOINT_PUBLISH_ADVANCE_FIXED_BYTES = 36;
constexpr size_t MAX_COMPATIBILITY_BYTES = 256;
constexpr size_t MAX_OWNER_BYTES = 128;
constexpr uint32_t CHECKPOINT_SCHEMA_ATOMIC_PUBLISH_ADVANCE = 3;

void append_u32(std::vector<uint8_t> & out, uint32_t value) {
    for (size_t i = 0; i < sizeof(value); ++i) {
        out.push_back((uint8_t) (value >> (8 * i)));
    }
}

void append_u64(std::vector<uint8_t> & out, uint64_t value) {
    for (size_t i = 0; i < sizeof(value); ++i) {
        out.push_back((uint8_t) (value >> (8 * i)));
    }
}

uint32_t read_u32(const std::vector<uint8_t> & data, size_t & offset) {
    if (offset > data.size() || data.size() - offset < sizeof(uint32_t)) {
        throw server_snapshot_manifest_error("truncated snapshot manifest integer");
    }
    uint32_t value = 0;
    for (size_t i = 0; i < sizeof(value); ++i) {
        value |= (uint32_t) data[offset++] << (8 * i);
    }
    return value;
}

uint64_t read_u64(const std::vector<uint8_t> & data, size_t & offset) {
    if (offset > data.size() || data.size() - offset < sizeof(uint64_t)) {
        throw server_snapshot_manifest_error("truncated snapshot manifest integer");
    }
    uint64_t value = 0;
    for (size_t i = 0; i < sizeof(value); ++i) {
        value |= (uint64_t) data[offset++] << (8 * i);
    }
    return value;
}

std::string read_string(const std::vector<uint8_t> & data, size_t & offset, size_t size) {
    if (offset > data.size() || size > data.size() - offset) {
        throw server_snapshot_manifest_error("truncated snapshot manifest string");
    }
    std::string value((const char *) data.data() + offset, size);
    offset += size;
    return value;
}

std::array<uint8_t, SHA256_DIGEST_SIZE> record_checksum(const uint8_t * data, size_t size) {
    static constexpr uint8_t domain[] = "turbo-statetree-manifest-record-v1\0";
    sha256_t hash;
    sha256_init(&hash);
    sha256_update(&hash, domain, sizeof(domain));
    sha256_update(&hash, data, size);
    std::array<uint8_t, SHA256_DIGEST_SIZE> digest {};
    sha256_final(&hash, digest.data());
    return digest;
}

#if !defined(_WIN32)
void write_all(int fd, const uint8_t * data, size_t size) {
    size_t offset = 0;
    while (offset < size) {
        const ssize_t written = ::write(fd, data + offset, size - offset);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw server_snapshot_manifest_error(
                    std::string("failed to write snapshot manifest: ") + std::strerror(errno));
        }
        if (written == 0) {
            throw server_snapshot_manifest_error("failed to make progress writing snapshot manifest");
        }
        offset += (size_t) written;
    }
}

void sync_file(const std::string & path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        throw server_snapshot_manifest_error("failed to open snapshot manifest for sync");
    }
    const int result = ::fsync(fd);
    const int saved_errno = errno;
    ::close(fd);
    if (result != 0) {
        throw server_snapshot_manifest_error(
                std::string("failed to sync snapshot manifest: ") + std::strerror(saved_errno));
    }
}
#endif

} // namespace

server_snapshot_manifest::server_snapshot_manifest(
        std::string namespace_path,
        std::string compatibility_id,
        uint64_t max_bytes)
    : namespace_path_(std::move(namespace_path)),
      path_((std::filesystem::path(namespace_path_) / "manifest.v1").string()),
      compatibility_id_(std::move(compatibility_id)),
      max_bytes_(max_bytes) {
    if (compatibility_id_.empty() || compatibility_id_.size() > MAX_COMPATIBILITY_BYTES) {
        throw server_snapshot_manifest_error("invalid snapshot manifest compatibility ID");
    }
    if (max_bytes_ < HEADER_FIXED_BYTES + compatibility_id_.size()) {
        throw server_snapshot_manifest_error("snapshot manifest byte budget is too small", true);
    }
    std::error_code error;
    if (!std::filesystem::is_directory(namespace_path_, error) || error) {
        throw server_snapshot_manifest_error("snapshot manifest namespace is unavailable");
    }
    discover();
}

const std::map<std::string, server_snapshot_manifest_ref> & server_snapshot_manifest::refs() const {
    return refs_;
}

const std::map<std::string, server_snapshot_publish_intent> &
server_snapshot_manifest::publish_intents() const {
    return publish_intents_;
}

const std::map<std::string, server_snapshot_publish_advance_intent> &
server_snapshot_manifest::publish_advance_intents() const {
    return publish_advance_intents_;
}

const std::set<std::string> & server_snapshot_manifest::managed_digests() const {
    return managed_digests_;
}

const std::map<std::string, server_snapshot_logical_head> &
server_snapshot_manifest::logical_heads() const {
    return logical_heads_;
}

uint64_t server_snapshot_manifest::ref_count(const std::string & digest) const {
    const uint64_t owner_refs = (uint64_t) std::count_if(refs_.begin(), refs_.end(), [&](const auto & item) {
        return item.second.digest == digest;
    });
    const uint64_t head_refs = (uint64_t) std::count_if(
            logical_heads_.begin(), logical_heads_.end(), [&](const auto & item) {
                return item.second.digest == digest;
            });
    return owner_refs + head_refs;
}

uint64_t server_snapshot_manifest::publish_intent_count(const std::string & digest) const {
    const uint64_t normal = (uint64_t) std::count_if(
            publish_intents_.begin(), publish_intents_.end(), [&](const auto & item) {
                return item.second.digest == digest;
            });
    const uint64_t combined = (uint64_t) std::count_if(
            publish_advance_intents_.begin(), publish_advance_intents_.end(), [&](const auto & item) {
                return item.second.digest == digest;
            });
    return normal + combined;
}

uint64_t server_snapshot_manifest::revision() const { return revision_; }
uint64_t server_snapshot_manifest::file_bytes() const { return file_bytes_; }
uint64_t server_snapshot_manifest::byte_budget() const { return max_bytes_; }
uint64_t server_snapshot_manifest::high_water_bytes() const { return high_water_bytes_; }
uint64_t server_snapshot_manifest::record_count() const { return record_count_; }
uint64_t server_snapshot_manifest::recovered_temp_files() const { return recovered_temp_files_; }
uint64_t server_snapshot_manifest::recovered_tail_bytes() const { return recovered_tail_bytes_; }
uint64_t server_snapshot_manifest::compactions() const { return compactions_; }

std::string server_snapshot_manifest::ref_key(const std::string & owner, const std::string & digest) {
    return owner + '\0' + digest;
}

void server_snapshot_manifest::validate_owner(const std::string & owner) {
    if (owner.empty() || owner.size() > MAX_OWNER_BYTES ||
            !std::all_of(owner.begin(), owner.end(), [](unsigned char value) {
                return (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') ||
                    (value >= '0' && value <= '9') || value == '.' || value == '_' ||
                    value == ':' || value == '/' || value == '-';
            })) {
        throw server_snapshot_manifest_error(
                "snapshot manifest owner must contain 1 to 128 portable identifier characters");
    }
}

void server_snapshot_manifest::validate_digest(const std::string & digest) {
    if (digest.size() != 71 || digest.compare(0, 7, "sha256:") != 0 ||
            !std::all_of(digest.begin() + 7, digest.end(), [](unsigned char value) {
                return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
            })) {
        throw server_snapshot_manifest_error("invalid snapshot manifest content digest");
    }
}

void server_snapshot_manifest::validate_retention_class(const std::string & retention_class) {
    if (retention_class != "pinned" && retention_class != "cache") {
        throw server_snapshot_manifest_error("snapshot retention_class must be pinned or cache");
    }
}

void server_snapshot_manifest::validate_head_name(const std::string & name) {
    validate_owner(name);
}

std::vector<uint8_t> server_snapshot_manifest::encode_header() const {
    std::vector<uint8_t> header;
    header.reserve(HEADER_FIXED_BYTES + compatibility_id_.size());
    header.insert(header.end(), MANIFEST_MAGIC.begin(), MANIFEST_MAGIC.end());
    append_u32(header, MANIFEST_VERSION);
    append_u32(header, (uint32_t) compatibility_id_.size());
    header.insert(header.end(), compatibility_id_.begin(), compatibility_id_.end());
    return header;
}

std::vector<uint8_t> server_snapshot_manifest::encode_mutation(
        record_type type,
        uint64_t revision,
        const std::string & owner,
        const std::string & digest,
        const std::string & retention_class) const {
    validate_digest(digest);
    if (type == record_type::evict_cache || type == record_type::forget_managed) {
        if (!owner.empty() || !retention_class.empty()) {
            throw server_snapshot_manifest_error("invalid managed snapshot mutation record");
        }
    } else {
        validate_owner(owner);
    }
    if (type == record_type::retain || type == record_type::begin_publish) {
        validate_retention_class(retention_class);
    } else if ((type != record_type::release && type != record_type::commit_publish &&
            type != record_type::abort_publish && type != record_type::evict_cache &&
            type != record_type::forget_managed) || !retention_class.empty()) {
        throw server_snapshot_manifest_error("invalid snapshot manifest mutation record");
    }
    const uint64_t total = RECORD_FIXED_BYTES + owner.size() + digest.size() +
        retention_class.size() + RECORD_CHECKSUM_BYTES;
    if (total > std::numeric_limits<uint32_t>::max() || total > max_bytes_) {
        throw server_snapshot_manifest_error("snapshot manifest mutation exceeds the byte budget", true);
    }
    std::vector<uint8_t> record;
    record.reserve((size_t) total);
    append_u32(record, (uint32_t) total);
    append_u32(record, RECORD_MAGIC);
    append_u64(record, revision);
    record.push_back((uint8_t) type);
    record.insert(record.end(), 7, 0);
    append_u32(record, (uint32_t) owner.size());
    append_u32(record, (uint32_t) digest.size());
    append_u32(record, (uint32_t) retention_class.size());
    append_u32(record, 0);
    record.insert(record.end(), owner.begin(), owner.end());
    record.insert(record.end(), digest.begin(), digest.end());
    record.insert(record.end(), retention_class.begin(), retention_class.end());
    const auto checksum = record_checksum(record.data(), record.size());
    record.insert(record.end(), checksum.begin(), checksum.end());
    return record;
}

std::vector<uint8_t> server_snapshot_manifest::encode_head_mutation(
        record_type type,
        uint64_t revision,
        const std::string & name,
        const std::string & digest,
        const std::string & parent_digest,
        uint64_t generation) const {
    if (type != record_type::create_head && type != record_type::advance_head &&
            type != record_type::delete_head) {
        throw server_snapshot_manifest_error("invalid logical head mutation type");
    }
    validate_head_name(name);
    validate_digest(digest);
    if (!parent_digest.empty()) {
        validate_digest(parent_digest);
    }
    if (generation == 0) {
        throw server_snapshot_manifest_error("logical head generation must be positive");
    }
    const uint64_t total = RECORD_FIXED_BYTES + sizeof(uint64_t) + name.size() +
        digest.size() + parent_digest.size() + RECORD_CHECKSUM_BYTES;
    if (total > std::numeric_limits<uint32_t>::max() || total > max_bytes_) {
        throw server_snapshot_manifest_error("logical head mutation exceeds the manifest byte budget", true);
    }
    std::vector<uint8_t> record;
    record.reserve((size_t) total);
    append_u32(record, (uint32_t) total);
    append_u32(record, RECORD_MAGIC);
    append_u64(record, revision);
    record.push_back((uint8_t) type);
    record.insert(record.end(), 7, 0);
    append_u32(record, (uint32_t) name.size());
    append_u32(record, (uint32_t) digest.size());
    append_u32(record, (uint32_t) parent_digest.size());
    append_u32(record, 0);
    append_u64(record, generation);
    record.insert(record.end(), name.begin(), name.end());
    record.insert(record.end(), digest.begin(), digest.end());
    record.insert(record.end(), parent_digest.begin(), parent_digest.end());
    const auto checksum = record_checksum(record.data(), record.size());
    record.insert(record.end(), checksum.begin(), checksum.end());
    return record;
}

std::vector<uint8_t> server_snapshot_manifest::encode_publish_advance_mutation(
        record_type type,
        uint64_t revision,
        const server_snapshot_publish_advance_intent & intent) const {
    if (type != record_type::begin_publish_advance &&
            type != record_type::commit_publish_advance &&
            type != record_type::abort_publish_advance) {
        throw server_snapshot_manifest_error("invalid publish-and-advance mutation type");
    }
    validate_owner(intent.owner);
    validate_digest(intent.digest);
    validate_retention_class(intent.retention_class);
    validate_head_name(intent.head_name);
    validate_digest(intent.expected_digest);
    if (intent.expected_generation == 0) {
        throw server_snapshot_manifest_error(
                "publish-and-advance expected generation must be positive");
    }
    const uint64_t total = RECORD_FIXED_BYTES + sizeof(uint32_t) + sizeof(uint64_t) +
        intent.owner.size() + intent.digest.size() + intent.retention_class.size() +
        intent.head_name.size() + intent.expected_digest.size() + RECORD_CHECKSUM_BYTES;
    if (total > std::numeric_limits<uint32_t>::max() || total > max_bytes_) {
        throw server_snapshot_manifest_error(
                "publish-and-advance mutation exceeds the manifest byte budget", true);
    }
    std::vector<uint8_t> record;
    record.reserve((size_t) total);
    append_u32(record, (uint32_t) total);
    append_u32(record, RECORD_MAGIC);
    append_u64(record, revision);
    record.push_back((uint8_t) type);
    record.insert(record.end(), 7, 0);
    append_u32(record, (uint32_t) intent.owner.size());
    append_u32(record, (uint32_t) intent.digest.size());
    append_u32(record, (uint32_t) intent.retention_class.size());
    append_u32(record, (uint32_t) intent.head_name.size());
    append_u32(record, (uint32_t) intent.expected_digest.size());
    append_u64(record, intent.expected_generation);
    record.insert(record.end(), intent.owner.begin(), intent.owner.end());
    record.insert(record.end(), intent.digest.begin(), intent.digest.end());
    record.insert(record.end(), intent.retention_class.begin(), intent.retention_class.end());
    record.insert(record.end(), intent.head_name.begin(), intent.head_name.end());
    record.insert(record.end(), intent.expected_digest.begin(), intent.expected_digest.end());
    const auto checksum = record_checksum(record.data(), record.size());
    record.insert(record.end(), checksum.begin(), checksum.end());
    return record;
}

std::vector<uint8_t> server_snapshot_manifest::encode_checkpoint(uint64_t revision) const {
    uint64_t total = RECORD_FIXED_BYTES + RECORD_CHECKSUM_BYTES;
    for (const auto & item : refs_) {
        const auto & ref = item.second;
        const uint64_t entry = CHECKPOINT_REF_FIXED_BYTES + ref.owner.size() +
            ref.digest.size() + ref.retention_class.size();
        if (entry > std::numeric_limits<uint64_t>::max() - total) {
            throw server_snapshot_manifest_error("snapshot manifest checkpoint size overflow");
        }
        total += entry;
    }
    for (const auto & item : publish_intents_) {
        const auto & intent = item.second;
        const uint64_t entry = CHECKPOINT_REF_FIXED_BYTES + intent.owner.size() +
            intent.digest.size() + intent.retention_class.size();
        if (entry > std::numeric_limits<uint64_t>::max() - total) {
            throw server_snapshot_manifest_error("snapshot manifest checkpoint size overflow");
        }
        total += entry;
    }
    for (const auto & digest : managed_digests_) {
        const uint64_t entry = sizeof(uint32_t) + digest.size();
        if (entry > std::numeric_limits<uint64_t>::max() - total) {
            throw server_snapshot_manifest_error("snapshot manifest checkpoint size overflow");
        }
        total += entry;
    }
    if (sizeof(uint32_t) > std::numeric_limits<uint64_t>::max() - total) {
        throw server_snapshot_manifest_error("snapshot manifest checkpoint size overflow");
    }
    total += sizeof(uint32_t);
    for (const auto & item : logical_heads_) {
        const auto & head = item.second;
        const uint64_t entry = 3 * sizeof(uint32_t) + 2 * sizeof(uint64_t) +
            head.name.size() + head.digest.size() + head.parent_digest.size();
        if (entry > std::numeric_limits<uint64_t>::max() - total) {
            throw server_snapshot_manifest_error("snapshot manifest checkpoint size overflow");
        }
        total += entry;
    }
    if (sizeof(uint32_t) > std::numeric_limits<uint64_t>::max() - total) {
        throw server_snapshot_manifest_error("snapshot manifest checkpoint size overflow");
    }
    total += sizeof(uint32_t);
    for (const auto & item : logical_head_tombstones_) {
        const auto & tombstone = item.second;
        const auto & head = tombstone.head;
        const uint64_t entry = 3 * sizeof(uint32_t) + 3 * sizeof(uint64_t) +
            head.name.size() + head.digest.size() + head.parent_digest.size();
        if (entry > std::numeric_limits<uint64_t>::max() - total) {
            throw server_snapshot_manifest_error("snapshot manifest checkpoint size overflow");
        }
        total += entry;
    }
    if (sizeof(uint32_t) > std::numeric_limits<uint64_t>::max() - total) {
        throw server_snapshot_manifest_error("snapshot manifest checkpoint size overflow");
    }
    total += sizeof(uint32_t);
    for (const auto & item : publish_advance_intents_) {
        const auto & intent = item.second;
        const uint64_t entry = CHECKPOINT_PUBLISH_ADVANCE_FIXED_BYTES +
            intent.owner.size() + intent.digest.size() + intent.retention_class.size() +
            intent.head_name.size() + intent.expected_digest.size();
        if (entry > std::numeric_limits<uint64_t>::max() - total) {
            throw server_snapshot_manifest_error("snapshot manifest checkpoint size overflow");
        }
        total += entry;
    }
    if (total > std::numeric_limits<uint32_t>::max() ||
            total + HEADER_FIXED_BYTES + compatibility_id_.size() > max_bytes_) {
        throw server_snapshot_manifest_error("live snapshot ownership exceeds the manifest byte budget", true);
    }
    std::vector<uint8_t> record;
    record.reserve((size_t) total);
    append_u32(record, (uint32_t) total);
    append_u32(record, RECORD_MAGIC);
    append_u64(record, revision);
    record.push_back((uint8_t) record_type::checkpoint);
    record.insert(record.end(), 7, 0);
    append_u32(record, (uint32_t) publish_intents_.size());
    append_u32(record, (uint32_t) managed_digests_.size());
    append_u32(record, CHECKPOINT_SCHEMA_ATOMIC_PUBLISH_ADVANCE);
    append_u32(record, (uint32_t) refs_.size());
    for (const auto & item : refs_) {
        const auto & ref = item.second;
        append_u32(record, (uint32_t) ref.owner.size());
        append_u32(record, (uint32_t) ref.digest.size());
        append_u32(record, (uint32_t) ref.retention_class.size());
        append_u64(record, ref.revision);
        record.insert(record.end(), ref.owner.begin(), ref.owner.end());
        record.insert(record.end(), ref.digest.begin(), ref.digest.end());
        record.insert(record.end(), ref.retention_class.begin(), ref.retention_class.end());
    }
    for (const auto & item : publish_intents_) {
        const auto & intent = item.second;
        append_u32(record, (uint32_t) intent.owner.size());
        append_u32(record, (uint32_t) intent.digest.size());
        append_u32(record, (uint32_t) intent.retention_class.size());
        append_u64(record, intent.revision);
        record.insert(record.end(), intent.owner.begin(), intent.owner.end());
        record.insert(record.end(), intent.digest.begin(), intent.digest.end());
        record.insert(record.end(), intent.retention_class.begin(), intent.retention_class.end());
    }
    for (const auto & digest : managed_digests_) {
        append_u32(record, (uint32_t) digest.size());
        record.insert(record.end(), digest.begin(), digest.end());
    }
    append_u32(record, (uint32_t) logical_heads_.size());
    for (const auto & item : logical_heads_) {
        const auto & head = item.second;
        append_u32(record, (uint32_t) head.name.size());
        append_u32(record, (uint32_t) head.digest.size());
        append_u32(record, (uint32_t) head.parent_digest.size());
        append_u64(record, head.generation);
        append_u64(record, head.revision);
        record.insert(record.end(), head.name.begin(), head.name.end());
        record.insert(record.end(), head.digest.begin(), head.digest.end());
        record.insert(record.end(), head.parent_digest.begin(), head.parent_digest.end());
    }
    append_u32(record, (uint32_t) logical_head_tombstones_.size());
    for (const auto & item : logical_head_tombstones_) {
        const auto & tombstone = item.second;
        const auto & head = tombstone.head;
        append_u32(record, (uint32_t) head.name.size());
        append_u32(record, (uint32_t) head.digest.size());
        append_u32(record, (uint32_t) head.parent_digest.size());
        append_u64(record, head.generation);
        append_u64(record, head.revision);
        append_u64(record, tombstone.deletion_revision);
        record.insert(record.end(), head.name.begin(), head.name.end());
        record.insert(record.end(), head.digest.begin(), head.digest.end());
        record.insert(record.end(), head.parent_digest.begin(), head.parent_digest.end());
    }
    append_u32(record, (uint32_t) publish_advance_intents_.size());
    for (const auto & item : publish_advance_intents_) {
        const auto & intent = item.second;
        append_u32(record, (uint32_t) intent.owner.size());
        append_u32(record, (uint32_t) intent.digest.size());
        append_u32(record, (uint32_t) intent.retention_class.size());
        append_u32(record, (uint32_t) intent.head_name.size());
        append_u32(record, (uint32_t) intent.expected_digest.size());
        append_u64(record, intent.expected_generation);
        append_u64(record, intent.revision);
        record.insert(record.end(), intent.owner.begin(), intent.owner.end());
        record.insert(record.end(), intent.digest.begin(), intent.digest.end());
        record.insert(record.end(), intent.retention_class.begin(), intent.retention_class.end());
        record.insert(record.end(), intent.head_name.begin(), intent.head_name.end());
        record.insert(record.end(), intent.expected_digest.begin(), intent.expected_digest.end());
    }
    const auto checksum = record_checksum(record.data(), record.size());
    record.insert(record.end(), checksum.begin(), checksum.end());
    return record;
}

void server_snapshot_manifest::apply_record(const std::vector<uint8_t> & record) {
    if (record.size() < RECORD_MIN_BYTES) {
        throw server_snapshot_manifest_error("snapshot manifest record is too small");
    }
    const auto expected_checksum = record_checksum(
            record.data(), record.size() - RECORD_CHECKSUM_BYTES);
    if (!std::equal(expected_checksum.begin(), expected_checksum.end(),
            record.end() - RECORD_CHECKSUM_BYTES)) {
        throw server_snapshot_manifest_error("snapshot manifest record checksum mismatch");
    }
    size_t offset = 0;
    const uint32_t total_size = read_u32(record, offset);
    const uint32_t magic = read_u32(record, offset);
    const uint64_t revision = read_u64(record, offset);
    if (total_size != record.size() || magic != RECORD_MAGIC || revision == 0) {
        throw server_snapshot_manifest_error("invalid snapshot manifest record header");
    }
    const record_type type = (record_type) record.at(offset++);
    for (size_t i = 0; i < 7; ++i) {
        if (record.at(offset++) != 0) {
            throw server_snapshot_manifest_error("invalid snapshot manifest reserved fields");
        }
    }
    const uint32_t owner_size = read_u32(record, offset);
    const uint32_t digest_size = read_u32(record, offset);
    const uint32_t class_size = read_u32(record, offset);
    const uint32_t n_refs = read_u32(record, offset);

    if (type == record_type::checkpoint) {
        if (record_count_ != 0 || class_size != CHECKPOINT_SCHEMA_ATOMIC_PUBLISH_ADVANCE) {
            throw server_snapshot_manifest_error("invalid snapshot manifest checkpoint placement");
        }
        std::map<std::string, server_snapshot_manifest_ref> checkpoint_refs;
        for (uint32_t i = 0; i < n_refs; ++i) {
            const uint32_t ref_owner_size = read_u32(record, offset);
            const uint32_t ref_digest_size = read_u32(record, offset);
            const uint32_t ref_class_size = read_u32(record, offset);
            const uint64_t ref_revision = read_u64(record, offset);
            const std::string owner = read_string(record, offset, ref_owner_size);
            const std::string digest = read_string(record, offset, ref_digest_size);
            const std::string retention_class = read_string(record, offset, ref_class_size);
            validate_owner(owner);
            validate_digest(digest);
            validate_retention_class(retention_class);
            if (ref_revision == 0 || ref_revision > revision) {
                throw server_snapshot_manifest_error("invalid snapshot manifest reference revision");
            }
            const auto inserted = checkpoint_refs.emplace(ref_key(owner, digest), server_snapshot_manifest_ref {
                owner, digest, retention_class, ref_revision,
            });
            if (!inserted.second) {
                throw server_snapshot_manifest_error("duplicate snapshot manifest checkpoint reference");
            }
        }
        std::map<std::string, server_snapshot_publish_intent> checkpoint_intents;
        for (uint32_t i = 0; i < owner_size; ++i) {
            const uint32_t intent_owner_size = read_u32(record, offset);
            const uint32_t intent_digest_size = read_u32(record, offset);
            const uint32_t intent_class_size = read_u32(record, offset);
            const uint64_t intent_revision = read_u64(record, offset);
            const std::string owner = read_string(record, offset, intent_owner_size);
            const std::string digest = read_string(record, offset, intent_digest_size);
            const std::string retention_class = read_string(record, offset, intent_class_size);
            validate_owner(owner);
            validate_digest(digest);
            validate_retention_class(retention_class);
            if (intent_revision == 0 || intent_revision > revision) {
                throw server_snapshot_manifest_error("invalid snapshot publish intent revision");
            }
            const std::string key = ref_key(owner, digest);
            if (checkpoint_refs.find(key) != checkpoint_refs.end() ||
                    !checkpoint_intents.emplace(key, server_snapshot_publish_intent {
                        owner, digest, retention_class, intent_revision,
                    }).second) {
                throw server_snapshot_manifest_error("duplicate snapshot publish checkpoint intent");
            }
        }
        std::set<std::string> checkpoint_managed;
        for (uint32_t i = 0; i < digest_size; ++i) {
            const uint32_t managed_digest_size = read_u32(record, offset);
            const std::string digest = read_string(record, offset, managed_digest_size);
            validate_digest(digest);
            if (!checkpoint_managed.insert(digest).second) {
                throw server_snapshot_manifest_error("duplicate managed snapshot checkpoint digest");
            }
        }
        std::map<std::string, server_snapshot_logical_head> checkpoint_heads;
        const uint32_t n_heads = read_u32(record, offset);
        for (uint32_t i = 0; i < n_heads; ++i) {
            const uint32_t name_size = read_u32(record, offset);
            const uint32_t head_digest_size = read_u32(record, offset);
            const uint32_t parent_digest_size = read_u32(record, offset);
            const uint64_t generation = read_u64(record, offset);
            const uint64_t head_revision = read_u64(record, offset);
            const std::string name = read_string(record, offset, name_size);
            const std::string digest = read_string(record, offset, head_digest_size);
            const std::string parent_digest = read_string(record, offset, parent_digest_size);
            validate_head_name(name);
            validate_digest(digest);
            if (!parent_digest.empty()) {
                validate_digest(parent_digest);
            }
            if (generation == 0 || head_revision == 0 || head_revision > revision) {
                throw server_snapshot_manifest_error("invalid logical head checkpoint generation");
            }
            if (!checkpoint_heads.emplace(name, server_snapshot_logical_head {
                    name, digest, parent_digest, generation, head_revision,
                }).second) {
                throw server_snapshot_manifest_error("duplicate logical head checkpoint name");
            }
        }
        std::map<std::string, server_snapshot_logical_head_tombstone> checkpoint_tombstones;
        const uint32_t n_tombstones = read_u32(record, offset);
        for (uint32_t i = 0; i < n_tombstones; ++i) {
            const uint32_t name_size = read_u32(record, offset);
            const uint32_t head_digest_size = read_u32(record, offset);
            const uint32_t parent_digest_size = read_u32(record, offset);
            const uint64_t generation = read_u64(record, offset);
            const uint64_t head_revision = read_u64(record, offset);
            const uint64_t deletion_revision = read_u64(record, offset);
            const std::string name = read_string(record, offset, name_size);
            const std::string digest = read_string(record, offset, head_digest_size);
            const std::string parent_digest = read_string(record, offset, parent_digest_size);
            validate_head_name(name);
            validate_digest(digest);
            if (!parent_digest.empty()) {
                validate_digest(parent_digest);
            }
            if (generation == 0 || head_revision == 0 || head_revision >= deletion_revision ||
                    deletion_revision > revision || checkpoint_heads.find(name) != checkpoint_heads.end()) {
                throw server_snapshot_manifest_error("invalid logical head tombstone checkpoint");
            }
            if (!checkpoint_tombstones.emplace(name, server_snapshot_logical_head_tombstone {
                    {name, digest, parent_digest, generation, head_revision}, deletion_revision,
                }).second) {
                throw server_snapshot_manifest_error("duplicate logical head tombstone checkpoint name");
            }
        }
        std::map<std::string, server_snapshot_publish_advance_intent> checkpoint_combined;
        const uint32_t n_combined = read_u32(record, offset);
        for (uint32_t i = 0; i < n_combined; ++i) {
            const uint32_t intent_owner_size = read_u32(record, offset);
            const uint32_t intent_digest_size = read_u32(record, offset);
            const uint32_t intent_class_size = read_u32(record, offset);
            const uint32_t head_name_size = read_u32(record, offset);
            const uint32_t expected_digest_size = read_u32(record, offset);
            const uint64_t expected_generation = read_u64(record, offset);
            const uint64_t intent_revision = read_u64(record, offset);
            const std::string owner = read_string(record, offset, intent_owner_size);
            const std::string digest = read_string(record, offset, intent_digest_size);
            const std::string retention_class = read_string(record, offset, intent_class_size);
            const std::string head_name = read_string(record, offset, head_name_size);
            const std::string expected_digest = read_string(record, offset, expected_digest_size);
            validate_owner(owner);
            validate_digest(digest);
            validate_retention_class(retention_class);
            validate_head_name(head_name);
            validate_digest(expected_digest);
            const std::string key = ref_key(owner, digest);
            const auto head = checkpoint_heads.find(head_name);
            const bool head_pending = std::any_of(
                    checkpoint_combined.begin(), checkpoint_combined.end(), [&](const auto & item) {
                        return item.second.head_name == head_name;
                    });
            if (expected_generation == 0 || intent_revision == 0 || intent_revision > revision ||
                    checkpoint_refs.find(key) != checkpoint_refs.end() ||
                    checkpoint_intents.find(key) != checkpoint_intents.end() ||
                    head == checkpoint_heads.end() || head->second.generation != expected_generation ||
                    head->second.digest != expected_digest || head_pending ||
                    !checkpoint_combined.emplace(key, server_snapshot_publish_advance_intent {
                        owner, digest, retention_class, head_name, expected_digest,
                        expected_generation, intent_revision,
                    }).second) {
                throw server_snapshot_manifest_error(
                        "invalid publish-and-advance checkpoint intent");
            }
        }
        if (offset != record.size() - RECORD_CHECKSUM_BYTES) {
            throw server_snapshot_manifest_error("snapshot manifest checkpoint has trailing payload");
        }
        refs_ = std::move(checkpoint_refs);
        publish_intents_ = std::move(checkpoint_intents);
        managed_digests_ = std::move(checkpoint_managed);
        logical_heads_ = std::move(checkpoint_heads);
        logical_head_tombstones_ = std::move(checkpoint_tombstones);
        publish_advance_intents_ = std::move(checkpoint_combined);
        revision_ = revision;
        return;
    }

    if (revision != revision_ + 1) {
        throw server_snapshot_manifest_error("snapshot manifest revision sequence is invalid");
    }
    const bool is_publish_advance = type == record_type::begin_publish_advance ||
        type == record_type::commit_publish_advance ||
        type == record_type::abort_publish_advance;
    if (is_publish_advance) {
        const uint32_t expected_digest_size = read_u32(record, offset);
        const uint64_t expected_generation = read_u64(record, offset);
        const std::string owner = read_string(record, offset, owner_size);
        const std::string digest = read_string(record, offset, digest_size);
        const std::string retention_class = read_string(record, offset, class_size);
        const std::string head_name = read_string(record, offset, n_refs);
        const std::string expected_digest = read_string(record, offset, expected_digest_size);
        if (offset != record.size() - RECORD_CHECKSUM_BYTES) {
            throw server_snapshot_manifest_error(
                    "publish-and-advance mutation has trailing payload");
        }
        validate_owner(owner);
        validate_digest(digest);
        validate_retention_class(retention_class);
        validate_head_name(head_name);
        validate_digest(expected_digest);
        if (expected_generation == 0) {
            throw server_snapshot_manifest_error(
                    "invalid publish-and-advance expected generation");
        }
        const std::string key = ref_key(owner, digest);
        const server_snapshot_publish_advance_intent payload {
            owner, digest, retention_class, head_name, expected_digest,
            expected_generation, revision,
        };
        const auto same_payload = [&](const server_snapshot_publish_advance_intent & pending) {
            return pending.owner == owner && pending.digest == digest &&
                pending.retention_class == retention_class && pending.head_name == head_name &&
                pending.expected_digest == expected_digest &&
                pending.expected_generation == expected_generation;
        };
        if (type == record_type::begin_publish_advance) {
            const auto head = logical_heads_.find(head_name);
            const bool head_pending = std::any_of(
                    publish_advance_intents_.begin(), publish_advance_intents_.end(),
                    [&](const auto & item) { return item.second.head_name == head_name; });
            if (refs_.find(key) != refs_.end() || publish_intents_.find(key) != publish_intents_.end() ||
                    publish_advance_intents_.find(key) != publish_advance_intents_.end() ||
                    head == logical_heads_.end() || head->second.generation != expected_generation ||
                    head->second.digest != expected_digest || head_pending) {
                throw server_snapshot_manifest_error(
                        "invalid publish-and-advance begin transition");
            }
            publish_advance_intents_[key] = payload;
        } else {
            const auto pending = publish_advance_intents_.find(key);
            if (pending == publish_advance_intents_.end() || !same_payload(pending->second)) {
                throw server_snapshot_manifest_error(
                        "publish-and-advance terminal record references no matching intent");
            }
            const auto head = logical_heads_.find(head_name);
            if (refs_.find(key) != refs_.end() || head == logical_heads_.end() ||
                    head->second.generation != expected_generation ||
                    head->second.digest != expected_digest) {
                throw server_snapshot_manifest_error(
                        "invalid publish-and-advance terminal transition");
            }
            if (type == record_type::commit_publish_advance) {
                if (expected_generation == std::numeric_limits<uint64_t>::max()) {
                    throw server_snapshot_manifest_error(
                            "invalid publish-and-advance commit transition");
                }
                refs_[key] = {owner, digest, retention_class, revision};
                managed_digests_.insert(digest);
                head->second = {
                    head_name, digest, expected_digest, expected_generation + 1, revision,
                };
            }
            publish_advance_intents_.erase(pending);
        }
        revision_ = revision;
        return;
    }
    if (n_refs != 0) {
        throw server_snapshot_manifest_error("snapshot manifest revision sequence is invalid");
    }
    const bool is_head_mutation = type == record_type::create_head ||
        type == record_type::advance_head || type == record_type::delete_head;
    const uint64_t head_generation = is_head_mutation ? read_u64(record, offset) : 0;
    const std::string owner = read_string(record, offset, owner_size);
    const std::string digest = read_string(record, offset, digest_size);
    const std::string retention_class = read_string(record, offset, class_size);
    if (offset != record.size() - RECORD_CHECKSUM_BYTES) {
        throw server_snapshot_manifest_error("snapshot manifest mutation has trailing payload");
    }
    validate_digest(digest);
    if (type == record_type::evict_cache || type == record_type::forget_managed) {
        if (!owner.empty() || !retention_class.empty()) {
            throw server_snapshot_manifest_error("invalid managed snapshot mutation payload");
        }
    } else if (!is_head_mutation) {
        validate_owner(owner);
    } else {
        validate_head_name(owner);
        if (!retention_class.empty()) {
            validate_digest(retention_class);
        }
    }
    const std::string key = ref_key(owner, digest);
    if (type == record_type::retain) {
        validate_retention_class(retention_class);
        if (publish_intents_.find(key) != publish_intents_.end() ||
                publish_advance_intents_.find(key) != publish_advance_intents_.end()) {
            throw server_snapshot_manifest_error("snapshot retain conflicts with a publish intent");
        }
        refs_[key] = {owner, digest, retention_class, revision};
    } else if (type == record_type::release) {
        if (!retention_class.empty() || refs_.erase(key) != 1) {
            throw server_snapshot_manifest_error("snapshot manifest release references no live owner");
        }
    } else if (type == record_type::begin_publish) {
        validate_retention_class(retention_class);
        if (refs_.find(key) != refs_.end() || publish_intents_.find(key) != publish_intents_.end() ||
                publish_advance_intents_.find(key) != publish_advance_intents_.end()) {
            throw server_snapshot_manifest_error("snapshot publish intent conflicts with existing ownership");
        }
        publish_intents_[key] = {owner, digest, retention_class, revision};
    } else if (type == record_type::commit_publish) {
        const auto pending = publish_intents_.find(key);
        if (!retention_class.empty() || pending == publish_intents_.end()) {
            throw server_snapshot_manifest_error("snapshot publish commit references no live intent");
        }
        refs_[key] = {owner, digest, pending->second.retention_class, revision};
        publish_intents_.erase(pending);
        managed_digests_.insert(digest);
    } else if (type == record_type::abort_publish) {
        if (!retention_class.empty() || publish_intents_.erase(key) != 1) {
            throw server_snapshot_manifest_error("snapshot publish abort references no live intent");
        }
    } else if (type == record_type::evict_cache) {
        if (managed_digests_.find(digest) == managed_digests_.end() ||
                publish_intent_count(digest) != 0 ||
                std::any_of(logical_heads_.begin(), logical_heads_.end(), [&](const auto & item) {
                    return item.second.digest == digest;
                })) {
            throw server_snapshot_manifest_error("cache eviction references unmanaged or pending content");
        }
        const bool pinned = std::any_of(refs_.begin(), refs_.end(), [&](const auto & item) {
            return item.second.digest == digest && item.second.retention_class != "cache";
        });
        if (pinned) {
            throw server_snapshot_manifest_error("cache eviction crossed a pinned owner fence");
        }
        uint64_t released = 0;
        for (auto it = refs_.begin(); it != refs_.end();) {
            if (it->second.digest == digest) {
                it = refs_.erase(it);
                released++;
            } else {
                ++it;
            }
        }
        if (released == 0) {
            throw server_snapshot_manifest_error("cache eviction references no live cache owners");
        }
    } else if (type == record_type::forget_managed) {
        if (ref_count(digest) != 0 || publish_intent_count(digest) != 0 ||
                managed_digests_.erase(digest) != 1) {
            throw server_snapshot_manifest_error("managed snapshot cannot be forgotten while reachable");
        }
    } else if (type == record_type::create_head) {
        const bool head_pending = std::any_of(
                publish_advance_intents_.begin(), publish_advance_intents_.end(),
                [&](const auto & item) { return item.second.head_name == owner; });
        if (!retention_class.empty() || head_generation != 1 ||
                logical_heads_.find(owner) != logical_heads_.end() ||
                logical_head_tombstones_.find(owner) != logical_head_tombstones_.end() || head_pending) {
            throw server_snapshot_manifest_error("invalid logical head create transition");
        }
        logical_heads_[owner] = {owner, digest, {}, head_generation, revision};
    } else if (type == record_type::advance_head) {
        const auto found = logical_heads_.find(owner);
        const bool head_pending = std::any_of(
                publish_advance_intents_.begin(), publish_advance_intents_.end(),
                [&](const auto & item) { return item.second.head_name == owner; });
        if (found == logical_heads_.end() || head_generation != found->second.generation + 1 ||
                retention_class != found->second.digest || head_pending) {
            throw server_snapshot_manifest_error("invalid logical head advance transition");
        }
        found->second = {owner, digest, retention_class, head_generation, revision};
    } else if (type == record_type::delete_head) {
        const auto found = logical_heads_.find(owner);
        const bool head_pending = std::any_of(
                publish_advance_intents_.begin(), publish_advance_intents_.end(),
                [&](const auto & item) { return item.second.head_name == owner; });
        if (found == logical_heads_.end() || !retention_class.empty() ||
                head_generation != found->second.generation || digest != found->second.digest || head_pending) {
            throw server_snapshot_manifest_error("invalid logical head delete transition");
        }
        logical_head_tombstones_[owner] = {found->second, revision};
        logical_heads_.erase(found);
    } else {
        throw server_snapshot_manifest_error("unknown snapshot manifest record type");
    }
    revision_ = revision;
}

void server_snapshot_manifest::discover() {
    std::error_code error;
    for (const auto & item : std::filesystem::directory_iterator(namespace_path_, error)) {
        if (error) {
            throw server_snapshot_manifest_error("failed to scan snapshot manifest namespace");
        }
        const std::string filename = item.path().filename().string();
        if (item.is_regular_file(error) && !error && filename.rfind(".manifest.tmp-", 0) == 0) {
            if (!std::filesystem::remove(item.path(), error) || error) {
                throw server_snapshot_manifest_error("failed to remove interrupted snapshot manifest temp file");
            }
            recovered_temp_files_++;
        }
        error.clear();
    }
    if (error) {
        throw server_snapshot_manifest_error("failed to scan snapshot manifest namespace");
    }
    if (recovered_temp_files_ > 0) {
        sync_namespace_directory();
    }

    if (!std::filesystem::exists(path_, error)) {
        if (error) {
            throw server_snapshot_manifest_error("failed to inspect snapshot manifest");
        }
        write_compacted({});
        return;
    }
    const uint64_t physical_size = std::filesystem::file_size(path_, error);
    if (error || physical_size > max_bytes_ || physical_size > std::numeric_limits<size_t>::max()) {
        throw server_snapshot_manifest_error("snapshot manifest exceeds its byte budget", true);
    }
    std::ifstream input(path_, std::ios::binary);
    std::vector<uint8_t> data((size_t) physical_size);
    input.read((char *) data.data(), data.size());
    if (!input) {
        throw server_snapshot_manifest_error("failed to read snapshot manifest");
    }
    if (data.size() < HEADER_FIXED_BYTES ||
            !std::equal(MANIFEST_MAGIC.begin(), MANIFEST_MAGIC.end(), data.begin())) {
        throw server_snapshot_manifest_error("invalid snapshot manifest header");
    }
    size_t offset = MANIFEST_MAGIC.size();
    if (read_u32(data, offset) != MANIFEST_VERSION) {
        throw server_snapshot_manifest_error("unsupported snapshot manifest version");
    }
    const uint32_t compatibility_size = read_u32(data, offset);
    if (compatibility_size == 0 || compatibility_size > MAX_COMPATIBILITY_BYTES ||
            read_string(data, offset, compatibility_size) != compatibility_id_) {
        throw server_snapshot_manifest_error("snapshot manifest compatibility ID mismatch");
    }
    const size_t header_bytes = offset;
    size_t valid_bytes = offset;
    while (offset < data.size()) {
        const size_t remaining = data.size() - offset;
        if (remaining < RECORD_MIN_BYTES) {
            break;
        }
        size_t size_offset = offset;
        const uint32_t record_size = read_u32(data, size_offset);
        if (record_size < RECORD_MIN_BYTES || record_size > max_bytes_) {
            throw server_snapshot_manifest_error("invalid snapshot manifest record size");
        }
        if (record_size > remaining) {
            break;
        }
        std::vector<uint8_t> record(data.begin() + offset, data.begin() + offset + record_size);
        apply_record(record);
        record_count_++;
        offset += record_size;
        valid_bytes = offset;
    }
    if (valid_bytes < data.size()) {
        recovered_tail_bytes_ = data.size() - valid_bytes;
        std::filesystem::resize_file(path_, valid_bytes, error);
        if (error) {
            throw server_snapshot_manifest_error("failed to truncate interrupted snapshot manifest tail");
        }
#if !defined(_WIN32)
        sync_file(path_);
#endif
        sync_namespace_directory();
    }
    file_bytes_ = valid_bytes;
    high_water_bytes_ = file_bytes_;
    if (file_bytes_ < header_bytes) {
        throw server_snapshot_manifest_error("snapshot manifest header was truncated");
    }
}

void server_snapshot_manifest::append_record(const std::vector<uint8_t> & record) {
    if (record.size() > max_bytes_ - std::min(max_bytes_, file_bytes_)) {
        compact();
    }
    if (file_bytes_ > max_bytes_ || record.size() > max_bytes_ - file_bytes_) {
        throw server_snapshot_manifest_error("snapshot manifest byte budget is exhausted", true);
    }
    const uint64_t old_size = file_bytes_;
#if !defined(_WIN32)
    int fd = ::open(path_.c_str(), O_WRONLY | O_APPEND | O_CLOEXEC);
    if (fd < 0) {
        throw server_snapshot_manifest_error("failed to open snapshot manifest for append");
    }
    try {
        write_all(fd, record.data(), record.size());
        if (::fsync(fd) != 0) {
            throw server_snapshot_manifest_error(
                    std::string("failed to sync snapshot manifest append: ") + std::strerror(errno));
        }
        if (::close(fd) != 0) {
            fd = -1;
            throw server_snapshot_manifest_error("failed to close snapshot manifest append");
        }
        fd = -1;
    } catch (...) {
        if (fd >= 0) {
            const bool rollback_failed = ::ftruncate(fd, (off_t) old_size) != 0 || ::fsync(fd) != 0;
            ::close(fd);
            if (rollback_failed) {
                throw server_snapshot_manifest_error(
                        "snapshot manifest append failed and its partial tail could not be rolled back");
            }
        } else {
            std::error_code error;
            std::filesystem::resize_file(path_, old_size, error);
            try {
                sync_file(path_);
            } catch (...) {
            }
        }
        throw;
    }
#else
    std::ofstream output(path_, std::ios::binary | std::ios::app);
    output.write((const char *) record.data(), record.size());
    output.flush();
    if (!output) {
        output.close();
        std::error_code error;
        std::filesystem::resize_file(path_, old_size, error);
        throw server_snapshot_manifest_error("failed to append snapshot manifest");
    }
#endif
    file_bytes_ += record.size();
    high_water_bytes_ = std::max(high_water_bytes_, file_bytes_);
    record_count_++;
}

void server_snapshot_manifest::write_compacted(const std::vector<uint8_t> & checkpoint) {
    std::vector<uint8_t> bytes = encode_header();
    if (!checkpoint.empty()) {
        if (checkpoint.size() > max_bytes_ - bytes.size()) {
            throw server_snapshot_manifest_error("snapshot manifest checkpoint exceeds its byte budget", true);
        }
        bytes.insert(bytes.end(), checkpoint.begin(), checkpoint.end());
    }
    const std::filesystem::path temp_path = std::filesystem::path(namespace_path_) /
        (".manifest.tmp-" + std::to_string(++temp_sequence_));
#if !defined(_WIN32)
    int fd = ::open(temp_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, S_IRUSR | S_IWUSR);
    if (fd < 0) {
        throw server_snapshot_manifest_error("failed to create compacted snapshot manifest");
    }
    bool published = false;
    try {
        write_all(fd, bytes.data(), bytes.size());
        if (::fsync(fd) != 0) {
            throw server_snapshot_manifest_error("failed to sync compacted snapshot manifest");
        }
        if (::close(fd) != 0) {
            fd = -1;
            throw server_snapshot_manifest_error("failed to close compacted snapshot manifest");
        }
        fd = -1;
        if (::rename(temp_path.c_str(), path_.c_str()) != 0) {
            throw server_snapshot_manifest_error("failed to publish compacted snapshot manifest");
        }
        published = true;
        sync_namespace_directory();
    } catch (...) {
        if (fd >= 0) {
            ::close(fd);
        }
        if (!published) {
            std::error_code error;
            std::filesystem::remove(temp_path, error);
        }
        throw;
    }
#else
    {
        std::ofstream output(temp_path, std::ios::binary | std::ios::trunc);
        output.write((const char *) bytes.data(), bytes.size());
        output.flush();
        if (!output) {
            throw server_snapshot_manifest_error("failed to write compacted snapshot manifest");
        }
    }
    std::error_code error;
    std::filesystem::rename(temp_path, path_, error);
    if (error) {
        std::filesystem::remove(temp_path, error);
        throw server_snapshot_manifest_error("failed to publish compacted snapshot manifest");
    }
#endif
    file_bytes_ = bytes.size();
    high_water_bytes_ = std::max(high_water_bytes_, file_bytes_);
    record_count_ = checkpoint.empty() ? 0 : 1;
}

server_snapshot_manifest_result server_snapshot_manifest::retain(
        const std::string & owner,
        const std::string & digest,
        const std::string & retention_class) {
    validate_owner(owner);
    validate_digest(digest);
    validate_retention_class(retention_class);
    const std::string key = ref_key(owner, digest);
    if (publish_intents_.find(key) != publish_intents_.end() ||
            publish_advance_intents_.find(key) != publish_advance_intents_.end()) {
        throw server_snapshot_manifest_error("snapshot ownership publish is still pending", true);
    }
    const auto found = refs_.find(key);
    if (found != refs_.end() && found->second.retention_class == retention_class) {
        return {found->second, true, revision_, file_bytes_};
    }
    if (revision_ == std::numeric_limits<uint64_t>::max()) {
        throw server_snapshot_manifest_error("snapshot manifest revision space is exhausted");
    }
    const uint64_t next_revision = revision_ + 1;
    const auto record = encode_mutation(
            record_type::retain, next_revision, owner, digest, retention_class);
    append_record(record);
    server_snapshot_manifest_ref ref {owner, digest, retention_class, next_revision};
    refs_[key] = ref;
    revision_ = next_revision;
    return {ref, false, revision_, file_bytes_};
}

void server_snapshot_manifest::validate_publish_terminal_space(
        const server_snapshot_publish_intent & intent) {
    if (revision_ > std::numeric_limits<uint64_t>::max() - 2) {
        throw server_snapshot_manifest_error("snapshot manifest revision space is exhausted");
    }
    const std::string key = ref_key(intent.owner, intent.digest);
    const auto inserted = publish_intents_.emplace(key, intent);
    if (!inserted.second) {
        throw server_snapshot_manifest_error("snapshot publish intent already exists");
    }
    const auto managed_inserted = managed_digests_.insert(intent.digest);
    try {
        const auto checkpoint = encode_checkpoint(revision_ + 1);
        const auto terminal = encode_mutation(
                record_type::commit_publish,
                revision_ + 2,
                intent.owner,
                intent.digest,
                "");
        const uint64_t header_bytes = encode_header().size();
        if (header_bytes > max_bytes_ || checkpoint.size() > max_bytes_ - header_bytes ||
                terminal.size() > max_bytes_ - header_bytes - checkpoint.size()) {
            throw server_snapshot_manifest_error(
                    "snapshot publish transaction exceeds the manifest byte budget", true);
        }
    } catch (...) {
        publish_intents_.erase(inserted.first);
        if (managed_inserted.second) {
            managed_digests_.erase(managed_inserted.first);
        }
        throw;
    }
    publish_intents_.erase(inserted.first);
    if (managed_inserted.second) {
        managed_digests_.erase(managed_inserted.first);
    }
}

server_snapshot_publish_begin_result server_snapshot_manifest::begin_publish(
        const std::string & owner,
        const std::string & digest,
        const std::string & retention_class) {
    validate_owner(owner);
    validate_digest(digest);
    validate_retention_class(retention_class);
    const std::string key = ref_key(owner, digest);
    const auto retained = refs_.find(key);
    if (retained != refs_.end()) {
        return {{}, retained->second, true,
            retained->second.retention_class == retention_class, revision_, file_bytes_};
    }
    const auto pending = publish_intents_.find(key);
    if (pending != publish_intents_.end()) {
        if (pending->second.retention_class != retention_class) {
            throw server_snapshot_manifest_error(
                    "snapshot publish retry changed retention_class", true);
        }
        return {pending->second, {}, false, true, revision_, file_bytes_};
    }
    if (publish_advance_intents_.find(key) != publish_advance_intents_.end()) {
        throw server_snapshot_manifest_error(
                "snapshot publish conflicts with publish-and-advance", true);
    }
    if (revision_ == std::numeric_limits<uint64_t>::max()) {
        throw server_snapshot_manifest_error("snapshot manifest revision space is exhausted");
    }
    const uint64_t next_revision = revision_ + 1;
    server_snapshot_publish_intent intent {owner, digest, retention_class, next_revision};
    validate_publish_terminal_space(intent);
    const auto record = encode_mutation(
            record_type::begin_publish, next_revision, owner, digest, retention_class);
    append_record(record);
    publish_intents_[key] = intent;
    revision_ = next_revision;
    return {intent, {}, false, false, revision_, file_bytes_};
}

server_snapshot_manifest_result server_snapshot_manifest::commit_publish(
        const std::string & owner,
        const std::string & digest) {
    validate_owner(owner);
    validate_digest(digest);
    const std::string key = ref_key(owner, digest);
    const auto pending = publish_intents_.find(key);
    if (pending == publish_intents_.end()) {
        throw server_snapshot_manifest_error("snapshot publish intent is unavailable", true);
    }
    if (revision_ == std::numeric_limits<uint64_t>::max()) {
        throw server_snapshot_manifest_error("snapshot manifest revision space is exhausted");
    }
    const server_snapshot_publish_intent intent = pending->second;
    const uint64_t next_revision = revision_ + 1;
    const auto record = encode_mutation(
            record_type::commit_publish, next_revision, owner, digest, "");
    append_record(record);
    server_snapshot_manifest_ref ref {owner, digest, intent.retention_class, next_revision};
    publish_intents_.erase(pending);
    refs_[key] = ref;
    managed_digests_.insert(intent.digest);
    revision_ = next_revision;
    return {ref, false, revision_, file_bytes_};
}

server_snapshot_manifest_result server_snapshot_manifest::abort_publish(
        const std::string & owner,
        const std::string & digest) {
    validate_owner(owner);
    validate_digest(digest);
    const std::string key = ref_key(owner, digest);
    const auto pending = publish_intents_.find(key);
    if (pending == publish_intents_.end()) {
        throw server_snapshot_manifest_error("snapshot publish intent is unavailable", true);
    }
    if (revision_ == std::numeric_limits<uint64_t>::max()) {
        throw server_snapshot_manifest_error("snapshot manifest revision space is exhausted");
    }
    const auto intent = pending->second;
    const uint64_t next_revision = revision_ + 1;
    const auto record = encode_mutation(
            record_type::abort_publish, next_revision, owner, digest, "");
    append_record(record);
    publish_intents_.erase(pending);
    revision_ = next_revision;
    return {{intent.owner, intent.digest, intent.retention_class, intent.revision},
        false, revision_, file_bytes_};
}

void server_snapshot_manifest::validate_publish_advance_terminal_space(
        const server_snapshot_publish_advance_intent & intent) {
    if (revision_ > std::numeric_limits<uint64_t>::max() - 2 ||
            intent.expected_generation == std::numeric_limits<uint64_t>::max()) {
        throw server_snapshot_manifest_error(
                "publish-and-advance revision or generation space is exhausted");
    }
    const std::string key = ref_key(intent.owner, intent.digest);
    const auto inserted = publish_advance_intents_.emplace(key, intent);
    if (!inserted.second) {
        throw server_snapshot_manifest_error("publish-and-advance intent already exists");
    }
    try {
        const auto checkpoint = encode_checkpoint(revision_ + 1);
        const auto commit = encode_publish_advance_mutation(
                record_type::commit_publish_advance, revision_ + 2, intent);
        const auto abort = encode_publish_advance_mutation(
                record_type::abort_publish_advance, revision_ + 2, intent);
        const uint64_t terminal_bytes = std::max(commit.size(), abort.size());
        const uint64_t header_bytes = encode_header().size();
        if (header_bytes > max_bytes_ || checkpoint.size() > max_bytes_ - header_bytes ||
                terminal_bytes > max_bytes_ - header_bytes - checkpoint.size()) {
            throw server_snapshot_manifest_error(
                    "publish-and-advance transaction exceeds the manifest byte budget", true);
        }

        publish_advance_intents_.erase(inserted.first);
        const auto ref_inserted = refs_.emplace(key, server_snapshot_manifest_ref {
            intent.owner, intent.digest, intent.retention_class, revision_ + 2,
        });
        const auto managed_inserted = managed_digests_.insert(intent.digest);
        auto head = logical_heads_.find(intent.head_name);
        const server_snapshot_logical_head previous = head->second;
        head->second = {
            intent.head_name, intent.digest, intent.expected_digest,
            intent.expected_generation + 1, revision_ + 2,
        };
        try {
            encode_checkpoint(revision_ + 2);
        } catch (...) {
            head->second = previous;
            if (managed_inserted.second) {
                managed_digests_.erase(managed_inserted.first);
            }
            refs_.erase(ref_inserted.first);
            publish_advance_intents_.emplace(key, intent);
            throw;
        }
        head->second = previous;
        if (managed_inserted.second) {
            managed_digests_.erase(managed_inserted.first);
        }
        refs_.erase(ref_inserted.first);
        publish_advance_intents_.emplace(key, intent);
    } catch (...) {
        publish_advance_intents_.erase(key);
        throw;
    }
    publish_advance_intents_.erase(key);
}

server_snapshot_publish_advance_begin_result server_snapshot_manifest::begin_publish_advance(
        const std::string & owner,
        const std::string & digest,
        const std::string & retention_class,
        const std::string & head_name,
        uint64_t expected_generation,
        const std::string & expected_digest) {
    validate_owner(owner);
    validate_digest(digest);
    validate_retention_class(retention_class);
    validate_head_name(head_name);
    validate_digest(expected_digest);
    if (expected_generation == 0) {
        throw server_snapshot_manifest_error(
                "publish-and-advance expected generation must be positive");
    }
    const std::string key = ref_key(owner, digest);
    const auto retained = refs_.find(key);
    const auto head = logical_heads_.find(head_name);
    const bool committed_head = expected_generation < std::numeric_limits<uint64_t>::max() &&
        head != logical_heads_.end() && head->second.generation == expected_generation + 1 &&
        head->second.digest == digest && head->second.parent_digest == expected_digest;
    const bool committed_ref = retained != refs_.end() &&
        retained->second.retention_class == retention_class;
    if (committed_head || retained != refs_.end()) {
        if (committed_head && committed_ref) {
            return {{}, retained->second, head->second, true, true, revision_, file_bytes_};
        }
        throw server_snapshot_manifest_error(
                "publish-and-advance conflicts with partially committed state", true);
    }

    const auto pending = publish_advance_intents_.find(key);
    if (pending != publish_advance_intents_.end()) {
        const auto & intent = pending->second;
        if (intent.retention_class != retention_class || intent.head_name != head_name ||
                intent.expected_generation != expected_generation ||
                intent.expected_digest != expected_digest) {
            throw server_snapshot_manifest_error(
                    "publish-and-advance retry changed transaction fields", true);
        }
        return {intent, {}, {}, false, true, revision_, file_bytes_};
    }
    if (publish_intents_.find(key) != publish_intents_.end() ||
            std::any_of(publish_advance_intents_.begin(), publish_advance_intents_.end(),
                [&](const auto & item) { return item.second.head_name == head_name; })) {
        throw server_snapshot_manifest_error("publish-and-advance transaction conflicts", true);
    }
    if (head == logical_heads_.end() || head->second.generation != expected_generation ||
            head->second.digest != expected_digest) {
        throw server_snapshot_manifest_error("logical head compare-and-swap conflict", true);
    }
    if (revision_ == std::numeric_limits<uint64_t>::max()) {
        throw server_snapshot_manifest_error("snapshot manifest revision space is exhausted");
    }
    const uint64_t next_revision = revision_ + 1;
    const server_snapshot_publish_advance_intent intent {
        owner, digest, retention_class, head_name, expected_digest,
        expected_generation, next_revision,
    };
    validate_publish_advance_terminal_space(intent);
    const auto record = encode_publish_advance_mutation(
            record_type::begin_publish_advance, next_revision, intent);
    append_record(record);
    publish_advance_intents_[key] = intent;
    revision_ = next_revision;
    return {intent, {}, {}, false, false, revision_, file_bytes_};
}

server_snapshot_publish_advance_result server_snapshot_manifest::commit_publish_advance(
        const std::string & owner,
        const std::string & digest) {
    validate_owner(owner);
    validate_digest(digest);
    const std::string key = ref_key(owner, digest);
    const auto pending = publish_advance_intents_.find(key);
    if (pending == publish_advance_intents_.end()) {
        throw server_snapshot_manifest_error("publish-and-advance intent is unavailable", true);
    }
    const auto intent = pending->second;
    const auto head = logical_heads_.find(intent.head_name);
    if (head == logical_heads_.end() || head->second.generation != intent.expected_generation ||
            head->second.digest != intent.expected_digest || refs_.find(key) != refs_.end()) {
        throw server_snapshot_manifest_error(
                "publish-and-advance commit invariant is unavailable");
    }
    if (revision_ == std::numeric_limits<uint64_t>::max() ||
            intent.expected_generation == std::numeric_limits<uint64_t>::max()) {
        throw server_snapshot_manifest_error(
                "publish-and-advance revision or generation space is exhausted");
    }
    const uint64_t next_revision = revision_ + 1;
    const auto record = encode_publish_advance_mutation(
            record_type::commit_publish_advance, next_revision, intent);
    append_record(record);
    const server_snapshot_manifest_ref ref {
        intent.owner, intent.digest, intent.retention_class, next_revision,
    };
    const server_snapshot_logical_head advanced {
        intent.head_name, intent.digest, intent.expected_digest,
        intent.expected_generation + 1, next_revision,
    };
    publish_advance_intents_.erase(pending);
    refs_[key] = ref;
    managed_digests_.insert(intent.digest);
    head->second = advanced;
    revision_ = next_revision;
    return {ref, advanced, false, revision_, file_bytes_};
}

server_snapshot_publish_advance_result server_snapshot_manifest::abort_publish_advance(
        const std::string & owner,
        const std::string & digest) {
    validate_owner(owner);
    validate_digest(digest);
    const std::string key = ref_key(owner, digest);
    const auto pending = publish_advance_intents_.find(key);
    if (pending == publish_advance_intents_.end()) {
        throw server_snapshot_manifest_error("publish-and-advance intent is unavailable", true);
    }
    if (revision_ == std::numeric_limits<uint64_t>::max()) {
        throw server_snapshot_manifest_error("snapshot manifest revision space is exhausted");
    }
    const auto intent = pending->second;
    const auto head = logical_heads_.find(intent.head_name);
    const uint64_t next_revision = revision_ + 1;
    const auto record = encode_publish_advance_mutation(
            record_type::abort_publish_advance, next_revision, intent);
    append_record(record);
    publish_advance_intents_.erase(pending);
    revision_ = next_revision;
    return {{intent.owner, intent.digest, intent.retention_class, intent.revision},
        head == logical_heads_.end() ? server_snapshot_logical_head {} : head->second,
        false, revision_, file_bytes_};
}

std::vector<server_snapshot_cache_eviction_candidate>
server_snapshot_manifest::cache_eviction_candidates() const {
    std::map<std::string, server_snapshot_cache_eviction_candidate> candidates;
    std::set<std::string> fenced;
    for (const auto & digest : managed_digests_) {
        if (publish_intent_count(digest) == 0) {
            candidates.emplace(digest, server_snapshot_cache_eviction_candidate {digest, 0, 0});
        }
    }
    for (const auto & item : refs_) {
        const auto & ref = item.second;
        const auto found = candidates.find(ref.digest);
        if (found == candidates.end()) {
            continue;
        }
        if (ref.retention_class != "cache") {
            fenced.insert(ref.digest);
            continue;
        }
        found->second.refs++;
        found->second.newest_ref_revision = std::max(
                found->second.newest_ref_revision, ref.revision);
    }
    for (const auto & item : logical_heads_) {
        fenced.insert(item.second.digest);
    }
    std::vector<server_snapshot_cache_eviction_candidate> result;
    for (const auto & item : candidates) {
        if (fenced.find(item.first) == fenced.end()) {
            result.push_back(item.second);
        }
    }
    std::sort(result.begin(), result.end(), [](const auto & left, const auto & right) {
        return std::tie(left.newest_ref_revision, left.digest) <
            std::tie(right.newest_ref_revision, right.digest);
    });
    return result;
}

server_snapshot_cache_eviction_result server_snapshot_manifest::evict_cache(
        const std::string & digest) {
    validate_digest(digest);
    if (managed_digests_.find(digest) == managed_digests_.end() ||
            publish_intent_count(digest) != 0) {
        throw server_snapshot_manifest_error("snapshot content is not an evictable managed object", true);
    }
    std::vector<std::pair<std::string, server_snapshot_manifest_ref>> victims;
    for (const auto & item : refs_) {
        if (item.second.digest != digest) {
            continue;
        }
        if (item.second.retention_class != "cache") {
            throw server_snapshot_manifest_error("snapshot content has a pinned owner", true);
        }
        victims.push_back(item);
    }
    if (victims.empty()) {
        throw server_snapshot_manifest_error("snapshot content has no live cache owners", true);
    }
    if (revision_ == std::numeric_limits<uint64_t>::max()) {
        throw server_snapshot_manifest_error("snapshot manifest revision space is exhausted");
    }
    const uint64_t next_revision = revision_ + 1;
    const auto record = encode_mutation(record_type::evict_cache, next_revision, "", digest, "");
    try {
        append_record(record);
    } catch (const server_snapshot_manifest_error & error) {
        if (!error.unavailable) {
            throw;
        }
        for (const auto & victim : victims) {
            refs_.erase(victim.first);
        }
        std::vector<uint8_t> checkpoint;
        try {
            checkpoint = encode_checkpoint(next_revision);
        } catch (...) {
            for (const auto & victim : victims) {
                refs_.insert(victim);
            }
            throw;
        }
        for (const auto & victim : victims) {
            refs_.insert(victim);
        }
        write_compacted(checkpoint);
        compactions_++;
    }
    for (const auto & victim : victims) {
        refs_.erase(victim.first);
    }
    revision_ = next_revision;
    return {digest, victims.size(), revision_, file_bytes_};
}

void server_snapshot_manifest::forget_managed(const std::string & digest) {
    validate_digest(digest);
    if (ref_count(digest) != 0 || publish_intent_count(digest) != 0 ||
            managed_digests_.find(digest) == managed_digests_.end()) {
        throw server_snapshot_manifest_error("managed snapshot is still reachable or unavailable", true);
    }
    if (revision_ == std::numeric_limits<uint64_t>::max()) {
        throw server_snapshot_manifest_error("snapshot manifest revision space is exhausted");
    }
    const uint64_t next_revision = revision_ + 1;
    const auto record = encode_mutation(record_type::forget_managed, next_revision, "", digest, "");
    try {
        append_record(record);
    } catch (const server_snapshot_manifest_error & error) {
        if (!error.unavailable) {
            throw;
        }
        managed_digests_.erase(digest);
        std::vector<uint8_t> checkpoint;
        try {
            checkpoint = encode_checkpoint(next_revision);
        } catch (...) {
            managed_digests_.insert(digest);
            throw;
        }
        managed_digests_.insert(digest);
        write_compacted(checkpoint);
        compactions_++;
    }
    managed_digests_.erase(digest);
    revision_ = next_revision;
}

server_snapshot_logical_head_result server_snapshot_manifest::create_head(
        const std::string & name,
        const std::string & digest) {
    validate_head_name(name);
    validate_digest(digest);
    if (std::any_of(publish_advance_intents_.begin(), publish_advance_intents_.end(),
            [&](const auto & item) { return item.second.head_name == name; })) {
        throw server_snapshot_manifest_error("logical head transaction is pending", true);
    }
    const auto found = logical_heads_.find(name);
    if (found != logical_heads_.end()) {
        if (found->second.generation == 1 && found->second.digest == digest &&
                found->second.parent_digest.empty()) {
            return {found->second, true, revision_, file_bytes_};
        }
        throw server_snapshot_manifest_error("logical head already exists", true);
    }
    if (logical_head_tombstones_.find(name) != logical_head_tombstones_.end()) {
        throw server_snapshot_manifest_error(
                "logical head name is retired and cannot be recreated", true);
    }
    if (revision_ == std::numeric_limits<uint64_t>::max()) {
        throw server_snapshot_manifest_error("snapshot manifest revision space is exhausted");
    }
    const uint64_t next_revision = revision_ + 1;
    const server_snapshot_logical_head head {name, digest, {}, 1, next_revision};
    const auto record = encode_head_mutation(
            record_type::create_head, next_revision, name, digest, {}, head.generation);
    try {
        append_record(record);
    } catch (const server_snapshot_manifest_error & error) {
        if (!error.unavailable) {
            throw;
        }
        logical_heads_[name] = head;
        std::vector<uint8_t> checkpoint;
        try {
            checkpoint = encode_checkpoint(next_revision);
        } catch (...) {
            logical_heads_.erase(name);
            throw;
        }
        logical_heads_.erase(name);
        write_compacted(checkpoint);
        compactions_++;
    }
    logical_heads_[name] = head;
    revision_ = next_revision;
    return {head, false, revision_, file_bytes_};
}

server_snapshot_logical_head_result server_snapshot_manifest::advance_head(
        const std::string & name,
        uint64_t expected_generation,
        const std::string & expected_digest,
        const std::string & digest) {
    validate_head_name(name);
    validate_digest(expected_digest);
    validate_digest(digest);
    const auto found = logical_heads_.find(name);
    if (found == logical_heads_.end()) {
        throw server_snapshot_manifest_error("logical head is unavailable", true);
    }
    if (expected_generation < std::numeric_limits<uint64_t>::max() &&
            found->second.generation == expected_generation + 1 &&
            found->second.parent_digest == expected_digest && found->second.digest == digest) {
        return {found->second, true, revision_, file_bytes_};
    }
    if (found->second.generation != expected_generation || found->second.digest != expected_digest) {
        throw server_snapshot_manifest_error("logical head compare-and-swap conflict", true);
    }
    if (std::any_of(publish_advance_intents_.begin(), publish_advance_intents_.end(),
            [&](const auto & item) { return item.second.head_name == name; })) {
        throw server_snapshot_manifest_error("logical head transaction is pending", true);
    }
    if (revision_ == std::numeric_limits<uint64_t>::max() ||
            found->second.generation == std::numeric_limits<uint64_t>::max()) {
        throw server_snapshot_manifest_error("logical head generation space is exhausted");
    }
    const uint64_t next_revision = revision_ + 1;
    const server_snapshot_logical_head previous = found->second;
    const server_snapshot_logical_head head {
        name, digest, expected_digest, expected_generation + 1, next_revision,
    };
    const auto record = encode_head_mutation(
            record_type::advance_head,
            next_revision,
            name,
            digest,
            expected_digest,
            head.generation);
    try {
        append_record(record);
    } catch (const server_snapshot_manifest_error & error) {
        if (!error.unavailable) {
            throw;
        }
        found->second = head;
        std::vector<uint8_t> checkpoint;
        try {
            checkpoint = encode_checkpoint(next_revision);
        } catch (...) {
            found->second = previous;
            throw;
        }
        found->second = previous;
        write_compacted(checkpoint);
        compactions_++;
    }
    found->second = head;
    revision_ = next_revision;
    return {head, false, revision_, file_bytes_};
}

server_snapshot_logical_head_result server_snapshot_manifest::delete_head(
        const std::string & name,
        uint64_t expected_generation,
        const std::string & expected_digest) {
    validate_head_name(name);
    validate_digest(expected_digest);
    const auto found = logical_heads_.find(name);
    if (found == logical_heads_.end()) {
        const auto tombstone = logical_head_tombstones_.find(name);
        if (tombstone != logical_head_tombstones_.end() &&
                tombstone->second.head.generation == expected_generation &&
                tombstone->second.head.digest == expected_digest) {
            return {tombstone->second.head, true, revision_, file_bytes_};
        }
    }
    if (found == logical_heads_.end() || found->second.generation != expected_generation ||
            found->second.digest != expected_digest) {
        throw server_snapshot_manifest_error("logical head compare-and-swap conflict", true);
    }
    if (std::any_of(publish_advance_intents_.begin(), publish_advance_intents_.end(),
            [&](const auto & item) { return item.second.head_name == name; })) {
        throw server_snapshot_manifest_error("logical head transaction is pending", true);
    }
    if (revision_ == std::numeric_limits<uint64_t>::max()) {
        throw server_snapshot_manifest_error("snapshot manifest revision space is exhausted");
    }
    const uint64_t next_revision = revision_ + 1;
    const server_snapshot_logical_head removed = found->second;
    const auto record = encode_head_mutation(
            record_type::delete_head,
            next_revision,
            name,
            expected_digest,
            {},
            expected_generation);
    try {
        append_record(record);
    } catch (const server_snapshot_manifest_error & error) {
        if (!error.unavailable) {
            throw;
        }
        logical_heads_.erase(found);
        logical_head_tombstones_[name] = {removed, next_revision};
        std::vector<uint8_t> checkpoint;
        try {
            checkpoint = encode_checkpoint(next_revision);
        } catch (...) {
            logical_head_tombstones_.erase(name);
            logical_heads_[name] = removed;
            throw;
        }
        logical_head_tombstones_.erase(name);
        logical_heads_[name] = removed;
        write_compacted(checkpoint);
        compactions_++;
    }
    logical_head_tombstones_[name] = {removed, next_revision};
    logical_heads_.erase(name);
    revision_ = next_revision;
    return {removed, false, revision_, file_bytes_};
}

server_snapshot_manifest_result server_snapshot_manifest::release(
        const std::string & owner,
        const std::string & digest) {
    validate_owner(owner);
    validate_digest(digest);
    const std::string key = ref_key(owner, digest);
    const auto found = refs_.find(key);
    if (found == refs_.end()) {
        throw server_snapshot_manifest_error("snapshot ownership reference is unavailable", true);
    }
    if (revision_ == std::numeric_limits<uint64_t>::max()) {
        throw server_snapshot_manifest_error("snapshot manifest revision space is exhausted");
    }
    const server_snapshot_manifest_ref ref = found->second;
    const uint64_t next_revision = revision_ + 1;
    const auto record = encode_mutation(record_type::release, next_revision, owner, digest, "");
    append_record(record);
    refs_.erase(found);
    revision_ = next_revision;
    return {ref, false, revision_, file_bytes_};
}

server_snapshot_manifest_compact_result server_snapshot_manifest::compact() {
    const uint64_t records_before = record_count_;
    const uint64_t bytes_before = file_bytes_;
    std::vector<uint8_t> checkpoint;
    if (revision_ > 0) {
        checkpoint = encode_checkpoint(revision_);
    }
    write_compacted(checkpoint);
    compactions_++;
    return {
        revision_, refs_.size(), records_before, record_count_, bytes_before, file_bytes_,
    };
}

void server_snapshot_manifest::sync_namespace_directory() const {
#if !defined(_WIN32)
    const int fd = ::open(namespace_path_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) {
        throw server_snapshot_manifest_error("failed to open snapshot manifest namespace for sync");
    }
    const int result = ::fsync(fd);
    const int saved_errno = errno;
    ::close(fd);
    if (result != 0) {
        throw server_snapshot_manifest_error(
                std::string("failed to sync snapshot manifest namespace: ") + std::strerror(saved_errno));
    }
#endif
}
