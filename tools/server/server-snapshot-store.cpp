#include "server-snapshot-store.h"

extern "C" {
#include "sha256.h"
}

#include "common.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <system_error>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace {

constexpr std::array<uint8_t, 8> SNAPSHOT_MAGIC = {'T', 'S', 'T', 'C', 'O', 'L', 'D', '1'};
constexpr uint32_t SNAPSHOT_VERSION = 1;
constexpr uint64_t SNAPSHOT_FIXED_HEADER_BYTES = 40;
constexpr size_t SNAPSHOT_MAX_COMPATIBILITY_BYTES = 256;

void append_u32(std::vector<uint8_t> & out, uint32_t value) {
    for (size_t i = 0; i < 4; ++i) {
        out.push_back((uint8_t) (value >> (8 * i)));
    }
}

void append_u64(std::vector<uint8_t> & out, uint64_t value) {
    for (size_t i = 0; i < 8; ++i) {
        out.push_back((uint8_t) (value >> (8 * i)));
    }
}

uint32_t read_u32(std::istream & input) {
    uint8_t bytes[4];
    input.read((char *) bytes, sizeof(bytes));
    if (!input) {
        throw server_snapshot_store_error("truncated snapshot content header");
    }
    uint32_t value = 0;
    for (size_t i = 0; i < sizeof(bytes); ++i) {
        value |= (uint32_t) bytes[i] << (8 * i);
    }
    return value;
}

uint64_t read_u64(std::istream & input) {
    uint8_t bytes[8];
    input.read((char *) bytes, sizeof(bytes));
    if (!input) {
        throw server_snapshot_store_error("truncated snapshot content header");
    }
    uint64_t value = 0;
    for (size_t i = 0; i < sizeof(bytes); ++i) {
        value |= (uint64_t) bytes[i] << (8 * i);
    }
    return value;
}

std::string read_string(std::istream & input, size_t size) {
    std::string value(size, '\0');
    input.read(value.data(), size);
    if (!input) {
        throw server_snapshot_store_error("truncated snapshot content metadata");
    }
    return value;
}

#if !defined(_WIN32)
void write_all(int fd, const void * data, size_t size) {
    const uint8_t * cursor = (const uint8_t *) data;
    while (size > 0) {
        const ssize_t written = ::write(fd, cursor, size);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw server_snapshot_store_error(std::string("snapshot content write failed: ") + std::strerror(errno));
        }
        if (written == 0) {
            throw server_snapshot_store_error("snapshot content write made no progress");
        }
        cursor += written;
        size -= (size_t) written;
    }
}
#endif

} // namespace

server_snapshot_store::server_snapshot_store(
        std::string base_path,
        std::string compatibility_id,
        uint64_t max_disk_bytes,
        digest_fn compute_digest)
    : base_path_(std::move(base_path)),
      compatibility_id_(std::move(compatibility_id)),
      max_disk_bytes_(max_disk_bytes),
      compute_digest_(std::move(compute_digest)) {
    if (!fs_is_directory(base_path_)) {
        throw server_snapshot_store_error("snapshot store base path is not a directory");
    }
    if (compatibility_id_.empty() || compatibility_id_.size() > SNAPSHOT_MAX_COMPATIBILITY_BYTES) {
        throw server_snapshot_store_error("snapshot compatibility ID must contain 1 to 256 bytes");
    }
    if (max_disk_bytes_ == 0) {
        throw server_snapshot_store_error("snapshot disk budget must be positive");
    }
    if (!compute_digest_) {
        throw server_snapshot_store_error("snapshot content digest function is missing");
    }

    namespace_id_ = compatibility_namespace(compatibility_id_);
    namespace_path_ = (std::filesystem::path(base_path_) / namespace_id_).string();
    std::error_code namespace_error;
    std::filesystem::create_directories(namespace_path_, namespace_error);
    if (namespace_error || !std::filesystem::is_directory(namespace_path_, namespace_error) || namespace_error) {
        throw server_snapshot_store_error("failed to create snapshot compatibility namespace");
    }
    discover();
    disk_high_water_bytes_ = disk_bytes_;
}

std::map<std::string, server_snapshot_store_entry> server_snapshot_store::entries() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_;
}

bool server_snapshot_store::find_entry(
        const std::string & digest,
        server_snapshot_store_entry & entry) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = entries_.find(digest);
    if (found == entries_.end()) {
        return false;
    }
    entry = found->second;
    return true;
}

uint64_t server_snapshot_store::projected_file_bytes(uint64_t n_tokens, uint64_t state_bytes) const {
    return object_file_bytes(compatibility_id_.size(), 71, n_tokens, state_bytes);
}

const std::string & server_snapshot_store::compatibility_id() const {
    return compatibility_id_;
}

const std::string & server_snapshot_store::namespace_id() const {
    return namespace_id_;
}

const std::string & server_snapshot_store::namespace_path() const {
    return namespace_path_;
}

uint64_t server_snapshot_store::disk_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return disk_bytes_;
}
uint64_t server_snapshot_store::disk_budget_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return max_disk_bytes_;
}
uint64_t server_snapshot_store::disk_high_water_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return disk_high_water_bytes_;
}
uint64_t server_snapshot_store::recovered_temp_files() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return recovered_temp_files_;
}
uint64_t server_snapshot_store::ignored_corrupt_files() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ignored_corrupt_files_;
}
uint64_t server_snapshot_store::runtime_integrity_failures() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return runtime_integrity_failures_;
}
uint64_t server_snapshot_store::orphaned_disk_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return orphaned_disk_bytes_;
}

bool server_snapshot_store::valid_digest(const std::string & digest) {
    if (digest.size() != 71 || digest.compare(0, 7, "sha256:") != 0) {
        return false;
    }
    return std::all_of(digest.begin() + 7, digest.end(), [](unsigned char value) {
        return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
    });
}

std::string server_snapshot_store::digest_hex(const std::string & digest) {
    if (!valid_digest(digest)) {
        throw server_snapshot_store_error("invalid snapshot content digest");
    }
    return digest.substr(7);
}

std::string server_snapshot_store::compatibility_namespace(const std::string & compatibility_id) {
    static constexpr unsigned char domain[] = "turbo-statetree-cold-namespace-v1\0";
    sha256_t hash;
    sha256_init(&hash);
    sha256_update(&hash, domain, sizeof(domain));
    sha256_update(&hash, (const unsigned char *) compatibility_id.data(), compatibility_id.size());
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_final(&hash, digest);
    std::ostringstream encoded;
    encoded << std::hex << std::setfill('0');
    for (const unsigned char byte : digest) {
        encoded << std::setw(2) << (unsigned int) byte;
    }
    return encoded.str();
}

uint64_t server_snapshot_store::object_file_bytes(
        size_t compatibility_size,
        size_t digest_size,
        uint64_t n_tokens,
        uint64_t state_bytes) {
    uint64_t total = SNAPSHOT_FIXED_HEADER_BYTES;
    const auto add = [&](uint64_t value) {
        if (value > std::numeric_limits<uint64_t>::max() - total) {
            throw server_snapshot_store_error("snapshot content object size overflow");
        }
        total += value;
    };
    add(compatibility_size);
    add(digest_size);
    if (n_tokens > std::numeric_limits<uint64_t>::max() / sizeof(int32_t)) {
        throw server_snapshot_store_error("snapshot content object size overflow");
    }
    add(n_tokens * sizeof(int32_t));
    add(state_bytes);
    return total;
}

server_snapshot_store_entry server_snapshot_store::inspect_file(
        const std::string & path,
        const std::string & expected_digest) const {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw server_snapshot_store_error("failed to open snapshot content object");
    }
    std::array<uint8_t, 8> magic {};
    input.read((char *) magic.data(), magic.size());
    if (!input || magic != SNAPSHOT_MAGIC) {
        throw server_snapshot_store_error("invalid snapshot content magic");
    }
    if (read_u32(input) != SNAPSHOT_VERSION) {
        throw server_snapshot_store_error("unsupported snapshot content version");
    }
    const uint32_t compatibility_size = read_u32(input);
    const uint32_t digest_size = read_u32(input);
    const uint32_t reserved = read_u32(input);
    const uint64_t n_tokens = read_u64(input);
    const uint64_t state_bytes = read_u64(input);
    if (reserved != 0 || compatibility_size == 0 ||
            compatibility_size > SNAPSHOT_MAX_COMPATIBILITY_BYTES || digest_size != 71) {
        throw server_snapshot_store_error("invalid snapshot content header fields");
    }
    const std::string compatibility_id = read_string(input, compatibility_size);
    const std::string digest = read_string(input, digest_size);
    if (compatibility_id != compatibility_id_ || digest != expected_digest || !valid_digest(digest)) {
        throw server_snapshot_store_error("snapshot content identity is incompatible");
    }
    const uint64_t expected_file_bytes = object_file_bytes(
            compatibility_size, digest_size, n_tokens, state_bytes);
    std::error_code error;
    const uint64_t actual_file_bytes = std::filesystem::file_size(path, error);
    if (error || actual_file_bytes != expected_file_bytes) {
        throw server_snapshot_store_error("snapshot content file size is inconsistent");
    }
    return {
        digest,
        path,
        n_tokens,
        state_bytes,
        n_tokens * sizeof(llama_token) + state_bytes,
        actual_file_bytes,
    };
}

void server_snapshot_store::discover() {
    std::error_code error;
    for (const auto & item : std::filesystem::directory_iterator(namespace_path_, error)) {
        if (error) {
            throw server_snapshot_store_error("failed to scan snapshot compatibility namespace");
        }
        if (!item.is_regular_file(error) || error) {
            error.clear();
            continue;
        }
        const std::string filename = item.path().filename().string();
        if (filename.rfind(".tmp-", 0) == 0) {
            if (!std::filesystem::remove(item.path(), error) || error) {
                throw server_snapshot_store_error("failed to remove interrupted snapshot temp object");
            }
            recovered_temp_files_++;
            continue;
        }
        if (filename.size() < 4 || filename.compare(filename.size() - 4, 4, ".tss") != 0) {
            continue;
        }
        if (filename.size() != 68) {
            ignored_corrupt_files_++;
            error.clear();
            const uint64_t corrupt_bytes = std::filesystem::file_size(item.path(), error);
            if (error || corrupt_bytes > std::numeric_limits<uint64_t>::max() - disk_bytes_) {
                throw server_snapshot_store_error("failed to account malformed snapshot content filename");
            }
            orphaned_disk_bytes_ += corrupt_bytes;
            disk_bytes_ += corrupt_bytes;
            continue;
        }
        const std::string digest = "sha256:" + filename.substr(0, 64);
        try {
            auto entry = inspect_file(item.path().string(), digest);
            const auto inserted = entries_.emplace(digest, std::move(entry));
            if (!inserted.second || disk_bytes_ > std::numeric_limits<uint64_t>::max() - inserted.first->second.file_bytes) {
                throw server_snapshot_store_error("duplicate or overflowing snapshot content index");
            }
            disk_bytes_ += inserted.first->second.file_bytes;
        } catch (const server_snapshot_store_error &) {
            ignored_corrupt_files_++;
            error.clear();
            const uint64_t corrupt_bytes = std::filesystem::file_size(item.path(), error);
            if (error || corrupt_bytes > std::numeric_limits<uint64_t>::max() - disk_bytes_) {
                throw server_snapshot_store_error("failed to account corrupt snapshot content object");
            }
            orphaned_disk_bytes_ += corrupt_bytes;
            disk_bytes_ += corrupt_bytes;
        }
    }
    if (error) {
        throw server_snapshot_store_error("failed to scan snapshot compatibility namespace");
    }
    if (recovered_temp_files_ > 0) {
        sync_namespace_directory();
    }
}

server_snapshot_store_spill_result server_snapshot_store::spill(
        const std::string & digest,
        const std::vector<llama_token> & tokens,
        const std::vector<uint8_t> & state) {
    std::lock_guard<std::mutex> lock(mutex_);
    const std::string hex = digest_hex(digest);
    const auto existing = entries_.find(digest);
    if (existing != entries_.end()) {
        const auto payload = load_unlocked(digest, std::numeric_limits<uint64_t>::max());
        if (payload.tokens != tokens || payload.state != state) {
            throw server_snapshot_store_error("snapshot content digest collision in durable store");
        }
        return {existing->second, true};
    }

    const uint64_t file_bytes = object_file_bytes(
            compatibility_id_.size(), digest.size(), tokens.size(), state.size());
    if (file_bytes > max_disk_bytes_ || disk_bytes_ > max_disk_bytes_ - file_bytes) {
        throw server_snapshot_store_error("snapshot content disk budget is exhausted", true);
    }

    std::vector<uint8_t> header;
    header.reserve(SNAPSHOT_FIXED_HEADER_BYTES + compatibility_id_.size() + digest.size());
    header.insert(header.end(), SNAPSHOT_MAGIC.begin(), SNAPSHOT_MAGIC.end());
    append_u32(header, SNAPSHOT_VERSION);
    append_u32(header, (uint32_t) compatibility_id_.size());
    append_u32(header, (uint32_t) digest.size());
    append_u32(header, 0);
    append_u64(header, tokens.size());
    append_u64(header, state.size());
    header.insert(header.end(), compatibility_id_.begin(), compatibility_id_.end());
    header.insert(header.end(), digest.begin(), digest.end());

    const std::filesystem::path final_path = std::filesystem::path(namespace_path_) / (hex + ".tss");
    const std::filesystem::path temp_path = std::filesystem::path(namespace_path_) /
        (".tmp-" + std::to_string(++temp_sequence_) + "-" + hex + ".tss");

#if !defined(_WIN32)
    int fd = ::open(temp_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, S_IRUSR | S_IWUSR);
    if (fd < 0) {
        throw server_snapshot_store_error(std::string("failed to create snapshot temp object: ") + std::strerror(errno));
    }
    bool complete = false;
    bool published = false;
    try {
        write_all(fd, header.data(), header.size());
        std::array<uint8_t, 4096> encoded_tokens {};
        size_t offset = 0;
        while (offset < tokens.size()) {
            const size_t count = std::min(tokens.size() - offset, encoded_tokens.size() / sizeof(int32_t));
            for (size_t i = 0; i < count; ++i) {
                const uint32_t value = (uint32_t) tokens[offset + i];
                for (size_t byte = 0; byte < sizeof(int32_t); ++byte) {
                    encoded_tokens[i * sizeof(int32_t) + byte] = (uint8_t) (value >> (8 * byte));
                }
            }
            write_all(fd, encoded_tokens.data(), count * sizeof(int32_t));
            offset += count;
        }
        if (!state.empty()) {
            write_all(fd, state.data(), state.size());
        }
        if (::fsync(fd) != 0) {
            throw server_snapshot_store_error(std::string("failed to sync snapshot temp object: ") + std::strerror(errno));
        }
        if (::close(fd) != 0) {
            fd = -1;
            throw server_snapshot_store_error(std::string("failed to close snapshot temp object: ") + std::strerror(errno));
        }
        fd = -1;
        if (::rename(temp_path.c_str(), final_path.c_str()) != 0) {
            throw server_snapshot_store_error(std::string("failed to publish snapshot content object: ") + std::strerror(errno));
        }
        published = true;
        sync_namespace_directory();
        complete = true;
    } catch (...) {
        if (fd >= 0) {
            ::close(fd);
        }
        std::error_code cleanup_error;
        std::filesystem::remove(temp_path, cleanup_error);
        if (published && !complete) {
            cleanup_error.clear();
            std::filesystem::remove(final_path, cleanup_error);
        }
        if (!complete) {
            throw;
        }
    }
#else
    {
        std::ofstream output(temp_path, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw server_snapshot_store_error("failed to create snapshot temp object");
        }
        output.write((const char *) header.data(), header.size());
        for (const llama_token token : tokens) {
            const uint32_t value = (uint32_t) token;
            uint8_t bytes[4];
            for (size_t i = 0; i < sizeof(bytes); ++i) bytes[i] = (uint8_t) (value >> (8 * i));
            output.write((const char *) bytes, sizeof(bytes));
        }
        output.write((const char *) state.data(), state.size());
        output.flush();
        if (!output) {
            throw server_snapshot_store_error("failed to write snapshot temp object");
        }
    }
    std::error_code rename_error;
    std::filesystem::rename(temp_path, final_path, rename_error);
    if (rename_error) {
        std::filesystem::remove(temp_path, rename_error);
        throw server_snapshot_store_error("failed to publish snapshot content object");
    }
#endif

    auto entry = inspect_file(final_path.string(), digest);
    disk_bytes_ += entry.file_bytes;
    const auto inserted = entries_.emplace(digest, entry);
    if (!inserted.second) {
        throw server_snapshot_store_error("snapshot content index collision after publish");
    }
    try {
        const auto verified_payload = load_unlocked(digest, std::numeric_limits<uint64_t>::max());
        if (verified_payload.tokens != tokens || verified_payload.state != state) {
            throw server_snapshot_store_error("snapshot content read-back mismatch", true);
        }
    } catch (...) {
        entries_.erase(digest);
        std::error_code cleanup_error;
        const bool removed = std::filesystem::remove(final_path, cleanup_error);
        if (removed && !cleanup_error) {
            disk_bytes_ -= entry.file_bytes;
        } else {
            orphaned_disk_bytes_ += entry.file_bytes;
        }
        try {
            sync_namespace_directory();
        } catch (...) {
        }
        throw;
    }
    disk_high_water_bytes_ = std::max(disk_high_water_bytes_, disk_bytes_);
    return {entry, false};
}

server_snapshot_store_payload server_snapshot_store::load(
        const std::string & digest,
        uint64_t max_payload_bytes) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return load_unlocked(digest, max_payload_bytes);
}

server_snapshot_store_payload server_snapshot_store::load_unlocked(
        const std::string & digest,
        uint64_t max_payload_bytes) const {
    const auto found = entries_.find(digest);
    if (found == entries_.end()) {
        throw server_snapshot_store_error("durable snapshot content is unavailable", true);
    }
    if (found->second.payload_bytes > max_payload_bytes) {
        throw server_snapshot_store_error("durable snapshot content exceeds the cold-load byte ceiling", true);
    }
    try {
        const auto verified = inspect_file(found->second.path, digest);
        // Recheck the live envelope: the object may have been replaced after
        // discovery, so indexed metadata alone is not an allocation fence.
        if (verified.payload_bytes > max_payload_bytes) {
            throw server_snapshot_store_error(
                    "durable snapshot content exceeds the cold-load byte ceiling", true);
        }
        std::ifstream input(verified.path, std::ios::binary);
        input.seekg((std::streamoff) (SNAPSHOT_FIXED_HEADER_BYTES + compatibility_id_.size() + digest.size()));
        if (!input) {
            throw server_snapshot_store_error("failed to seek snapshot content payload");
        }
        if (verified.n_tokens > std::numeric_limits<size_t>::max() ||
                verified.state_bytes > std::numeric_limits<size_t>::max()) {
            throw server_snapshot_store_error("snapshot content payload exceeds address space");
        }
        server_snapshot_store_payload payload;
        payload.entry = verified;
        payload.tokens.resize((size_t) verified.n_tokens);
        for (size_t i = 0; i < payload.tokens.size(); ++i) {
            payload.tokens[i] = (llama_token) read_u32(input);
        }
        payload.state.resize((size_t) verified.state_bytes);
        input.read((char *) payload.state.data(), payload.state.size());
        if (!input) {
            throw server_snapshot_store_error("truncated snapshot content payload");
        }
        if (compute_digest_(payload.tokens, payload.state) != digest) {
            throw server_snapshot_store_error("snapshot content integrity check failed", true);
        }
        return payload;
    } catch (const server_snapshot_store_error & error) {
        runtime_integrity_failures_++;
        throw server_snapshot_store_error(error.what(), true);
    } catch (const std::exception & error) {
        runtime_integrity_failures_++;
        throw server_snapshot_store_error(
                std::string("failed to load durable snapshot content: ") + error.what(), true);
    }
}

server_snapshot_store_entry server_snapshot_store::erase(const std::string & digest) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = entries_.find(digest);
    if (found == entries_.end()) {
        throw server_snapshot_store_error("durable snapshot content is unavailable", true);
    }
    const auto entry = found->second;
    std::error_code error;
    if (!std::filesystem::remove(entry.path, error) || error) {
        throw server_snapshot_store_error("failed to erase durable snapshot content");
    }
    entries_.erase(found);
    if (disk_bytes_ < entry.file_bytes) {
        throw server_snapshot_store_error("snapshot content disk accounting underflow");
    }
    disk_bytes_ -= entry.file_bytes;
    sync_namespace_directory();
    return entry;
}

void server_snapshot_store::sync_namespace_directory() const {
#if !defined(_WIN32)
    const int fd = ::open(namespace_path_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) {
        throw server_snapshot_store_error(std::string("failed to open snapshot namespace for sync: ") + std::strerror(errno));
    }
    const int result = ::fsync(fd);
    const int saved_errno = errno;
    ::close(fd);
    if (result != 0) {
        throw server_snapshot_store_error(std::string("failed to sync snapshot namespace: ") + std::strerror(saved_errno));
    }
#endif
}
