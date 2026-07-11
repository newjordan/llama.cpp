#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" {
#include "sha256.h"
}

static bool digest_matches(const unsigned char * digest, const char * expected) {
    char actual[SHA256_DIGEST_SIZE * 2 + 1] = {};
    for (size_t i = 0; i < SHA256_DIGEST_SIZE; ++i) {
        std::snprintf(actual + 2 * i, 3, "%02x", digest[i]);
    }
    return std::strcmp(actual, expected) == 0;
}

static bool check_digest(const unsigned char * data, size_t size, size_t chunk_size, const char * expected) {
    sha256_t hash;
    sha256_init(&hash);

    for (size_t offset = 0; offset < size;) {
        const size_t chunk = std::min(chunk_size, size - offset);
        sha256_update(&hash, data + offset, chunk);
        offset += chunk;
    }

    std::array<unsigned char, SHA256_DIGEST_SIZE> digest {};
    sha256_final(&hash, digest.data());
    return digest_matches(digest.data(), expected);
}

int main() {
    constexpr const char * empty_digest =
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    constexpr const char * abc_digest =
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    constexpr const char * million_a_digest =
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0";

    std::array<unsigned char, SHA256_DIGEST_SIZE> digest {};
    sha256_hash(digest.data(), nullptr, 0);
    if (!digest_matches(digest.data(), empty_digest)) {
        return 1;
    }

    const auto * abc = reinterpret_cast<const unsigned char *>("abc");
    sha256_hash(digest.data(), abc, 3);
    if (!digest_matches(digest.data(), abc_digest)) {
        return 1;
    }

    std::vector<unsigned char> million_a(1'000'001, 'a');
    const unsigned char * input = million_a.data() + 1;
    for (const size_t chunk_size : { size_t(1), size_t(7), size_t(63), size_t(64), size_t(65), size_t(511), size_t(4096) }) {
        if (!check_digest(input, 1'000'000, chunk_size, million_a_digest)) {
            return 1;
        }
    }

    return 0;
}
