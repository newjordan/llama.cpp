#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace jspace {

struct suffix_pool_row {
    size_t             horizon = 0;
    std::vector<float> values;
    double             l2  = 0.0;
    double             rms = 0.0;
};

inline std::vector<size_t> fibonacci_horizons(size_t depth) {
    std::vector<size_t> result;
    result.reserve(depth);
    if (depth == 0) {
        return result;
    }

    result.push_back(1);
    if (depth == 1) {
        return result;
    }

    result.push_back(2);
    while (result.size() < depth) {
        const size_t previous = result[result.size() - 1];
        const size_t before_previous = result[result.size() - 2];
        if (previous > std::numeric_limits<size_t>::max() - before_previous) {
            throw std::overflow_error("Fibonacci pooling horizon overflow");
        }
        result.push_back(previous + before_previous);
    }
    return result;
}

inline std::vector<size_t> fibonacci_horizons_up_to(size_t max_tokens) {
    if (max_tokens == 0) {
        return {};
    }
    std::vector<size_t> result = { 1 };
    if (max_tokens == 1) {
        return result;
    }
    result.push_back(2);
    while (true) {
        const size_t previous = result[result.size() - 1];
        const size_t before_previous = result[result.size() - 2];
        if (previous > std::numeric_limits<size_t>::max() - before_previous) {
            break;
        }
        const size_t next = previous + before_previous;
        if (next > max_tokens) {
            break;
        }
        result.push_back(next);
    }
    return result;
}

// Apply a sparse row-stochastic matrix whose kth row uniformly averages the
// requested suffix horizon. Building one running suffix sum evaluates every
// active row in O(max(horizon) * width), rather than separately scanning each
// overlapping window. Keeping the evaluator schedule-agnostic lets Fibonacci,
// logarithmic, and linear controls share exactly the same arithmetic.
inline std::vector<suffix_pool_row> suffix_boxcar_pool(
        const std::vector<std::vector<float>> & chronological_columns,
        const std::vector<size_t> &              requested_horizons) {
    if (chronological_columns.empty() || requested_horizons.empty()) {
        return {};
    }

    const size_t width = chronological_columns.front().size();
    if (width == 0) {
        throw std::invalid_argument("suffix pooling columns must be non-empty");
    }
    for (const auto & column : chronological_columns) {
        if (column.size() != width) {
            throw std::invalid_argument("suffix pooling column widths differ");
        }
        for (float value : column) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("suffix pooling input contains a non-finite value");
            }
        }
    }

    size_t previous_horizon = 0;
    for (size_t horizon : requested_horizons) {
        if (horizon == 0 || horizon <= previous_horizon) {
            throw std::invalid_argument(
                    "suffix pooling horizons must be positive and strictly increasing");
        }
        previous_horizon = horizon;
    }

    std::vector<size_t> active;
    active.reserve(requested_horizons.size());
    for (size_t horizon : requested_horizons) {
        if (horizon > chronological_columns.size()) {
            break;
        }
        active.push_back(horizon);
    }
    if (active.empty()) {
        return {};
    }

    std::vector<double> suffix_sum(width, 0.0);
    std::vector<suffix_pool_row> result;
    result.reserve(active.size());
    size_t next_horizon = 0;
    for (size_t count = 1; count <= active.back(); ++count) {
        const auto & column = chronological_columns[chronological_columns.size() - count];
        for (size_t coordinate = 0; coordinate < width; ++coordinate) {
            suffix_sum[coordinate] += column[coordinate];
        }
        if (count != active[next_horizon]) {
            continue;
        }

        suffix_pool_row row;
        row.horizon = count;
        row.values.resize(width);
        double sum_squares = 0.0;
        for (size_t coordinate = 0; coordinate < width; ++coordinate) {
            const double mean = suffix_sum[coordinate] / static_cast<double>(count);
            row.values[coordinate] = static_cast<float>(mean);
            const double serialized_mean = row.values[coordinate];
            sum_squares += serialized_mean * serialized_mean;
        }
        row.l2 = std::sqrt(sum_squares);
        row.rms = std::sqrt(sum_squares / static_cast<double>(width));
        result.push_back(std::move(row));
        ++next_horizon;
        if (next_horizon == active.size()) {
            break;
        }
    }
    return result;
}

// The unnormalized support masks at Fibonacci horizons obey
// M_k(t) = M_(k-1)(t) + shift_(F_(k-1)) M_(k-2)(t).
inline std::vector<suffix_pool_row> fibonacci_suffix_pool(
        const std::vector<std::vector<float>> & chronological_columns,
        size_t                                  depth) {
    return suffix_boxcar_pool(chronological_columns, fibonacci_horizons(depth));
}

} // namespace jspace
