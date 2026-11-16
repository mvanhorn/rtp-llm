#pragma once

#include <vector>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace rtp_llm {

// Adaptive target verification is keyed by total packed rows, not B * ceil(R/B).
// Since every request contributes at least one row, min(R, concurrency) metadata
// slots cover every legal batch for this row count without padding token tensors.
struct AdaptiveVerifyGraphShape {
    int rows;
    int batch_capacity;
};

inline int adaptiveVerifyMaxRows(int concurrency, int tokens_per_request) {
    if (concurrency <= 0 || tokens_per_request <= 1
        || concurrency > std::numeric_limits<int>::max() / tokens_per_request) {
        throw std::invalid_argument("invalid adaptive verification capture limits");
    }
    return concurrency * tokens_per_request;
}

inline AdaptiveVerifyGraphShape adaptiveVerifyGraphShape(int concurrency, int tokens_per_request, int rows) {
    const int max_rows = adaptiveVerifyMaxRows(concurrency, tokens_per_request);
    if (rows <= 0 || rows > max_rows) {
        throw std::invalid_argument("adaptive verification rows exceed capture limits");
    }
    return {rows, std::min(rows, concurrency)};
}

inline AdaptiveVerifyGraphShape
adaptiveVerifyReplayShape(int concurrency, int tokens_per_request, int batch, int rows) {
    auto shape = adaptiveVerifyGraphShape(concurrency, tokens_per_request, rows);
    if (batch <= 0 || batch > concurrency || rows < batch
        || static_cast<int64_t>(rows) > static_cast<int64_t>(batch) * tokens_per_request) {
        throw std::invalid_argument("invalid adaptive verification replay dimensions");
    }
    return shape;
}

inline std::vector<int> adaptiveVerifyCaptureRows(int concurrency, int tokens_per_request) {
    const int        max_rows = adaptiveVerifyMaxRows(concurrency, tokens_per_request);
    std::vector<int> rows;
    rows.reserve(max_rows);
    for (int64_t row = 1; row <= max_rows; ++row) {
        rows.push_back(static_cast<int>(row));
    }
    return rows;
}

// Valid ragged capture metadata: every slot is nonempty, each length is within
// the configured verify limit, and the sum is exactly the graph's row count.
inline std::vector<int> adaptiveVerifyCaptureLengths(const AdaptiveVerifyGraphShape& shape) {
    std::vector<int> lengths(shape.batch_capacity, shape.rows / shape.batch_capacity);
    for (int i = 0; i < shape.rows % shape.batch_capacity; ++i) {
        ++lengths[i];
    }
    return lengths;
}

inline std::vector<int> draftPrefillCaptureSeqLens(const std::vector<int>& decode_batch_sizes, int num_tokens_per_bs) {
    std::vector<int> seq_lens;
    seq_lens.reserve(decode_batch_sizes.size());
    for (int batch_size : decode_batch_sizes) {
        seq_lens.push_back(batch_size * num_tokens_per_bs);
    }
    return seq_lens;
}

}  // namespace rtp_llm
