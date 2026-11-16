#include "gtest/gtest.h"

#include "rtp_llm/cpp/cuda_graph/cuda_graph_capture_range.h"

namespace rtp_llm {
namespace {

TEST(CudaGraphCaptureRangeTest, DraftPrefillUsesDecodeBatchBuckets) {
    const std::vector<int> decode_batch_sizes{1, 4, 8, 16, 24, 32, 64};
    const std::vector<int> expected_seq_lens{4, 16, 32, 64, 96, 128, 256};

    EXPECT_EQ(draftPrefillCaptureSeqLens(decode_batch_sizes, 4), expected_seq_lens);
}

TEST(CudaGraphCaptureRangeTest, EmptyDecodeRangeStaysEmpty) {
    EXPECT_TRUE(draftPrefillCaptureSeqLens({}, 4).empty());
}

TEST(CudaGraphCaptureRangeTest, AdaptiveCapturesEveryRowWithoutPadding) {
    for (int concurrency : {1, 2, 7, 48}) {
        for (int width : {2, 3, 6}) {
            const auto rows = adaptiveVerifyCaptureRows(concurrency, width);
            ASSERT_EQ(rows.size(), concurrency * width);
            for (int r = 1; r <= concurrency * width; ++r) {
                EXPECT_EQ(rows[r - 1], r);
                auto shape = adaptiveVerifyGraphShape(concurrency, width, r);
                EXPECT_EQ(shape.rows, r);
                EXPECT_EQ(shape.batch_capacity, std::min(r, concurrency));
                auto lengths = adaptiveVerifyCaptureLengths(shape);
                int  sum     = 0;
                for (int length : lengths) {
                    EXPECT_GE(length, 1);
                    EXPECT_LE(length, width);
                    sum += length;
                }
                EXPECT_EQ(sum, r);
            }
            // Every feasible combination, including cap=0 and heterogeneous caps.
            for (int b = 1; b <= concurrency; ++b) {
                for (int r = b; r <= b * width; ++r) {
                    auto shape = adaptiveVerifyReplayShape(concurrency, width, b, r);
                    EXPECT_EQ(shape.rows, r);
                    EXPECT_GE(shape.batch_capacity, b);
                }
            }
        }
    }
}

TEST(CudaGraphCaptureRangeTest, AdaptiveRejectsInvalidDimensions) {
    EXPECT_THROW(adaptiveVerifyCaptureRows(0, 6), std::invalid_argument);
    EXPECT_THROW(adaptiveVerifyCaptureRows(4, 1), std::invalid_argument);
    EXPECT_THROW(adaptiveVerifyCaptureRows(std::numeric_limits<int>::max(), 6), std::invalid_argument);
    for (const auto& dims : std::vector<std::pair<int, int>>{{0, 1}, {5, 5}, {2, 1}, {2, 13}, {1, 0}, {4, 25}}) {
        EXPECT_THROW(adaptiveVerifyReplayShape(4, 6, dims.first, dims.second), std::invalid_argument);
    }
}

}  // namespace
}  // namespace rtp_llm
