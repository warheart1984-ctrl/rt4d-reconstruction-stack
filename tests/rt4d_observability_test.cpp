#include "rt4d_observability.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {
int fail(const char* message) {
    std::fprintf(stderr, "[observability-test] FAIL: %s\n", message);
    return 1;
}

bool near(double a, double b, double tolerance = 1e-6) {
    return std::abs(a - b) <= tolerance;
}
}

int main() {
    const RT4DHistoryDecision initial = rt4dResolveHistoryDecision(
        true, RT4DHistoryResetReason::None);
    const RT4DHistoryDecision stable = rt4dResolveHistoryDecision(
        false, RT4DHistoryResetReason::None);
    const RT4DHistoryDecision cut = rt4dResolveHistoryDecision(
        false, RT4DHistoryResetReason::CameraCut);
    const RT4DHistoryDecision resize = rt4dResolveHistoryDecision(
        false, RT4DHistoryResetReason::TransactionalResize);
    if (initial.historyValid ||
        initial.resetReason != RT4DHistoryResetReason::InitialFrame)
        return fail("initial history decision is incorrect");
    if (!stable.historyValid || stable.resetReason != RT4DHistoryResetReason::None)
        return fail("stable history decision is incorrect");
    if (cut.historyValid || cut.resetReason != RT4DHistoryResetReason::CameraCut)
        return fail("camera-cut reset decision is incorrect");
    if (resize.historyValid ||
        resize.resetReason != RT4DHistoryResetReason::TransactionalResize)
        return fail("transactional-resize reset decision is incorrect");

    const float center[3] = {1.0f, 2.0f, 3.0f};
    const RT4DCameraPose a = rt4dDeterministicCameraPose(
        RT4DCameraSequenceMode::DeterministicOrbit, center, 2.0f, 7, 0);
    const RT4DCameraPose b = rt4dDeterministicCameraPose(
        RT4DCameraSequenceMode::DeterministicOrbit, center, 2.0f, 7, 0);
    const RT4DCameraPose afterCut = rt4dDeterministicCameraPose(
        RT4DCameraSequenceMode::DeterministicOrbit, center, 2.0f, 7, 1);
    for (size_t axis = 0; axis < 3; ++axis)
        if (!near(a.eye[axis], b.eye[axis]))
            return fail("camera sequence is not deterministic");
    if (near(a.eye[0], afterCut.eye[0], 0.1) &&
        near(a.eye[2], afterCut.eye[2], 0.1))
        return fail("camera cut did not change the deterministic segment");

    if (!near(rt4dPercentile({1.0, 2.0, 3.0, 4.0}, 0.50), 2.5) ||
        !near(rt4dPercentile({1.0, 2.0, 3.0, 4.0}, 0.95), 3.85) ||
        !near(rt4dMean({1.0, 2.0, 3.0, 4.0}), 2.5))
        return fail("summary statistics are incorrect");

    const std::vector<uint8_t> black = {0, 0, 0, 255};
    const std::vector<uint8_t> white = {255, 255, 255, 255};
    if (!near(rt4dCompositeLumaMae(black, white), 1.0))
        return fail("composite luma delta is incorrect");

    std::fprintf(stderr,
                 "[observability-test] PASS: camera sequence, history reset, "
                 "percentiles, and frame delta\n");
    return 0;
}
