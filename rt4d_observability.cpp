#include "rt4d_observability.h"

#include <algorithm>
#include <cmath>

const char* rt4dCameraSequenceName(RT4DCameraSequenceMode mode) {
    switch (mode) {
        case RT4DCameraSequenceMode::Static: return "static";
        case RT4DCameraSequenceMode::DeterministicOrbit:
            return "deterministic-orbit-v1";
    }
    return "unknown";
}

const char* rt4dHistoryResetReasonName(RT4DHistoryResetReason reason) {
    switch (reason) {
        case RT4DHistoryResetReason::None: return "none";
        case RT4DHistoryResetReason::InitialFrame: return "initial_frame";
        case RT4DHistoryResetReason::CameraCut: return "camera_cut";
        case RT4DHistoryResetReason::TransactionalResize:
            return "transactional_resize";
    }
    return "unknown";
}

RT4DHistoryDecision rt4dResolveHistoryDecision(
    bool resourcesFirstFrame,
    RT4DHistoryResetReason pendingReset) {
    if (resourcesFirstFrame)
        return {false, RT4DHistoryResetReason::InitialFrame};
    if (pendingReset != RT4DHistoryResetReason::None)
        return {false, pendingReset};
    return {true, RT4DHistoryResetReason::None};
}

RT4DCameraPose rt4dDeterministicCameraPose(
    RT4DCameraSequenceMode mode,
    const float center[3],
    float radius,
    uint32_t frameIndex,
    uint32_t sequenceSegment) {
    RT4DCameraPose pose{};
    pose.target[0] = center[0];
    pose.target[1] = center[1];
    pose.target[2] = center[2];

    const float distance = std::max(radius, 1e-4f) * 2.5f;
    if (mode == RT4DCameraSequenceMode::Static) {
        pose.eye[0] = center[0];
        pose.eye[1] = center[1];
        pose.eye[2] = center[2] + distance;
        return pose;
    }

    const float framePhase = static_cast<float>(frameIndex) * 0.025f;
    const float cutOffset = static_cast<float>(sequenceSegment) * 0.75f;
    const float angle = -0.20f + framePhase + cutOffset;
    pose.eye[0] = center[0] + std::sin(angle) * distance;
    pose.eye[1] = center[1] + radius *
        (0.05f + 0.02f * std::sin(static_cast<float>(frameIndex) * 0.17f));
    pose.eye[2] = center[2] + std::cos(angle) * distance;
    return pose;
}

double rt4dPercentile(std::vector<double> values, double quantile) {
    if (values.empty()) return 0.0;
    quantile = std::clamp(quantile, 0.0, 1.0);
    std::sort(values.begin(), values.end());
    const double position = quantile * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(std::floor(position));
    const size_t upper = static_cast<size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower);
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

double rt4dMean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    double sum = 0.0;
    for (double value : values) sum += value;
    return sum / static_cast<double>(values.size());
}

double rt4dCompositeLumaMae(const std::vector<uint8_t>& previousRgba,
                            const std::vector<uint8_t>& currentRgba) {
    if (previousRgba.empty() || previousRgba.size() != currentRgba.size() ||
        previousRgba.size() % 4 != 0)
        return 0.0;
    double sum = 0.0;
    const size_t pixels = previousRgba.size() / 4;
    for (size_t pixel = 0; pixel < pixels; ++pixel) {
        const size_t base = pixel * 4;
        const double previous =
            0.2126 * previousRgba[base] +
            0.7152 * previousRgba[base + 1] +
            0.0722 * previousRgba[base + 2];
        const double current =
            0.2126 * currentRgba[base] +
            0.7152 * currentRgba[base + 1] +
            0.0722 * currentRgba[base + 2];
        sum += std::abs(current - previous) / 255.0;
    }
    return sum / static_cast<double>(pixels);
}
