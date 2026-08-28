#pragma once

#include <cstdint>
#include <vector>

enum class RT4DCameraSequenceMode : uint8_t {
    Static = 0,
    DeterministicOrbit = 1,
};

enum class RT4DHistoryResetReason : uint8_t {
    None = 0,
    InitialFrame = 1,
    CameraCut = 2,
    TransactionalResize = 3,
};

struct RT4DHistoryDecision {
    bool historyValid = false;
    RT4DHistoryResetReason resetReason = RT4DHistoryResetReason::InitialFrame;
};

struct RT4DCameraPose {
    float eye[3] = {};
    float target[3] = {};
};

struct RT4DCameraSample {
    uint32_t frameIndex = 0;
    uint32_t segment = 0;
    bool cameraCut = false;
    RT4DCameraPose pose{};
};

struct RT4DFrameMetrics {
    uint32_t frameIndex = 0;
    bool historyValid = false;
    RT4DHistoryResetReason resetReason = RT4DHistoryResetReason::None;
    uint64_t foregroundPixels = 0;
    double meanMotionPixels = 0.0;
    double p95MotionPixels = 0.0;
    double maxMotionPixels = 0.0;
    double meanReprojectionConfidence = 0.0;
    double acceptedHistoryRatio = 0.0;
    double meanAcceptedLumaResidual = 0.0;
    double compositeLumaMaeFromPrevious = 0.0;
};

const char* rt4dCameraSequenceName(RT4DCameraSequenceMode mode);
const char* rt4dHistoryResetReasonName(RT4DHistoryResetReason reason);

RT4DHistoryDecision rt4dResolveHistoryDecision(
    bool resourcesFirstFrame,
    RT4DHistoryResetReason pendingReset);

RT4DCameraPose rt4dDeterministicCameraPose(
    RT4DCameraSequenceMode mode,
    const float center[3],
    float radius,
    uint32_t frameIndex,
    uint32_t sequenceSegment);

double rt4dPercentile(std::vector<double> values, double quantile);
double rt4dMean(const std::vector<double>& values);
double rt4dCompositeLumaMae(const std::vector<uint8_t>& previousRgba,
                            const std::vector<uint8_t>& currentRgba);
