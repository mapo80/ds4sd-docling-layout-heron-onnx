using System;

namespace LayoutSdk.Metrics;

public readonly record struct LayoutExecutionMetrics(
    TimeSpan PreprocessDuration,
    TimeSpan InferenceDuration,
    TimeSpan OverlayDuration)
{
    public TimeSpan TotalDuration => PreprocessDuration + InferenceDuration + OverlayDuration;
}
