using System;
using System.Diagnostics;
using LayoutSdk.Inference;
using LayoutSdk.Metrics;
using SkiaSharp;

namespace LayoutSdk.Processing;

internal sealed class LayoutPipeline : IDisposable
{
    private readonly ILayoutBackend _backend;
    private readonly IImagePreprocessor _preprocessor;
    private readonly LayoutPostprocessor _postprocessor;
    private bool _disposed;

    public LayoutPipeline(ILayoutBackend backend, IImagePreprocessor preprocessor, LayoutPostprocessor postprocessor)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _preprocessor = preprocessor ?? throw new ArgumentNullException(nameof(preprocessor));
        _postprocessor = postprocessor ?? throw new ArgumentNullException(nameof(postprocessor));
    }

    public LayoutPipelineResult Execute(SKBitmap image)
    {
        if (image is null)
        {
            throw new ArgumentNullException(nameof(image));
        }

        var preprocessWatch = Stopwatch.StartNew();
        using var tensor = _preprocessor.Preprocess(image);
        preprocessWatch.Stop();

        var inferenceWatch = Stopwatch.StartNew();
        var backendResult = _backend.Infer(tensor);
        inferenceWatch.Stop();

        // Apply advanced post-processing
        var postprocessWatch = Stopwatch.StartNew();
        var postprocessedBoxes = _postprocessor.Postprocess(backendResult.Boxes);
        postprocessWatch.Stop();

        var metrics = new LayoutExecutionMetrics(
            PreprocessDuration: preprocessWatch.Elapsed,
            InferenceDuration: inferenceWatch.Elapsed,
            OverlayDuration: TimeSpan.Zero)
        {
            PostprocessDuration = postprocessWatch.Elapsed
        };

        return new LayoutPipelineResult(postprocessedBoxes, metrics);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        if (_backend is IDisposable disposable)
        {
            disposable.Dispose();
        }

        _disposed = true;
    }
}
