using LayoutSdk.Inference;
using LayoutSdk.Metrics;
using LayoutSdk.Processing;
using SkiaSharp;
using System;
using System.Collections.Generic;
using Xunit;

namespace LayoutSdk.Tests;

public class LayoutPipelineTests
{
    private sealed class RecordingBackend : ILayoutBackend, IDisposable
    {
        public bool Disposed { get; private set; }
        public bool Called { get; private set; }

        public LayoutBackendResult Infer(ImageTensor tensor)
        {
            Called = true;
            return new LayoutBackendResult(new List<BoundingBox> { new(0, 0, 1, 1, "box") });
        }

        public void Dispose()
        {
            Disposed = true;
        }
    }

    private sealed class RecordingPreprocessor : IImagePreprocessor
    {
        public bool Called { get; private set; }

        public ImageTensor Preprocess(SKBitmap image)
        {
            Called = true;
            return ImageTensor.Rent(image.Width, image.Height, 3);
        }
    }

    [Fact]
    public void Execute_NullImage_Throws()
    {
        var pipeline = new LayoutPipeline(new RecordingBackend(), new RecordingPreprocessor());
        Assert.Throws<ArgumentNullException>(() => pipeline.Execute(null!));
    }

    [Fact]
    public void Execute_RunsPreprocessAndInference()
    {
        using var image = new SKBitmap(2, 2);
        var backend = new RecordingBackend();
        var preprocessor = new RecordingPreprocessor();
        var pipeline = new LayoutPipeline(backend, preprocessor);

        var result = pipeline.Execute(image);
        Assert.True(preprocessor.Called);
        Assert.True(backend.Called);
        Assert.Single(result.Boxes);
        Assert.True(result.Metrics.PreprocessDuration >= TimeSpan.Zero);
        Assert.True(result.Metrics.InferenceDuration >= TimeSpan.Zero);
        Assert.Equal(TimeSpan.Zero, result.Metrics.OverlayDuration);
        Assert.Equal(result.Metrics.PreprocessDuration + result.Metrics.InferenceDuration, result.Metrics.TotalDuration);
    }

    [Fact]
    public void Dispose_DisposesBackendOnce()
    {
        var backend = new RecordingBackend();
        var pipeline = new LayoutPipeline(backend, new RecordingPreprocessor());
        pipeline.Dispose();
        pipeline.Dispose();
        Assert.True(backend.Disposed);
    }
}
