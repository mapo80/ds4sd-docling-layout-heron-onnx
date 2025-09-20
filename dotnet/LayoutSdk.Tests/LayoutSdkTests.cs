using LayoutSdk;
using LayoutSdk.Configuration;
using LayoutSdk.Factories;
using System;
using System.Collections.Generic;
using System.IO;
using Xunit;
using LayoutSdk.Inference;
using LayoutSdk.Metrics;
using LayoutSdk.Processing;
using SkiaSharp;

namespace LayoutSdk.Tests;

file class FakeBackend : ILayoutBackend
{
    public LayoutBackendResult Infer(ImageTensor tensor) =>
        new(new List<BoundingBox> { new(1, 1, 2, 2, "box") });
}

file class FakeBackendFactory : ILayoutBackendFactory
{
    private readonly ILayoutBackend _backend = new FakeBackend();

    public ILayoutBackend Create(LayoutRuntime runtime) => _backend;
}

public class LayoutSdkTests
{
    private static LayoutSdk CreateSdkWithFakeBackend()
    {
        var options = new LayoutSdkOptions(
            "onnx-path",
            new OpenVinoModelOptions("xml-path", "bin-path"),
            DocumentLanguage.Italian);
        return new LayoutSdk(options, new FakeBackendFactory());
    }

    [Fact]
    public void Process_EmptyPath_Throws()
    {
        var sdk = CreateSdkWithFakeBackend();
        Assert.Throws<ArgumentException>(() => sdk.Process("", false, LayoutRuntime.OnnxRuntime));
    }

    [Fact]
    public void Process_MissingImage_Throws()
    {
        var sdk = CreateSdkWithFakeBackend();
        Assert.Throws<FileNotFoundException>(() => sdk.Process("missing.png", false, LayoutRuntime.OnnxRuntime));
    }

    private static string SampleImage =>
        Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..","..","..","..","..","dataset", "gazette_de_france.jpg"));

    [Fact]
    public void Process_Overlay_ReturnsBitmap()
    {
        var sdk = CreateSdkWithFakeBackend();
        var result = sdk.Process(SampleImage, true, LayoutRuntime.OnnxRuntime);
        Assert.NotNull(result.OverlayImage);
        Assert.Single(result.Boxes);
        Assert.Equal(DocumentLanguage.Italian, result.Language);
        Assert.True(result.Metrics.OverlayDuration >= TimeSpan.Zero);
        Assert.True(result.Metrics.PreprocessDuration >= TimeSpan.Zero);
        Assert.True(result.Metrics.InferenceDuration >= TimeSpan.Zero);
        Assert.Equal(result.Metrics.PreprocessDuration + result.Metrics.InferenceDuration + result.Metrics.OverlayDuration, result.Metrics.TotalDuration);
    }

    [Fact]
    public void Process_NoOverlay_ReturnsNull()
    {
        var sdk = CreateSdkWithFakeBackend();
        var result = sdk.Process(SampleImage, false, LayoutRuntime.OnnxRuntime);
        Assert.Null(result.OverlayImage);
        Assert.Equal(TimeSpan.Zero, result.Metrics.OverlayDuration);
    }

    [Fact]
    public void OptionsEnsureModelPathsThrowsWhenMissing()
    {
        var options = new LayoutSdkOptions(
            "missing",
            new OpenVinoModelOptions("missing.xml", "missing.bin"),
            validateModelPaths: true);
        Assert.Throws<FileNotFoundException>(() => options.EnsureModelPaths());
    }

    [Fact]
    public void OptionsEnsureModelPathsSucceedsWhenFilesExist()
    {
        var onnx = Path.GetTempFileName();
        var xml = Path.GetTempFileName();
        var bin = Path.GetTempFileName();
        try
        {
            var options = new LayoutSdkOptions(
                onnx,
                new OpenVinoModelOptions(xml, bin),
                DocumentLanguage.English,
                validateModelPaths: true);
            options.EnsureModelPaths();
        }
        finally
        {
            File.Delete(onnx);
            File.Delete(xml);
            File.Delete(bin);
        }
    }

    [Fact]
    public void OpenVinoModelOptionsInfersBinPathWhenMissing()
    {
        var options = new OpenVinoModelOptions("/tmp/model.xml");
        Assert.Equal(Path.ChangeExtension("/tmp/model.xml", ".bin"), options.WeightsBinPath);
}
}
