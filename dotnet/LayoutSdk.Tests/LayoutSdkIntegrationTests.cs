using System;
using System.IO;
using SkiaSharp;
using LayoutSdk;
using System.Linq;
using LayoutSdk.Configuration;
using LayoutSdkClient = LayoutSdk.LayoutSdk;
using Xunit;
using Xunit.Abstractions;

namespace LayoutSdk.Tests;

public sealed class LayoutSdkIntegrationTests : IClassFixture<DatasetFixture>
{
    private readonly DatasetFixture _fixture;
    private readonly ITestOutputHelper _output;

    public LayoutSdkIntegrationTests(DatasetFixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture;
        _output = output;
    }

    [Fact]
    public void OnnxRuntime_ProducesBoundingBoxes_ForSampleImage()
    {
        var options = new LayoutSdkOptions(
            onnxModelPath: Path.Combine(_fixture.ModelsRoot, "heron-optimized.onnx"),
            ortModelPath: null,
            openVino: new OpenVinoModelOptions(
                modelXmlPath: Path.Combine(_fixture.ModelsRoot, "ov-ir", "heron-converted.xml"),
                weightsBinPath: Path.Combine(_fixture.ModelsRoot, "ov-ir", "heron-converted.bin")),
            defaultLanguage: DocumentLanguage.English,
            validateModelPaths: true);
        options.EnsureModelPaths();

        using var sdk = new LayoutSdkClient(options);

        var normalizedPath = CreateNormalizedImage(_fixture.ImagePath);
        try
        {
            var result = sdk.Process(normalizedPath, overlay: false, LayoutRuntime.Onnx);

            Assert.NotNull(result);
            Assert.NotNull(result.Boxes);
            Assert.NotEmpty(result.Boxes);
            Assert.InRange(result.Boxes.Count, 5, 20);

            _output.WriteLine($"Detected {result.Boxes.Count} layout boxes via ONNX backend.");
            foreach (var box in result.Boxes.Take(5))
            {
                _output.WriteLine($"- {box.Label}: x={box.X:F1}, y={box.Y:F1}, w={box.Width:F1}, h={box.Height:F1}");
            }
        }
        finally
        {
            if (File.Exists(normalizedPath))
            {
                File.Delete(normalizedPath);
            }
        }
    }

    private static string CreateNormalizedImage(string inputPath)
    {
        const int targetSize = 640;
        using var bitmap = SKBitmap.Decode(inputPath) ?? throw new FileNotFoundException("Unable to decode test image.", inputPath);
        var scale = Math.Min((float)targetSize / bitmap.Width, (float)targetSize / bitmap.Height);
        var scaledWidth = Math.Clamp((int)Math.Round(bitmap.Width * scale, MidpointRounding.AwayFromZero), 1, targetSize);
        var scaledHeight = Math.Clamp((int)Math.Round(bitmap.Height * scale, MidpointRounding.AwayFromZero), 1, targetSize);

        using var surface = SKSurface.Create(new SKImageInfo(targetSize, targetSize, SKColorType.Rgba8888, SKAlphaType.Premul));
        if (surface is null)
        {
            throw new InvalidOperationException("Failed to create letterboxing surface for layout test.");
        }

        var canvas = surface.Canvas;
        canvas.Clear(SKColors.Black);

        var offsetX = (targetSize - scaledWidth) / 2f;
        var offsetY = (targetSize - scaledHeight) / 2f;
        var destination = SKRect.Create(offsetX, offsetY, scaledWidth, scaledHeight);
        canvas.DrawBitmap(bitmap, destination);
        canvas.Flush();

        using var image = surface.Snapshot();
        using var data = image.Encode(SKEncodedImageFormat.Png, 100);
        var tempPath = Path.Combine(Path.GetTempPath(), $"layout-sdk-test-{Guid.NewGuid():N}.png");
        using (var stream = File.Open(tempPath, FileMode.Create, FileAccess.Write))
        {
            data.SaveTo(stream);
        }

        return tempPath;
    }

    [Fact]
    public void ComparativeAnalysis_DatasetImage_2305_03393v1_pg9()
    {
        var options = new LayoutSdkOptions(
            onnxModelPath: Path.Combine(_fixture.ModelsRoot, "heron-optimized.onnx"),
            ortModelPath: null,
            openVino: new OpenVinoModelOptions(
                modelXmlPath: Path.Combine(_fixture.ModelsRoot, "ov-ir", "heron-converted.xml"),
                weightsBinPath: Path.Combine(_fixture.ModelsRoot, "ov-ir", "heron-converted.bin")),
            defaultLanguage: DocumentLanguage.English,
            validateModelPaths: true);
        options.EnsureModelPaths();

        using var sdk = new LayoutSdkClient(options);

        // Test with the specific dataset image
        var result = sdk.Process(_fixture.ImagePath, overlay: false, LayoutRuntime.Onnx);

        Assert.NotNull(result);
        Assert.NotNull(result.Boxes);
        Assert.NotEmpty(result.Boxes);

        // Log detailed results for comparison
        _output.WriteLine($"=== LAYOUT ANALYSIS RESULTS ===");
        _output.WriteLine($"Image: {_fixture.ImagePath}");
        _output.WriteLine($"Total detections: {result.Boxes.Count}");
        _output.WriteLine($"Processing time: {result.Metrics.TotalDuration.TotalMilliseconds:F2}ms");
        _output.WriteLine($"Language: {result.Language}");

        _output.WriteLine($"\n=== DETAILED BOX ANALYSIS ===");
        var boxesByLabel = result.Boxes.GroupBy(b => b.Label).OrderByDescending(g => g.Count());

        foreach (var labelGroup in boxesByLabel)
        {
            _output.WriteLine($"\n{labelGroup.Key}: {labelGroup.Count()} boxes");
            foreach (var box in labelGroup.Take(3)) // Show first 3 of each label
            {
                _output.WriteLine($"  - Box: x={box.X:F1}, y={box.Y:F1}, w={box.Width:F1}, h={box.Height:F1}, conf={box.Confidence:F3}");
            }
            if (labelGroup.Count() > 3)
            {
                _output.WriteLine($"  ... and {labelGroup.Count() - 3} more");
            }
        }

        // Validate expected results based on historical data
        var textBoxes = result.Boxes.Where(b => b.Label.Equals("Text", StringComparison.OrdinalIgnoreCase)).ToList();
        var tableBoxes = result.Boxes.Where(b => b.Label.Equals("Table", StringComparison.OrdinalIgnoreCase)).ToList();
        var titleBoxes = result.Boxes.Where(b => b.Label.Equals("Title", StringComparison.OrdinalIgnoreCase)).ToList();

        _output.WriteLine($"\n=== VALIDATION METRICS ===");
        _output.WriteLine($"Text boxes: {textBoxes.Count} (expected: ~8-10)");
        _output.WriteLine($"Table boxes: {tableBoxes.Count} (expected: ~3-4)");
        _output.WriteLine($"Title boxes: {titleBoxes.Count} (expected: ~1-2)");

        // Performance assertions
        Assert.True(result.Boxes.Count >= 10, $"Expected at least 10 detections, got {result.Boxes.Count}");
        Assert.True(result.Boxes.Count <= 20, $"Expected at most 20 detections, got {result.Boxes.Count}");
        Assert.True(result.Metrics.FullTotalDuration.TotalMilliseconds < 2000, $"Expected < 2s, got {result.Metrics.FullTotalDuration.TotalMilliseconds:F2}ms");

        // Quality assertions
        Assert.True(textBoxes.Count >= 5, $"Expected at least 5 text boxes, got {textBoxes.Count}");
        Assert.True(tableBoxes.Count >= 2, $"Expected at least 2 table boxes, got {tableBoxes.Count}");

        // All boxes should have valid dimensions and confidence
        foreach (var box in result.Boxes)
        {
            Assert.True(box.Width > 0, $"Box {box.Label} has invalid width: {box.Width}");
            Assert.True(box.Height > 0, $"Box {box.Label} has invalid height: {box.Height}");
            Assert.True(box.Confidence > 0, $"Box {box.Label} has invalid confidence: {box.Confidence}");
            Assert.True(box.Confidence <= 1, $"Box {box.Label} has invalid confidence: {box.Confidence}");
        }

        _output.WriteLine($"\n=== PERFORMANCE METRICS ===");
        _output.WriteLine($"Preprocess time: {result.Metrics.PreprocessDuration.TotalMilliseconds:F2}ms");
        _output.WriteLine($"Inference time: {result.Metrics.InferenceDuration.TotalMilliseconds:F2}ms");
        _output.WriteLine($"Postprocess time: {result.Metrics.PostprocessDuration.TotalMilliseconds:F2}ms");
        _output.WriteLine($"Total time: {result.Metrics.FullTotalDuration.TotalMilliseconds:F2}ms");
        _output.WriteLine($"Average confidence: {result.Boxes.Average(b => b.Confidence):F3}");

        // Store results for comparison
        _output.WriteLine($"\n=== COMPARISON DATA ===");
        _output.WriteLine($"Python baseline: 13-14 detections, ~800ms");
        _output.WriteLine($".NET result: {result.Boxes.Count} detections, {result.Metrics.FullTotalDuration.TotalMilliseconds:F2}ms");
        _output.WriteLine($"Parity: {Math.Round((double)result.Boxes.Count / 13.5 * 100, 1)}% vs Python count");
        _output.WriteLine($"Performance: {Math.Round(800.0 / result.Metrics.FullTotalDuration.TotalMilliseconds * 100, 1)}% vs Python speed");
        _output.WriteLine($"Post-processing: {result.Metrics.PostprocessDuration.TotalMilliseconds:F2}ms (advanced algorithms)");
    }

}
