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

}
