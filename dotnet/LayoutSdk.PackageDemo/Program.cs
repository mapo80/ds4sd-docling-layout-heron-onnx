using System;
using System.Collections.Generic;
using System.IO;
using LayoutSdk;
using LayoutSdk.Configuration;
using SkiaSharp;

Console.WriteLine("Docling Layout SDK (NuGet package demo)");
Console.WriteLine($"Models packaged under: {LayoutSdkBundledModels.ModelsRoot}");

LayoutSdkBundledModels.EnsureAllFilesExist();
Console.WriteLine("All bundled model files were found.");

var options = LayoutSdkBundledModels.CreateOptions(validateModelPaths: true);
options.EnsureModelPaths();

var imagePath = ResolveDatasetImage("gazette_de_france.jpg");
if (imagePath is null)
{
    Console.WriteLine("Sample image not found in the repo. A blank 640x640 page will be generated.");
}
else
{
    Console.WriteLine($"Using sample image: {imagePath}");
}

var preparedImagePath = PrepareInputImage(imagePath);
Console.WriteLine($"Running inference with input: {preparedImagePath}");

using var sdk = new LayoutSdk.LayoutSdk(options);

var runtimes = new[]
{
    LayoutRuntime.Onnx,
    LayoutRuntime.Ort,
    LayoutRuntime.OpenVino
};

var results = new Dictionary<LayoutRuntime, LayoutResult>();

foreach (var runtime in runtimes)
{
    Console.WriteLine();
    Console.WriteLine($"=== {runtime} ===");
    Console.WriteLine($"Model source: {DescribeModelSource(runtime)}");

    var result = RunInference(sdk, runtime, preparedImagePath);
    results[runtime] = result;

    Console.WriteLine($"Detected {result.Boxes.Count} layout elements.");
    Console.WriteLine($"Preprocess ms: {result.Metrics.PreprocessDuration.TotalMilliseconds:F2}");
    Console.WriteLine($"Inference ms: {result.Metrics.InferenceDuration.TotalMilliseconds:F2}");
    Console.WriteLine($"Total ms: {result.Metrics.TotalDuration.TotalMilliseconds:F2}");
}

Console.WriteLine();
ValidateParity(results, LayoutRuntime.Onnx, LayoutRuntime.Ort);
ValidateParity(results, LayoutRuntime.Onnx, LayoutRuntime.OpenVino);

static LayoutResult RunInference(LayoutSdk.LayoutSdk sdk, LayoutRuntime runtime, string imagePath)
{
    return sdk.Process(imagePath, overlay: false, runtime);
}

static void ValidateParity(IDictionary<LayoutRuntime, LayoutResult> results, LayoutRuntime baseline, LayoutRuntime other)
{
    if (!results.TryGetValue(baseline, out var baselineResult) ||
        !results.TryGetValue(other, out var otherResult))
    {
        return;
    }

    var difference = Math.Abs(baselineResult.Boxes.Count - otherResult.Boxes.Count);
    if (difference == 0)
    {
        Console.WriteLine($"Results parity check ({baseline} vs {other}): OK ({baselineResult.Boxes.Count} boxes).");
    }
    else
    {
        Console.WriteLine($"Results parity check ({baseline} vs {other}): WARNING - mismatch of {difference} boxes.");
    }
}

static string DescribeModelSource(LayoutRuntime runtime)
    => runtime switch
    {
        LayoutRuntime.Onnx => LayoutSdkBundledModels.GetOptimizedOnnxPath(),
        LayoutRuntime.Ort => LayoutSdkBundledModels.GetOptimizedRuntimeOrtPath(),
        LayoutRuntime.OpenVino => $"{LayoutSdkBundledModels.GetOpenVinoXmlPath()} (+ {LayoutSdkBundledModels.GetOpenVinoBinPath()})",
        _ => "Unknown runtime"
    };

static string PrepareInputImage(string? sourceImage)
{
    var tempPath = Path.Combine(Path.GetTempPath(), "docling-layout-sample.png");
    if (sourceImage is null)
    {
        using var blank = new SKBitmap(new SKImageInfo(640, 640));
        using (var canvas = new SKCanvas(blank))
        {
            canvas.Clear(SKColors.White);
        }

        using var blankImage = SKImage.FromBitmap(blank);
        using var blankData = blankImage.Encode(SKEncodedImageFormat.Png, 90);
        using var blankStream = File.Open(tempPath, FileMode.Create, FileAccess.Write);
        blankData.SaveTo(blankStream);
        return tempPath;
    }

    using var bitmap = SKBitmap.Decode(sourceImage)
                      ?? throw new InvalidOperationException($"Unable to decode image at {sourceImage}");
    var sampling = new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.Linear);
    using var resized = bitmap.Resize(new SKImageInfo(640, 640), sampling)
                      ?? throw new InvalidOperationException("Failed to resize sample image to 640x640.");
    using var image = SKImage.FromBitmap(resized);
    using var data = image.Encode(SKEncodedImageFormat.Png, 95);
    using var stream = File.Open(tempPath, FileMode.Create, FileAccess.Write);
    data.SaveTo(stream);
    return tempPath;
}

static string? ResolveDatasetImage(string fileName)
{
    var current = AppContext.BaseDirectory;
    for (var i = 0; i < 8; i++)
    {
        var candidate = Path.Combine(current, "dataset", fileName);
        if (File.Exists(candidate))
        {
            return candidate;
        }

        current = Path.GetFullPath(Path.Combine(current, ".."));
    }

    return null;
}
