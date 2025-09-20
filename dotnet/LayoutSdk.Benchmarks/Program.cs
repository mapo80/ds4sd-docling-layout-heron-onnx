using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json;
using System.Collections.Generic;
using System.Linq;
using LayoutSdk;
using LayoutSdk.Configuration;
using SkiaSharp;

static Dictionary<string, string> ParseArgs(string[] args)
{
    var dict = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
    for (int i = 0; i < args.Length; i++)
    {
        var token = args[i];
        if (!token.StartsWith("--", StringComparison.Ordinal))
        {
            continue;
        }

        var key = token;
        string value = (i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal))
            ? args[++i]
            : "true";
        dict[key] = value;
    }

    return dict;
}

static string ResizeToTemp(string path, int w, int h)
{
    using var bmp = SKBitmap.Decode(path);
    using var resized = bmp.Resize(new SKImageInfo(w, h), SKFilterQuality.High);
    var tmp = Path.GetTempFileName() + ".png";
    using var img = SKImage.FromBitmap(resized ?? bmp);
    using var data = img.Encode(SKEncodedImageFormat.Png, 90);
    using var fs = File.OpenWrite(tmp);
    data.SaveTo(fs);
    return tmp;
}

static IReadOnlyList<string> CollectImages(string directory)
{
    if (!Directory.Exists(directory))
    {
        return Array.Empty<string>();
    }

    return Directory.GetFiles(directory)
        .Where(f => f.EndsWith(".jpg", true, CultureInfo.InvariantCulture)
                 || f.EndsWith(".jpeg", true, CultureInfo.InvariantCulture)
                 || f.EndsWith(".png", true, CultureInfo.InvariantCulture)
                 || f.EndsWith(".bmp", true, CultureInfo.InvariantCulture)
                 || f.EndsWith(".tif", true, CultureInfo.InvariantCulture)
                 || f.EndsWith(".tiff", true, CultureInfo.InvariantCulture))
        .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
        .ToList();
}

static double Percentile(List<double> seq, double percentile)
{
    if (seq.Count == 0)
    {
        return double.NaN;
    }

    var ordered = seq.OrderBy(x => x).ToList();
    var index = (int)Math.Ceiling(percentile / 100.0 * ordered.Count) - 1;
    index = Math.Clamp(index, 0, ordered.Count - 1);
    return ordered[index];
}

static string Sha256Of(string path)
{
    using var sha = SHA256.Create();
    using var stream = File.OpenRead(path);
    return Convert.ToHexString(sha.ComputeHash(stream)).ToLowerInvariant();
}

record BenchmarkSummary(int Count, double MeanMs, double MedianMs, double P95Ms);

record BenchmarkArtifacts(LayoutRuntime Runtime, string OutputDirectory, BenchmarkSummary Summary);

static BenchmarkArtifacts RunBenchmark(
    LayoutRuntime runtime,
    LayoutSdkOptions options,
    IReadOnlyList<string> imageFiles,
    string outputDir,
    int warmup,
    int runsPerImage,
    int targetH,
    int targetW)
{
    Directory.CreateDirectory(outputDir);

    using var sdk = new LayoutSdk.LayoutSdk(options);

    var timings = new List<double>();

    var warmSource = imageFiles[0];
    var warmResized = ResizeToTemp(warmSource, targetW, targetH);
    for (int i = 0; i < warmup; i++)
    {
        sdk.Process(warmResized, overlay: false, runtime);
    }

    using (var csv = new StreamWriter(Path.Combine(outputDir, "timings.csv")))
    {
        csv.WriteLine("filename,ms");
        foreach (var file in imageFiles)
        {
            var prepPath = ResizeToTemp(file, targetW, targetH);
            for (int run = 0; run < runsPerImage; run++)
            {
                var result = sdk.Process(prepPath, overlay: false, runtime);
                var ms = result.Metrics.TotalDuration.TotalMilliseconds;
                csv.WriteLine($"{Path.GetFileName(file)},{ms:F3}");
                timings.Add(ms);
            }
        }
    }

    var summary = new BenchmarkSummary(
        Count: timings.Count,
        MeanMs: timings.Count > 0 ? timings.Average() : double.NaN,
        MedianMs: Percentile(timings, 50),
        P95Ms: Percentile(timings, 95));

    File.WriteAllText(
        Path.Combine(outputDir, "summary.json"),
        JsonSerializer.Serialize(summary, new JsonSerializerOptions { WriteIndented = true }));

    object modelInfo = runtime switch
    {
        LayoutRuntime.Onnx => new
        {
            runtime = "onnx",
            model_path = options.OnnxModelPath,
            model_size_bytes = File.Exists(options.OnnxModelPath) ? new FileInfo(options.OnnxModelPath).Length : 0L,
            device = "CPU",
            precision = options.OnnxModelPath.ToLowerInvariant().Contains("fp16") ? "fp16" : "fp32"
        },
        LayoutRuntime.Ort => new
        {
            runtime = "ort",
            model_path = options.OrtModelPath,
            model_size_bytes = !string.IsNullOrWhiteSpace(options.OrtModelPath) && File.Exists(options.OrtModelPath)
                ? new FileInfo(options.OrtModelPath).Length
                : 0L,
            device = "CPU",
            precision = options.OrtModelPath?.ToLowerInvariant().Contains("fp16") == true ? "fp16" : "fp32"
        },
        LayoutRuntime.OpenVino => new
        {
            runtime = "openvino",
            xml_path = options.OpenVino.ModelXmlPath,
            xml_size_bytes = File.Exists(options.OpenVino.ModelXmlPath) ? new FileInfo(options.OpenVino.ModelXmlPath).Length : 0L,
            bin_path = options.OpenVino.WeightsBinPath,
            bin_size_bytes = File.Exists(options.OpenVino.WeightsBinPath) ? new FileInfo(options.OpenVino.WeightsBinPath).Length : 0L,
            device = "CPU",
            precision = options.OpenVino.ModelXmlPath.ToLowerInvariant().Contains("fp16") ? "fp16" : "fp32"
        },
        _ => throw new ArgumentOutOfRangeException(nameof(runtime), runtime, null)
    };

    File.WriteAllText(
        Path.Combine(outputDir, "model_info.json"),
        JsonSerializer.Serialize(modelInfo, new JsonSerializerOptions { WriteIndented = true }));

    var env = new
    {
        dotnet = Environment.Version.ToString(),
        os = Environment.OSVersion.ToString()
    };
    File.WriteAllText(Path.Combine(outputDir, "env.json"), JsonSerializer.Serialize(env, new JsonSerializerOptions { WriteIndented = true }));

    File.WriteAllText(
        Path.Combine(outputDir, "config.json"),
        JsonSerializer.Serialize(new
        {
            runtime = runtime.ToString(),
            warmup,
            runs_per_image = runsPerImage,
            target_h = targetH,
            target_w = targetW
        }, new JsonSerializerOptions { WriteIndented = true }));

    var files = new[] { "timings.csv", "summary.json", "model_info.json", "env.json", "config.json" }
        .Select(f => new { file = f, sha256 = Sha256Of(Path.Combine(outputDir, f)) })
        .ToList();

    File.WriteAllText(
        Path.Combine(outputDir, "manifest.json"),
        JsonSerializer.Serialize(new { files }, new JsonSerializerOptions { WriteIndented = true }));

    File.WriteAllText(Path.Combine(outputDir, "logs.txt"), $"RUN {runtime} ok, N={timings.Count}\n");

    return new BenchmarkArtifacts(runtime, outputDir, summary);
}

var parameters = ParseArgs(args);

if (!parameters.TryGetValue("--variant-name", out var variant))
{
    Console.Error.WriteLine("--variant-name is required");
    return;
}

var compare = parameters.ContainsKey("--compare");
LayoutRuntime[] runtimes;
if (compare)
{
    runtimes = new[] { LayoutRuntime.Onnx, LayoutRuntime.OpenVino };
}
else if (parameters.TryGetValue("--runtime", out var runtimeValue))
{
    runtimes = new[] { Enum.Parse<LayoutRuntime>(runtimeValue, ignoreCase: true) };
}
else
{
    Console.Error.WriteLine("--runtime is required when --compare is not specified");
    return;
}

var imagesDir = parameters.GetValueOrDefault("--images", "./dataset");
var outputRoot = parameters.GetValueOrDefault("--output", "results");
int warmup = int.Parse(parameters.GetValueOrDefault("--warmup", "1"), CultureInfo.InvariantCulture);
int runsPerImage = int.Parse(parameters.GetValueOrDefault("--runs-per-image", "1"), CultureInfo.InvariantCulture);
int targetH = int.Parse(parameters.GetValueOrDefault("--target-h", "640"), CultureInfo.InvariantCulture);
int targetW = int.Parse(parameters.GetValueOrDefault("--target-w", "640"), CultureInfo.InvariantCulture);

var images = CollectImages(imagesDir).ToList();
if (images.Count == 0)
{
    using var bmp = new SKBitmap(targetW, targetH);
    using var img = SKImage.FromBitmap(bmp);
    var tmp = Path.GetTempFileName() + ".png";
    using var data = img.Encode(SKEncodedImageFormat.Png, 90);
    using var fs = File.OpenWrite(tmp);
    data.SaveTo(fs);
    images.Add(tmp);
}

if (compare && images.Count > 2)
{
    images = images.Take(2).ToList();
}

var onnxModelPath = parameters.GetValueOrDefault("--onnx-model", "models/heron-optimized.onnx");
var ortModelPath = parameters.GetValueOrDefault("--ort-model", "models/heron-optimized.with_runtime_opt.ort");
var openVinoXml = parameters.GetValueOrDefault("--openvino-xml", "models/ov-ir/heron-optimized.xml");
parameters.TryGetValue("--openvino-bin", out var openVinoBin);
var openVinoOptions = string.IsNullOrWhiteSpace(openVinoBin)
    ? new OpenVinoModelOptions(openVinoXml)
    : new OpenVinoModelOptions(openVinoXml, openVinoBin);

var options = new LayoutSdkOptions(
    onnxModelPath,
    ortModelPath,
    openVinoOptions,
    defaultLanguage: DocumentLanguage.English,
    validateModelPaths: parameters.ContainsKey("--validate-models"));

var timestamp = DateTime.UtcNow.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture);
var runDirectory = Path.Combine(outputRoot, variant, $"run-{timestamp}");
Directory.CreateDirectory(runDirectory);

var artifacts = new List<BenchmarkArtifacts>();
foreach (var runtime in runtimes)
{
    var runtimeDir = compare
        ? Path.Combine(runDirectory, runtime.ToString().ToLowerInvariant())
        : runDirectory;
    artifacts.Add(RunBenchmark(runtime, options, images, runtimeDir, warmup, runsPerImage, targetH, targetW));
}

if (compare)
{
    var payload = artifacts.Select(a => new
    {
        runtime = a.Runtime.ToString(),
        summary = a.Summary,
        output_directory = Path.GetRelativePath(runDirectory, a.OutputDirectory)
    });

    File.WriteAllText(
        Path.Combine(runDirectory, "comparison.json"),
        JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
}

Console.WriteLine($"OK: {runDirectory}");
