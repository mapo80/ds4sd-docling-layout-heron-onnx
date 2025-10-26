using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using LayoutSdk.Assets;

namespace LayoutSdk.Configuration;

public static class LayoutSdkBundledModels
{
    public const string OptimizedOnnxFileName = "heron-optimized.onnx";
    public const string ConvertedOnnxFileName = "heron-converted.onnx";
    public const string OptimizedFp16OnnxFileName = "heron-optimized-fp16.onnx";
    public const string OptimizedOrtFileName = "heron-optimized.ort";
    public const string OptimizedRuntimeOrtFileName = "heron-optimized.with_runtime_opt.ort";

    public static string ModelsRoot => Path.Combine(AppContext.BaseDirectory, "models");


    public static string GetOptimizedOnnxPath() => Path.Combine(ModelsRoot, OptimizedOnnxFileName);

    public static string GetConvertedOnnxPath() => Path.Combine(ModelsRoot, ConvertedOnnxFileName);

    public static string GetOptimizedFp16OnnxPath() => Path.Combine(ModelsRoot, OptimizedFp16OnnxFileName);

    public static string GetOptimizedOrtPath() => Path.Combine(ModelsRoot, OptimizedOrtFileName);

    public static string GetOptimizedRuntimeOrtPath() => Path.Combine(ModelsRoot, OptimizedRuntimeOrtFileName);


    public static LayoutSdkOptions CreateOptions(
        DocumentLanguage defaultLanguage = DocumentLanguage.English,
        bool validateModelPaths = true)
        => new(
            onnxModelPath: GetOptimizedOnnxPath(),
            defaultLanguage: defaultLanguage,
            validateModelPaths: validateModelPaths,
            modelVariant: LayoutModelVariant.Accurate);

    public static void EnsureAllFilesExist(
        Action<string>? logger = null,
        CancellationToken cancellationToken = default)
    {
        foreach (var path in EnumerateExpectedFiles())
        {
            LayoutModelDownloader.EnsureModel(path, logger: logger, cancellationToken: cancellationToken);
        }
    }

    public static IEnumerable<string> EnumerateExpectedFiles()
    {
        yield return GetOptimizedOnnxPath();
        yield return GetConvertedOnnxPath();
        yield return GetOptimizedFp16OnnxPath();
        yield return GetOptimizedOrtPath();
        yield return GetOptimizedRuntimeOrtPath();
    }
}
