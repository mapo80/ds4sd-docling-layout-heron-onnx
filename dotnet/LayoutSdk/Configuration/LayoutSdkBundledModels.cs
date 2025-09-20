using System;
using System.Collections.Generic;
using System.IO;

namespace LayoutSdk.Configuration;

public static class LayoutSdkBundledModels
{
    public const string OptimizedOnnxFileName = "heron-optimized.onnx";
    public const string ConvertedOnnxFileName = "heron-converted.onnx";
    public const string OptimizedFp16OnnxFileName = "heron-optimized-fp16.onnx";
    public const string OptimizedOrtFileName = "heron-optimized.ort";
    public const string OptimizedRuntimeOrtFileName = "heron-optimized.with_runtime_opt.ort";
    public const string OpenVinoXmlFileName = "heron-converted.xml";
    public const string OpenVinoBinFileName = "heron-converted.bin";

    public static string ModelsRoot => Path.Combine(AppContext.BaseDirectory, "models");

    public static string OpenVinoRoot => Path.Combine(ModelsRoot, "ov-ir");

    public static string GetOptimizedOnnxPath() => Path.Combine(ModelsRoot, OptimizedOnnxFileName);

    public static string GetConvertedOnnxPath() => Path.Combine(ModelsRoot, ConvertedOnnxFileName);

    public static string GetOptimizedFp16OnnxPath() => Path.Combine(ModelsRoot, OptimizedFp16OnnxFileName);

    public static string GetOptimizedOrtPath() => Path.Combine(ModelsRoot, OptimizedOrtFileName);

    public static string GetOptimizedRuntimeOrtPath() => Path.Combine(ModelsRoot, OptimizedRuntimeOrtFileName);

    public static string GetOpenVinoXmlPath() => Path.Combine(OpenVinoRoot, OpenVinoXmlFileName);

    public static string GetOpenVinoBinPath() => Path.Combine(OpenVinoRoot, OpenVinoBinFileName);

    public static LayoutSdkOptions CreateOptions(
        DocumentLanguage defaultLanguage = DocumentLanguage.English,
        bool validateModelPaths = true,
        bool useRuntimeOptimizedOrt = true)
        => new(
            onnxModelPath: GetOptimizedOnnxPath(),
            ortModelPath: useRuntimeOptimizedOrt ? GetOptimizedRuntimeOrtPath() : GetOptimizedOrtPath(),
            openVino: new OpenVinoModelOptions(
                modelXmlPath: GetOpenVinoXmlPath(),
                weightsBinPath: GetOpenVinoBinPath()),
            defaultLanguage: defaultLanguage,
            validateModelPaths: validateModelPaths);

    public static void EnsureAllFilesExist()
    {
        foreach (var path in EnumerateExpectedFiles())
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException($"Bundled model file not found: {Path.GetFileName(path)}", path);
            }
        }
    }

    public static IEnumerable<string> EnumerateExpectedFiles()
    {
        yield return GetOptimizedOnnxPath();
        yield return GetConvertedOnnxPath();
        yield return GetOptimizedFp16OnnxPath();
        yield return GetOptimizedOrtPath();
        yield return GetOptimizedRuntimeOrtPath();
        yield return GetOpenVinoXmlPath();
        yield return GetOpenVinoBinPath();
    }
}
