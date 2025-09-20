using System;
using System.IO;

namespace LayoutSdk.Configuration;

public sealed class LayoutSdkOptions
{
    public LayoutSdkOptions(
        string onnxModelPath,
        OpenVinoModelOptions openVino,
        DocumentLanguage defaultLanguage = DocumentLanguage.English,
        bool validateModelPaths = false)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
        {
            throw new ArgumentException("ONNX model path must be provided", nameof(onnxModelPath));
        }

        OnnxModelPath = onnxModelPath;
        OpenVino = openVino ?? throw new ArgumentNullException(nameof(openVino));
        DefaultLanguage = defaultLanguage;
        ValidateModelPaths = validateModelPaths;
    }

    public string OnnxModelPath { get; }

    public OpenVinoModelOptions OpenVino { get; }

    public DocumentLanguage DefaultLanguage { get; }

    public bool ValidateModelPaths { get; }

    public void EnsureModelPaths()
    {
        if (!ValidateModelPaths)
        {
            return;
        }

        ValidatePath(OnnxModelPath, nameof(OnnxModelPath));
        OpenVino.EnsureFilesExist();
    }

    private static void ValidatePath(string? path, string argumentName)
    {
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
        {
            throw new FileNotFoundException($"Model file not found for {argumentName}", path);
        }
    }
}
