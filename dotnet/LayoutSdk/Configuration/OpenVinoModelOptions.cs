using System;
using System.IO;

namespace LayoutSdk.Configuration;

public sealed class OpenVinoModelOptions
{
    public OpenVinoModelOptions(string modelXmlPath, string? weightsBinPath = null)
    {
        if (string.IsNullOrWhiteSpace(modelXmlPath))
        {
            throw new ArgumentException("Model XML path must be provided", nameof(modelXmlPath));
        }

        ModelXmlPath = modelXmlPath;
        WeightsBinPath = !string.IsNullOrWhiteSpace(weightsBinPath)
            ? weightsBinPath!
            : InferWeightsPath(modelXmlPath);
    }

    public string ModelXmlPath { get; }

    public string WeightsBinPath { get; }

    public void EnsureFilesExist()
    {
        ValidatePath(ModelXmlPath, nameof(ModelXmlPath));
        ValidatePath(WeightsBinPath, nameof(WeightsBinPath));
    }

    private static string InferWeightsPath(string xmlPath)
    {
        var binPath = Path.ChangeExtension(xmlPath, ".bin");
        if (string.IsNullOrWhiteSpace(binPath))
        {
            throw new InvalidOperationException("Unable to infer OpenVINO weights path from XML path.");
        }

        return binPath;
    }

    private static void ValidatePath(string path, string argumentName)
    {
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
        {
            throw new FileNotFoundException($"Model file not found for {argumentName}", path);
        }
    }
}
