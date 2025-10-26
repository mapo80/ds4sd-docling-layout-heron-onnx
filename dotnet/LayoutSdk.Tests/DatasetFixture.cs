using System;
using System.IO;

namespace LayoutSdk.Tests;

public sealed class DatasetFixture
{
    private const string PrimaryImageRelativePath = "dataset/2305.03393v1-pg9-img.png";
    private const string FallbackImageRelativePath = "dataset/gazette_de_france.jpg";
    private const string ModelsRelativePath = "src/submodules/ds4sd-docling-layout-heron-onnx/dotnet/LayoutSdk/PackagedModels/models";

    public DatasetFixture()
    {
        var root = LocateRepositoryRoot();
        RepositoryRoot = root;
        ImagePath = ResolveImagePath(root, out var usingFallback);
        UsingFallbackImage = usingFallback;
        ModelsRoot = Path.Combine(root, ModelsRelativePath);

        Directory.CreateDirectory(ModelsRoot);
    }

    public string RepositoryRoot { get; }

    public string ImagePath { get; }

    public bool UsingFallbackImage { get; }

    public string ModelsRoot { get; }

    private static string LocateRepositoryRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            var datasetDir = Path.Combine(dir.FullName, "dataset");
            if (Directory.Exists(datasetDir))
            {
                var primary = Path.Combine(datasetDir, Path.GetFileName(PrimaryImageRelativePath));
                var fallback = Path.Combine(datasetDir, Path.GetFileName(FallbackImageRelativePath));
                if (File.Exists(primary) || File.Exists(fallback))
                {
                    return dir.FullName;
                }
            }

            dir = dir.Parent;
        }

        throw new InvalidOperationException(
            $"Unable to locate repository root containing either {PrimaryImageRelativePath} or {FallbackImageRelativePath}.");
    }

    private static string ResolveImagePath(string root, out bool usingFallback)
    {
        var primaryPath = Path.Combine(root, PrimaryImageRelativePath);
        if (File.Exists(primaryPath))
        {
            usingFallback = false;
            return primaryPath;
        }

        var fallbackPath = Path.Combine(root, FallbackImageRelativePath);
        if (File.Exists(fallbackPath))
        {
            usingFallback = true;
            return fallbackPath;
        }

        throw new FileNotFoundException(
            $"Neither {PrimaryImageRelativePath} nor {FallbackImageRelativePath} were found under repository root.",
            primaryPath);
    }
}
