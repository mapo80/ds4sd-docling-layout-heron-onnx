using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using LayoutSdk.Assets;

namespace LayoutSdk.Configuration;

/// <summary>
/// Provides helpers to manage Docling Layout Heron models downloaded from the GitHub release.
/// </summary>
public static class LayoutReleaseModels
{
    /// <summary>
    /// Environment variable that, when set, overrides the directory used to cache downloaded models.
    /// </summary>
    public const string ModelsRootEnvironmentVariable = "LAYOUTSDK_MODELS_DIR";

    private static readonly string ModelsRootValue = ResolveModelsRoot();

    /// <summary>
    /// Directory that contains the downloaded ONNX assets.
    /// </summary>
    public static string ModelsRoot => ModelsRootValue;

    /// <summary>
    /// Resolve the file name for a given variant.
    /// </summary>
    public static string GetModelFileName(LayoutModelVariant variant) => variant switch
    {
        LayoutModelVariant.Accurate => LayoutSdkBundledModels.OptimizedOnnxFileName,
        LayoutModelVariant.Fast => LayoutSdkBundledModels.OptimizedFp16OnnxFileName,
        _ => throw new ArgumentOutOfRangeException(nameof(variant), variant, "Unsupported model variant.")
    };

    /// <summary>
    /// Resolve the full absolute path for the requested model variant.
    /// </summary>
    public static string GetModelPath(LayoutModelVariant variant)
        => Path.Combine(ModelsRoot, GetModelFileName(variant));

    /// <summary>
    /// Ensure that the ONNX model for the specified variant exists locally, downloading it if needed.
    /// </summary>
    public static Task EnsureModelAsync(
        LayoutModelVariant variant,
        GithubReleaseOptions? options = null,
        Action<string>? logger = null,
        CancellationToken cancellationToken = default)
        => LayoutModelDownloader.EnsureVariantAsync(
            variant,
            ModelsRoot,
            options,
            logger,
            cancellationToken);

    /// <summary>
    /// Synchronous wrapper for <see cref="EnsureModelAsync"/>.
    /// </summary>
    public static void EnsureModel(
        LayoutModelVariant variant,
        GithubReleaseOptions? options = null,
        Action<string>? logger = null,
        CancellationToken cancellationToken = default)
        => LayoutModelDownloader.EnsureVariant(
            variant,
            ModelsRoot,
            options,
            logger,
            cancellationToken);

    /// <summary>
    /// Create <see cref="LayoutSdkOptions"/> that are backed by release-managed models.
    /// The model is downloaded (if missing) before returning the options.
    /// </summary>
    public static LayoutSdkOptions CreateOptions(
        LayoutModelVariant variant = LayoutModelVariant.Accurate,
        DocumentLanguage defaultLanguage = DocumentLanguage.English,
        bool validateModelPaths = false,
        GithubReleaseOptions? releaseOptions = null,
        Action<string>? logger = null,
        CancellationToken cancellationToken = default)
    {
        var options = releaseOptions ?? new GithubReleaseOptions();
        EnsureModel(variant, options, logger, cancellationToken);

        return new LayoutSdkOptions(
            onnxModelPath: GetModelPath(variant),
            defaultLanguage: defaultLanguage,
            validateModelPaths: validateModelPaths,
            modelVariant: variant,
            releaseOptions: options);
    }

    private static string ResolveModelsRoot()
    {
        var overrideDir = Environment.GetEnvironmentVariable(ModelsRootEnvironmentVariable);
        if (!string.IsNullOrWhiteSpace(overrideDir))
        {
            return Path.GetFullPath(overrideDir);
        }

        static string? TryGetFolder(Environment.SpecialFolder folder)
        {
            var candidate = Environment.GetFolderPath(folder);
            return string.IsNullOrWhiteSpace(candidate) ? null : candidate;
        }

        var baseDir =
            TryGetFolder(Environment.SpecialFolder.LocalApplicationData) ??
            TryGetFolder(Environment.SpecialFolder.ApplicationData) ??
            TryGetFolder(Environment.SpecialFolder.UserProfile);

        if (!string.IsNullOrEmpty(baseDir))
        {
            return Path.Combine(baseDir, "Docling", "LayoutSdk", "models");
        }

        // As a last resort fall back to the application base directory.
        return Path.Combine(AppContext.BaseDirectory, "docling-layoutsdk-models");
    }
}
