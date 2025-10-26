using System;

namespace LayoutSdk.Assets;

/// <summary>
/// Options that identify the GitHub release containing Docling Layout Heron assets.
/// </summary>
public sealed record GithubReleaseOptions(
    string Repository = "mapo80/ds4sd-docling-layout-heron-onnx",
    string Tag = "models-2025-09-19")
{
    /// <summary>
    /// Build the base download URL for the configured release.
    /// </summary>
    public string BuildBaseUrl() => $"https://github.com/{Repository}/releases/download/{Tag}";

    /// <summary>
    /// Build a full asset URL for a specific release file.
    /// </summary>
    public string BuildAssetUrl(string assetName)
    {
        if (string.IsNullOrWhiteSpace(assetName))
        {
            throw new ArgumentException("Asset name must be provided.", nameof(assetName));
        }

        return $"{BuildBaseUrl().TrimEnd('/')}/{assetName}";
    }
}
