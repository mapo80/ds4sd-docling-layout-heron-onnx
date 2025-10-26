using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace LayoutSdk.Assets;

/// <summary>
/// Ensures Docling Layout Heron runtime assets are available locally, downloading them from GitHub if needed.
/// </summary>
public static class LayoutModelDownloader
{
    private sealed record ReleaseAsset(
        string FileName,
        string ReleaseName,
        string Md5)
    {
        public string BuildUrl(GithubReleaseOptions options) => options.BuildAssetUrl(ReleaseName);
    }

    private static readonly IReadOnlyDictionary<string, ReleaseAsset> Assets =
        new Dictionary<string, ReleaseAsset>(StringComparer.OrdinalIgnoreCase)
        {
            ["heron-converted.onnx"] = new(
                FileName: "heron-converted.onnx",
                ReleaseName: "heron-converted.onnx",
                Md5: "afc761f57bef639172f00fe0b1bc137a"),
            ["heron-optimized.onnx"] = new(
                FileName: "heron-optimized.onnx",
                ReleaseName: "heron-optimized.onnx",
                Md5: "5a941c78cdfd26c4e8f8788496dc5ed2"),
            ["heron-optimized-fp16.onnx"] = new(
                FileName: "heron-optimized-fp16.onnx",
                ReleaseName: "heron-optimized-fp16.onnx",
                Md5: "5725b4fc4393715a0bba4309e3ba9df9"),
            ["heron-optimized.ort"] = new(
                FileName: "heron-optimized.ort",
                ReleaseName: "heron-optimized.ort",
                Md5: "59c7c241455f4994548cf74fa8b4e652"),
            ["heron-optimized.with_runtime_opt.ort"] = new(
                FileName: "heron-optimized.with_runtime_opt.ort",
                ReleaseName: "heron-optimized.with_runtime_opt.ort",
                Md5: "4f95929544d98c76a0c99674627f716e"),
        };

    private static readonly HttpClient Http;

    static LayoutModelDownloader()
    {
        Http = new HttpClient
        {
            Timeout = TimeSpan.FromMinutes(10)
        };
        Http.DefaultRequestHeaders.UserAgent.Add(new ProductInfoHeaderValue("Docling.LayoutSdk", "1.0"));
    }

    private static readonly ConcurrentDictionary<string, SemaphoreSlim> Locks = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Ensure that the requested model file exists locally, downloading it if necessary.
    /// </summary>
    public static async Task EnsureModelAsync(
        string modelPath,
        GithubReleaseOptions? options = null,
        Action<string>? logger = null,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
        {
            throw new ArgumentException("Model path must be provided.", nameof(modelPath));
        }

        var fileName = Path.GetFileName(modelPath);
        if (!Assets.TryGetValue(fileName, out var asset))
        {
            // Unrecognized file name: assume caller manages it manually.
            return;
        }

        var fullPath = Path.GetFullPath(modelPath);
        var semaphore = Locks.GetOrAdd(fullPath, _ => new SemaphoreSlim(1, 1));
        await semaphore.WaitAsync(cancellationToken).ConfigureAwait(false);

        try
        {
            if (File.Exists(fullPath) && VerifyChecksum(fullPath, asset.Md5))
            {
                logger?.Invoke($"✓ {fileName} già presente (checksum verificato)");
                return;
            }

            var downloadOptions = options ?? new GithubReleaseOptions();
            var assetUrl = asset.BuildUrl(downloadOptions);
            Directory.CreateDirectory(Path.GetDirectoryName(fullPath)!);

            logger?.Invoke($"↓ Download {fileName} da {assetUrl}");
            using var response = await Http.GetAsync(assetUrl, HttpCompletionOption.ResponseHeadersRead, cancellationToken)
                .ConfigureAwait(false);
            response.EnsureSuccessStatusCode();

            await using (var target = File.Open(fullPath, FileMode.Create, FileAccess.Write, FileShare.None))
            {
                await response.Content.CopyToAsync(target, cancellationToken).ConfigureAwait(false);
            }

            if (!VerifyChecksum(fullPath, asset.Md5))
            {
                throw new InvalidOperationException($"Checksum md5 errato per {fileName}. Atteso {asset.Md5}.");
            }

            var sizeMb = new FileInfo(fullPath).Length / (1024d * 1024d);
            logger?.Invoke($"✓ {fileName} disponibile ({sizeMb:F1} MB)");
        }
        finally
        {
            semaphore.Release();
        }
    }

    /// <summary>
    /// Synchronous wrapper around <see cref="EnsureModelAsync"/>.
    /// </summary>
    public static void EnsureModel(
        string modelPath,
        GithubReleaseOptions? options = null,
        Action<string>? logger = null,
        CancellationToken cancellationToken = default)
    {
        EnsureModelAsync(modelPath, options, logger, cancellationToken).GetAwaiter().GetResult();
    }

    private static bool VerifyChecksum(string path, string expectedMd5)
    {
        var checksum = ComputeMd5(path);
        return checksum.Equals(expectedMd5, StringComparison.OrdinalIgnoreCase);
    }

    private static string ComputeMd5(string path)
    {
        using var stream = File.OpenRead(path);
        var hash = MD5.HashData(stream);
        var builder = new StringBuilder(hash.Length * 2);
        foreach (var b in hash)
        {
            builder.Append(b.ToString("x2"));
        }

        return builder.ToString();
    }
}
