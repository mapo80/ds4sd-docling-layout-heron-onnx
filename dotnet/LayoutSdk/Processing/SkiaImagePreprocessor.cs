using SkiaSharp;
using System;

namespace LayoutSdk.Processing;

public sealed class SkiaImagePreprocessor : IImagePreprocessor
{
    internal const int Channels = 3;
    internal const int ModelInputSize = 640;

    public ImageTensor Preprocess(SKBitmap image)
    {
        if (image is null)
        {
            throw new ArgumentNullException(nameof(image));
        }

        // Step 1: Resize to the model input dimensions (RT-DETR uses anisotropic resize without padding)
        using var resizedBitmap = ResizeToModelSize(image, ModelInputSize);

        // Step 2: Convert to tensor with CLIP/ViT normalization (same as HuggingFace)
        var tensor = ImageTensor.Rent(ModelInputSize, ModelInputSize, Channels);
        var span = tensor.AsSpan();

        var pixels = resizedBitmap.Pixels;
        for (var i = 0; i < pixels.Length; i++)
        {
            var color = pixels[i];

            // Convert to float [0,1] range
            var r = color.Red / 255f;
            var g = color.Green / 255f;
            var b = color.Blue / 255f;

            // Apply CLIP/ViT normalization (same as HuggingFace docling-layout-heron)
            span[i] = r;
            span[i + pixels.Length] = g;
            span[i + 2 * pixels.Length] = b;
        }

        return tensor;
    }

    private static SKBitmap ResizeToModelSize(SKBitmap original, int targetSize)
    {
        var resized = new SKBitmap(new SKImageInfo(targetSize, targetSize, original.ColorType, original.AlphaType));
        var sampling = new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.None);
        if (!original.ScalePixels(resized, sampling))
        {
            throw new InvalidOperationException("Failed to resize image to model input size.");
        }
        return resized;
    }
}
