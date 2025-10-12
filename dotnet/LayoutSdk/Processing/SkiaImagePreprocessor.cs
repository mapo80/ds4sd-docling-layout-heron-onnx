using SkiaSharp;
using System;

namespace LayoutSdk.Processing;

public sealed class SkiaImagePreprocessor : IImagePreprocessor
{
    private const int Channels = 3;
    private const int ModelInputSize = 640;

    // HuggingFace ImageNet normalization values for CLIP/ViT models
    private const float MeanR = 0.48145466f;
    private const float MeanG = 0.4578275f;
    private const float MeanB = 0.40821073f;
    private const float StdR = 0.26862954f;
    private const float StdG = 0.26130258f;
    private const float StdB = 0.27577711f;

    public ImageTensor Preprocess(SKBitmap image)
    {
        if (image is null)
        {
            throw new ArgumentNullException(nameof(image));
        }

        // Step 1: Resize maintaining aspect ratio with padding (letterboxing)
        var resizedBitmap = ResizeWithPadding(image, ModelInputSize);

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
            span[i] = (r - MeanR) / StdR;
            span[i + pixels.Length] = (g - MeanG) / StdG;
            span[i + 2 * pixels.Length] = (b - MeanB) / StdB;
        }

        return tensor;
    }

    private static SKBitmap ResizeWithPadding(SKBitmap original, int targetSize)
    {
        // Calculate scaling to maintain aspect ratio
        var scale = Math.Min((float)targetSize / original.Width, (float)targetSize / original.Height);

        var scaledWidth = (int)Math.Round(original.Width * scale);
        var scaledHeight = (int)Math.Round(original.Height * scale);

        // Create new bitmap with target size
        var resized = new SKBitmap(targetSize, targetSize, SKColorType.Rgba8888, SKAlphaType.Premul);

        using var canvas = new SKCanvas(resized);
        // Use WHITE background like HuggingFace CLIP models (instead of black)
        canvas.Clear(new SKColor(255, 255, 255, 255));

        // Calculate padding to center the image
        var offsetX = (targetSize - scaledWidth) / 2;
        var offsetY = (targetSize - scaledHeight) / 2;

        // Draw the scaled image centered
        var destRect = SKRect.Create(offsetX, offsetY, scaledWidth, scaledHeight);
        canvas.DrawBitmap(original, destRect);

        return resized;
    }
}
