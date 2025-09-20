using SkiaSharp;
using System;

namespace LayoutSdk.Processing;

public sealed class SkiaImagePreprocessor : IImagePreprocessor
{
    private const int Channels = 3;

    public ImageTensor Preprocess(SKBitmap image)
    {
        if (image is null)
        {
            throw new ArgumentNullException(nameof(image));
        }

        var tensor = ImageTensor.Rent(image.Width, image.Height, Channels);
        var span = tensor.AsSpan();
        var plane = image.Width * image.Height;

        var pixels = image.Pixels;
        for (var i = 0; i < plane; i++)
        {
            var color = pixels[i];
            span[i] = color.Red / 255f;
            span[i + plane] = color.Green / 255f;
            span[i + 2 * plane] = color.Blue / 255f;
        }

        return tensor;
    }
}
