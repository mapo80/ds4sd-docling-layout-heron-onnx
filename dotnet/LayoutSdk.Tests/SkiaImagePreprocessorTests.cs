using LayoutSdk.Processing;
using SkiaSharp;
using System;
using Xunit;

namespace LayoutSdk.Tests;

public class SkiaImagePreprocessorTests
{
    [Fact]
    public void Preprocess_NullImage_Throws()
    {
        var preprocessor = new SkiaImagePreprocessor();
        Assert.Throws<ArgumentNullException>(() => preprocessor.Preprocess(null!));
    }

    [Fact]
    public void Preprocess_NormalizesPixels()
    {
        using var bitmap = new SKBitmap(2, 2);
        bitmap.Erase(SKColors.Black);
        bitmap.SetPixel(0, 0, new SKColor(255, 128, 0));
        bitmap.SetPixel(1, 0, new SKColor(0, 64, 255));
        bitmap.SetPixel(0, 1, new SKColor(32, 16, 8));
        bitmap.SetPixel(1, 1, new SKColor(255, 255, 255));

        var preprocessor = new SkiaImagePreprocessor();
        using var tensor = preprocessor.Preprocess(bitmap);

        Assert.Equal(2, tensor.Width);
        Assert.Equal(2, tensor.Height);
        Assert.Equal(3, tensor.Channels);

        var plane = tensor.Width * tensor.Height;
        var span = tensor.AsSpan();

        Assert.Equal(1d, span[0], 3);
        Assert.Equal(0d, span[1], 3);
        Assert.Equal(32d / 255d, span[2], 3);
        Assert.Equal(1d, span[3], 3);

        Assert.Equal(128d / 255d, span[plane + 0], 3);
        Assert.Equal(64d / 255d, span[plane + 1], 3);
        Assert.Equal(16d / 255d, span[plane + 2], 3);
        Assert.Equal(1d, span[plane + 3], 3);

        Assert.Equal(0d, span[2 * plane + 0], 3);
        Assert.Equal(1d, span[2 * plane + 1], 3);
        Assert.Equal(8d / 255d, span[2 * plane + 2], 3);
        Assert.Equal(1d, span[2 * plane + 3], 3);
    }
}
