using System;

namespace LayoutSdk.Processing;

public sealed class ImageTensor : IDisposable
{
    private readonly float[] _buffer;
    private bool _disposed;

    private ImageTensor(float[] buffer, int width, int height, int channels)
    {
        _buffer = buffer;
        Width = width;
        Height = height;
        Channels = channels;
        Length = width * height * channels;
    }

    public int Width { get; }

    public int Height { get; }

    public int Channels { get; }

    public int Length { get; }

    public static ImageTensor Rent(int width, int height, int channels)
    {
        if (width <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(width));
        }

        if (height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(height));
        }

        if (channels <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(channels));
        }

        var required = checked(width * height * channels);
        var buffer = new float[required];
        return new ImageTensor(buffer, width, height, channels);
    }

    public Span<float> AsSpan() => _buffer.AsSpan(0, Length);

    public float[] Buffer => _buffer;

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
    }
}
