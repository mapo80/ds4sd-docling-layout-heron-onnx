using LayoutSdk;
using LayoutSdk.Configuration;
using LayoutSdk.Factories;
using LayoutSdk.Inference;
using LayoutSdk.Processing;
using Xunit;

namespace LayoutSdk.Tests;

public class LayoutBackendFactoryTests
{
    [Fact]
    public void Constructor_NullOptions_Throws()
    {
        Assert.Throws<System.ArgumentNullException>(() => new LayoutBackendFactory(null!));
    }

    private sealed class StubBackend : ILayoutBackend, IDisposable
    {
        public bool Disposed { get; private set; }

        public LayoutBackendResult Infer(ImageTensor tensor) => new(new System.Collections.Generic.List<BoundingBox>());

        public void Dispose()
        {
            Disposed = true;
        }
    }

    [Fact]
    public void Create_Onnx_ReturnsBackend()
    {
        var options = new LayoutSdkOptions("onnx", "ort", new OpenVinoModelOptions("xml", "bin"));
        var stub = new StubBackend();
        var factory = new LayoutBackendFactory(options, _ => stub);
        var backend = factory.Create(LayoutRuntime.Onnx);
        Assert.Same(stub, backend);
    }

    [Fact]
    public void Create_OpenVino_ReturnsBackend()
    {
        var options = new LayoutSdkOptions("onnx", "ort", new OpenVinoModelOptions("xml", "bin"));
        var stub = new StubBackend();
        var factory = new LayoutBackendFactory(options, onnxFactory: _ => stub, openVinoFactory: (_, _) => stub);
        var backend = factory.Create(LayoutRuntime.OpenVino);
        Assert.Same(stub, backend);
    }

    [Fact]
    public void Create_Ort_ReturnsBackend()
    {
        var options = new LayoutSdkOptions("onnx", "ort", new OpenVinoModelOptions("xml", "bin"));
        var stub = new StubBackend();
        var factory = new LayoutBackendFactory(options, ortFactory: _ => stub);
        var backend = factory.Create(LayoutRuntime.Ort);
        Assert.Same(stub, backend);
    }

    [Fact]
    public void Create_OrtWithoutPath_Throws()
    {
        var options = new LayoutSdkOptions("onnx", null, new OpenVinoModelOptions("xml", "bin"));
        var factory = new LayoutBackendFactory(options);
        Assert.Throws<InvalidOperationException>(() => factory.Create(LayoutRuntime.Ort));
    }

    [Fact]
    public void Create_InvalidRuntime_Throws()
    {
        var options = new LayoutSdkOptions("onnx", "ort", new OpenVinoModelOptions("xml", "bin"));
        var factory = new LayoutBackendFactory(options);
        Assert.Throws<System.ArgumentOutOfRangeException>(() => factory.Create((LayoutRuntime)999));
    }
}
