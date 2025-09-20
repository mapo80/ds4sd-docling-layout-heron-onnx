using LayoutSdk;
using LayoutSdk.Processing;
using System;
using Xunit;

namespace LayoutSdk.Tests;

public class OpenVinoBackendTests
{
    [Fact]
    public void Constructor_InvalidArguments_Throws()
    {
        Assert.Throws<ArgumentException>(() => new OpenVinoBackend("", "weights.bin"));
        Assert.Throws<ArgumentException>(() => new OpenVinoBackend("model.xml", ""));
    }

    [Fact]
    public void Constructor_UsesFactory()
    {
        var executor = new FakeExecutor();
        var backend = new OpenVinoBackend("model.xml", "weights.bin", (_, _) => executor);
        backend.Dispose();
        Assert.True(executor.Disposed);
    }

    private sealed class FakeExecutor : IOpenVinoExecutor
    {
        public bool Disposed { get; private set; }
        public ImageTensor? LastTensor { get; private set; }

        public void Infer(ImageTensor tensor)
        {
            LastTensor = tensor;
        }

        public void Dispose()
        {
            Disposed = true;
        }
    }

    [Fact]
    public void Infer_DelegatesToExecutor()
    {
        var executor = new FakeExecutor();
        var backend = new OpenVinoBackend(executor);
        using var tensor = ImageTensor.Rent(1, 1, 1);

        var result = backend.Infer(tensor);

        Assert.NotNull(result);
        Assert.Same(tensor, executor.LastTensor);

        backend.Dispose();
        Assert.True(executor.Disposed);
    }

    [Fact]
    public void CopyNativeBinariesNearExecutable_CopiesFiles()
    {
        var original = Environment.GetEnvironmentVariable("NUGET_PACKAGES");
        var tempRoot = System.IO.Path.Combine(System.IO.Path.GetTempPath(), System.Guid.NewGuid().ToString("N"));
        var nativeDir = System.IO.Path.Combine(tempRoot, "openvino.runtime.ubuntu.24-x86_64", "1.0.0", "runtimes", "ubuntu.24-x86_64", "native");
        System.IO.Directory.CreateDirectory(nativeDir);
        var src = System.IO.Path.Combine(nativeDir, "libdummy.so");
        System.IO.File.WriteAllText(src, string.Empty);

        Environment.SetEnvironmentVariable("NUGET_PACKAGES", tempRoot);

        try
        {
            var method = typeof(OpenVinoBackend).GetMethod("CopyNativeBinariesNearExecutable", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);
            method!.Invoke(null, null);

            var dest = System.IO.Path.Combine(AppContext.BaseDirectory, "libdummy.so");
            Assert.True(System.IO.File.Exists(dest));
            System.IO.File.Delete(dest);
        }
        finally
        {
            Environment.SetEnvironmentVariable("NUGET_PACKAGES", original);
            System.IO.Directory.Delete(tempRoot, recursive: true);
        }
    }

    [Fact]
    public void CopyNativeBinariesNearExecutable_MissingPackagesDirectory_NoOp()
    {
        var original = Environment.GetEnvironmentVariable("NUGET_PACKAGES");
        var tempRoot = System.IO.Path.Combine(System.IO.Path.GetTempPath(), System.Guid.NewGuid().ToString("N"));
        Environment.SetEnvironmentVariable("NUGET_PACKAGES", tempRoot);

        try
        {
            var method = typeof(OpenVinoBackend).GetMethod("CopyNativeBinariesNearExecutable", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);
            method!.Invoke(null, null);
        }
        finally
        {
            Environment.SetEnvironmentVariable("NUGET_PACKAGES", original);
        }
    }
}
