using LayoutSdk;
using LayoutSdk.Processing;
using System;
using System.IO;
using Xunit;

namespace LayoutSdk.Tests;

public class OnnxRuntimeBackendTests
{
    [Fact]
    public void Constructor_InvalidPath_Throws()
    {
        Assert.Throws<Microsoft.ML.OnnxRuntime.OnnxRuntimeException>(() => new OnnxRuntimeBackend("missing.onnx"));
    }

    [Fact]
    public void Infer_RunsSuccessfully()
    {
        var path = TestModelFiles.CreateOnnxModelFile();
        try
        {
            using var backend = new OnnxRuntimeBackend(path);
            using var tensor = ImageTensor.Rent(2, 2, 4);
            var result = backend.Infer(tensor);
            Assert.NotNull(result);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
