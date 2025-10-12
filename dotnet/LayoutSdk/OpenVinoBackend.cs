using System;
using LayoutSdk.Inference;
using LayoutSdk.Processing;

namespace LayoutSdk;

internal sealed class OpenVinoBackend : ILayoutBackend, IDisposable
{
    public OpenVinoBackend(string modelXmlPath, string weightsBinPath)
    {
        throw new NotSupportedException("OpenVinoBackend has been removed. Use only ONNX runtime.");
    }

    public LayoutBackendResult Infer(ImageTensor tensor)
    {
        throw new NotSupportedException("OpenVinoBackend has been removed. Use only ONNX runtime.");
    }

    public void Dispose()
    {
        // No resources to dispose
    }
}
