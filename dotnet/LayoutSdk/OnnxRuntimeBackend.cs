using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Linq;
using LayoutSdk.Inference;
using LayoutSdk.Processing;

namespace LayoutSdk;

internal enum OnnxModelFormat
{
    Onnx,
    Ort
}

internal sealed class OnnxRuntimeBackend : ILayoutBackend, IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;

    public OnnxRuntimeBackend(string modelPath, OnnxModelFormat format = OnnxModelFormat.Onnx)
    {
        using var options = CreateSessionOptions(format);
        _session = new InferenceSession(modelPath, options);
        _inputName = _session.InputMetadata.Keys.First();
    }

    public LayoutBackendResult Infer(ImageTensor tensor)
    {
        var dense = new DenseTensor<float>(
            tensor.Buffer.AsMemory(0, tensor.Length),
            new[] { 1, tensor.Channels, tensor.Height, tensor.Width });
        using var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(_inputName, dense) });
        return new LayoutBackendResult(new System.Collections.Generic.List<BoundingBox>());
    }

    public void Dispose() => _session.Dispose();

    private static SessionOptions CreateSessionOptions(OnnxModelFormat format)
    {
        var options = new SessionOptions
        {
            GraphOptimizationLevel = format == OnnxModelFormat.Ort
                ? GraphOptimizationLevel.ORT_DISABLE_ALL
                : GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            IntraOpNumThreads = 0,
            InterOpNumThreads = 1
        };

        return options;
    }
}
