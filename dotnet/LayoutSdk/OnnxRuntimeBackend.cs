using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Linq;
using LayoutSdk.Inference;
using LayoutSdk.Processing;

namespace LayoutSdk;

internal sealed class OnnxRuntimeBackend : ILayoutBackend, IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;

    public OnnxRuntimeBackend(string modelPath)
    {
        using var options = CreateSessionOptions();
        _session = new InferenceSession(modelPath, options);
        _inputName = _session.InputMetadata.Keys.First();
    }

    public LayoutBackendResult Infer(ImageTensor tensor)
    {
        var dense = new DenseTensor<float>(tensor.Buffer, new[] { 1, tensor.Channels, tensor.Height, tensor.Width });
        using var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(_inputName, dense) });
        return new LayoutBackendResult(new System.Collections.Generic.List<BoundingBox>());
    }

    public void Dispose() => _session.Dispose();

    private static SessionOptions CreateSessionOptions()
    {
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            IntraOpNumThreads = 0,
            InterOpNumThreads = 1
        };

        return options;
    }
}
