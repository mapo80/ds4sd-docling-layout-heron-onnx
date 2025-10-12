using System;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using LayoutSdk.Inference;
using LayoutSdk.Processing;

namespace LayoutSdk;

internal sealed class OnnxRuntimeBackend : ILayoutBackend, IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;

    public OnnxRuntimeBackend(string modelPath)
    {
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            IntraOpNumThreads = 0,
            InterOpNumThreads = 0
        };

        _session = new InferenceSession(modelPath, options);
        _inputName = _session.InputMetadata.Keys.First();
    }

    public LayoutBackendResult Infer(ImageTensor tensor)
    {
        ArgumentNullException.ThrowIfNull(tensor);

        var dense = new DenseTensor<float>(
            tensor.Buffer.AsMemory(0, tensor.Length),
            new[] { 1, tensor.Channels, tensor.Height, tensor.Width });

        var input = NamedOnnxValue.CreateFromTensor(_inputName, dense);
        using var results = _session.Run(new[] { input });
        (input as IDisposable)?.Dispose();

        return ParseOutputs(results);
    }

    public void Dispose() => _session.Dispose();

    private static LayoutBackendResult ParseOutputs(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results)
    {
        var logitsOutput = results.FirstOrDefault(r => r.Name == "logits")
                           ?? throw new InvalidOperationException("ONNX outputs do not contain 'logits'.");
        var boxesOutput = results.FirstOrDefault(r => r.Name == "pred_boxes")
                          ?? throw new InvalidOperationException("ONNX outputs do not contain 'pred_boxes'.");

        var logits = logitsOutput.AsTensor<float>();
        var boxes = boxesOutput.AsTensor<float>();

        return new LayoutBackendResult(
            logits.ToArray(),
            logits.Dimensions.ToArray(),
            boxes.ToArray(),
            boxes.Dimensions.ToArray());
    }
}
