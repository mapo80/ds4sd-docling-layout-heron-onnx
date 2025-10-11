using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
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
        ArgumentNullException.ThrowIfNull(tensor);

        var dense = new DenseTensor<float>(
            tensor.Buffer.AsMemory(0, tensor.Length),
            new[] { 1, tensor.Channels, tensor.Height, tensor.Width });

        var input = NamedOnnxValue.CreateFromTensor(_inputName, dense);
        using var results = _session.Run(new[] { input });
        (input as IDisposable)?.Dispose();

        var boxes = ParseOutputs(results, tensor.Width, tensor.Height);
        return new LayoutBackendResult(boxes);
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

    private static IReadOnlyList<BoundingBox> ParseOutputs(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results,
        int imageWidth,
        int imageHeight)
    {
        var logitsOutput = results.FirstOrDefault(r => r.Name == "logits");
        var boxesOutput = results.FirstOrDefault(r => r.Name == "pred_boxes");

        if (logitsOutput == null || boxesOutput == null)
        {
            return Array.Empty<BoundingBox>();
        }

        var logits = logitsOutput.AsTensor<float>();
        var boxes = boxesOutput.AsTensor<float>();

        if (logits.Rank != 3 || boxes.Rank != 3)
        {
            return Array.Empty<BoundingBox>();
        }

        var labelMap = new Dictionary<int, string>
        {
            { 0, "Background" },
            { 1, "Caption" },
            { 2, "Footnote" },
            { 3, "Formula" },
            { 4, "List-item" },
            { 5, "Page-footer" },
            { 6, "Page-header" },
            { 7, "Picture" },
            { 8, "Section-header" },
            { 9, "Table" },
            { 10, "Text" },
            { 11, "Title" },
            { 12, "Document Index" },
            { 13, "Code" },
            { 14, "Checkbox-Selected" },
            { 15, "Checkbox-Unselected" },
            { 16, "Form" },
            { 17, "Key-Value Region" }
        };

        const float scoreThreshold = 0.3f;

        var detections = new List<BoundingBox>();
        var numQueries = logits.Dimensions[1];
        var numClasses = logits.Dimensions[2];

        for (var q = 0; q < numQueries; q++)
        {
            var maxProb = float.MinValue;
            var maxClass = 0;

            for (var c = 0; c < numClasses; c++)
            {
                var prob = logits[0, q, c];
                if (prob > maxProb)
                {
                    maxProb = prob;
                    maxClass = c;
                }
            }

            var score = 1f / (1f + MathF.Exp(-maxProb));
            if (maxClass == 0 || score < scoreThreshold)
            {
                continue;
            }

            var cx = boxes[0, q, 0];
            var cy = boxes[0, q, 1];
            var w = boxes[0, q, 2];
            var h = boxes[0, q, 3];

            var width = w * imageWidth;
            var height = h * imageHeight;
            var x = (cx * imageWidth) - (width / 2f);
            var y = (cy * imageHeight) - (height / 2f);

            x = Math.Clamp(x, 0f, Math.Max(0f, imageWidth - width));
            y = Math.Clamp(y, 0f, Math.Max(0f, imageHeight - height));
            width = Math.Clamp(width, 0f, imageWidth - x);
            height = Math.Clamp(height, 0f, imageHeight - y);

            var label = labelMap.TryGetValue(maxClass, out var mapped) ? mapped : "Unknown";
            detections.Add(new BoundingBox(x, y, width, height, label));
        }

        return detections;
    }
}
