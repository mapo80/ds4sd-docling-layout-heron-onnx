using OpenVinoSharp;
using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using LayoutSdk.Inference;
using LayoutSdk.Processing;

namespace LayoutSdk;

internal sealed class OpenVinoBackend : ILayoutBackend, IDisposable
{
    private readonly Core _core;
    private readonly Model _model;
    private readonly CompiledModel _compiled;
    private readonly InferRequest _request;
    private readonly string _inputName;

    public OpenVinoBackend(string modelXmlPath, string weightsBinPath)
    {
        if (string.IsNullOrWhiteSpace(modelXmlPath))
        {
            throw new ArgumentException("Model XML path must be provided", nameof(modelXmlPath));
        }

        if (string.IsNullOrWhiteSpace(weightsBinPath))
        {
            throw new ArgumentException("Weights BIN path must be provided", nameof(weightsBinPath));
        }

        CopyNativeBinariesNearExecutable();

        _core = new Core();
        _model = _core.read_model(modelXmlPath, weightsBinPath);
        _compiled = _core.compile_model(_model, "CPU");
        _request = _compiled.create_infer_request();
        _inputName = _model.inputs()[0].get_any_name();
    }

    public LayoutBackendResult Infer(ImageTensor tensor)
    {
        using var inputTensor = new Tensor(new Shape(new long[] { 1, tensor.Channels, tensor.Height, tensor.Width }), tensor.Buffer);
        _request.set_tensor(_inputName, inputTensor);
        _request.infer();
        return new LayoutBackendResult(new List<BoundingBox>());
    }

    public void Dispose()
    {
        _request.Dispose();
        _compiled.Dispose();
        _model.Dispose();
        _core.Dispose();
    }

    private static void CopyNativeBinariesNearExecutable()
    {
        var nuget = Environment.GetEnvironmentVariable("NUGET_PACKAGES")
                   ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".nuget", "packages");
        var runtimes = Path.Combine(nuget, "openvino.runtime.ubuntu.24-x86_64");
        if (!Directory.Exists(runtimes))
        {
            return;
        }

        var latest = Directory.GetDirectories(runtimes).OrderByDescending(p => p).FirstOrDefault();
        if (latest == null)
        {
            return;
        }

        var native = Path.Combine(latest, "runtimes", "ubuntu.24-x86_64", "native");
        if (!Directory.Exists(native))
        {
            return;
        }

        var execDir = AppContext.BaseDirectory;
        foreach (var src in Directory.GetFiles(native))
        {
            var dest = Path.Combine(execDir, Path.GetFileName(src));
            if (!File.Exists(dest))
            {
                File.Copy(src, dest);
            }

            try
            {
                System.Runtime.InteropServices.NativeLibrary.Load(dest);
            }
            catch
            {
            }
        }
    }
}
