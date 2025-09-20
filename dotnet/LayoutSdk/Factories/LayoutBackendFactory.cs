using System;
using LayoutSdk.Configuration;

namespace LayoutSdk.Factories;

public sealed class LayoutBackendFactory : ILayoutBackendFactory
{
    private readonly LayoutSdkOptions _options;
    private readonly Func<string, ILayoutBackend> _onnxFactory;
    private readonly Func<string, string, ILayoutBackend> _openVinoFactory;

    public LayoutBackendFactory(
        LayoutSdkOptions options,
        Func<string, ILayoutBackend>? onnxFactory = null,
        Func<string, string, ILayoutBackend>? openVinoFactory = null)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _options.EnsureModelPaths();
        _onnxFactory = onnxFactory ?? (path => new OnnxRuntimeBackend(path));
        _openVinoFactory = openVinoFactory ?? ((xml, bin) => new OpenVinoBackend(xml, bin));
    }

    public ILayoutBackend Create(LayoutRuntime runtime) => runtime switch
    {
        LayoutRuntime.OnnxRuntime => _onnxFactory(_options.OnnxModelPath),
        LayoutRuntime.OpenVino => _openVinoFactory(
            _options.OpenVino.ModelXmlPath,
            _options.OpenVino.WeightsBinPath),
        _ => throw new ArgumentOutOfRangeException(nameof(runtime), runtime, null)
    };
}
