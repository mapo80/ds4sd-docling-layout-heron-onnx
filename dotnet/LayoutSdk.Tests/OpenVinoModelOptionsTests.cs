using LayoutSdk.Configuration;
using System;
using System.IO;
using Xunit;

namespace LayoutSdk.Tests;

public class OpenVinoModelOptionsTests
{
    [Fact]
    public void EnsureFilesExist_Succeeds()
    {
        var xml = Path.GetTempFileName();
        var bin = Path.GetTempFileName();
        try
        {
            var options = new OpenVinoModelOptions(xml, bin);
            options.EnsureFilesExist();
        }
        finally
        {
            File.Delete(xml);
            File.Delete(bin);
        }
    }
}
