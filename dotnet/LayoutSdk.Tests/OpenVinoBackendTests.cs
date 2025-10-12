using System;
using LayoutSdk.Processing;
using Xunit;

namespace LayoutSdk.Tests;

public class OpenVinoBackendTests
{
   [Fact]
   public void Constructor_ThrowsNotSupportedException()
   {
       Assert.Throws<NotSupportedException>(() => new OpenVinoBackend("missing.xml", "missing.bin"));
   }
}
