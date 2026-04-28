class CognimemServer < Formula
  desc "AI-powered memory system with associative recall and code graph understanding"
  homepage "https://github.com/cognimem/cognimem-server"
  url "https://github.com/cognimem/cognimem-server/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "TODO"
  license "MIT"
  version "0.1.0"

  depends_on "rust" => :build

  def install
    system "cargo", "build", "--release", "--bin", "cognimem-server"
    bin.install "target/release/cognimem-server"
  end

  def post_install
    (var/"cognimem-server").mkpath
  end

  plist_options :startup_name => "cognimem-server"

  def plist
    <<~EOS
      <?xml version="1.0" encoding="UTF-8"?>
      <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
      <plist version="1.0">
        <dict>
          <key>Label</key>
          <string>cognimem-server</string>
          <key>ProgramArguments</key>
          <array>
            <string>#{bin}/cognimem-server</string>
            <string>--data-path</string>
            <string>#{var}/cognimem-server/data</string>
          </array>
          <key>RunAtLoad</key>
          <true/>
        </dict>
      </plist>
    EOS
  end

  test do
    system "#{bin}/cognimem-server", "--version"
  end
end