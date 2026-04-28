// Design System / Theme Configuration
// Based on Aura theme by Josean Martinez & Custom Premium Themes

#[derive(Clone, Debug)]
pub struct Theme {
    pub name: &'static str,
    pub bg: &'static str,
    pub fg: &'static str,
    pub accent: &'static str,
    pub primary: &'static str,
    pub muted: &'static str,
    pub warning: &'static str,
    pub error: &'static str,
    pub success: &'static str,
    pub sidebar_width: &'static str,
    pub border_radius: &'static str,
}

impl Theme {
    // 1. Original Aura Theme
    pub fn aura() -> Self {
        Self {
            name: "aura",
            bg: "#011423",
            fg: "#CBE0F0",
            accent: "#47FF9C",
            primary: "#0FC5ED",
            muted: "#214969",
            warning: "#FFE073",
            error: "#E52E2E",
            success: "#44FFB1",
            sidebar_width: "240px",
            border_radius: "16px",
        }
    }

    // 2. Executive Premium Dark (Onyx & Gold)
    pub fn onyx() -> Self {
        Self {
            name: "onyx",
            bg: "#0B0C10",          // Deep void black
            fg: "#E0E2E6",          // Platinum off-white for low eye strain
            accent: "#D4AF37",      // Classic metallic gold
            primary: "#6366F1",     // Premium soft indigo
            muted: "#1F2937",       // Deep slate for subtle borders
            warning: "#FBBF24",     // Warm amber
            error: "#F43F5E",       // Soft rose/crimson
            success: "#10B981",     // Vibrant emerald
            sidebar_width: "250px", // Slightly wider for breathing room
            border_radius: "12px",  // Sharper, more modern corners
        }
    }

    // 3. Modern Premium Dark (Midnight Violet)
    pub fn midnight() -> Self {
        Self {
            name: "midnight",
            bg: "#0F0A19",      // Very dark violet
            fg: "#E2D8F0",      // Tinted lavender-white
            accent: "#00F0FF",  // Electric cyan
            primary: "#A855F7", // Bright purple
            muted: "#2D1B4E",   // Rich violet for muted backgrounds
            warning: "#F59E0B", // Orange/Amber
            error: "#EF4444",   // Bright red
            success: "#14B8A6", // Teal
            sidebar_width: "240px",
            border_radius: "16px",
        }
    }

    pub fn to_css_vars(&self) -> String {
        format!(
            "--bg:{};--fg:{};--accent:{};--primary:{};--muted:{};--warning:{};--error:{};--success:{}",
            self.bg,
            self.fg,
            self.accent,
            self.primary,
            self.muted,
            self.warning,
            self.error,
            self.success
        )
    }

    pub fn css(&self) -> String {
        let vars = self.to_css_vars();
        // Note: The CSS uses your original formatting, injecting the new theme variables.
        format!(
            r#"
:root{{ {vars} }}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Outfit',sans-serif;background:var(--bg);color:var(--fg);line-height:1.6;min-height:100vh}}
.layout{{display:flex;min-height:100vh}}
.sidebar{{width:{w};background:linear-gradient(180deg,rgba(33,73,105,.1),rgba(0,0,0,.3));border-right:1px solid var(--muted);padding:24px 16px}}
.logo{{font-size:24px;font-weight:700;color:var(--accent);margin-bottom:32px}}
.logo span{{color:var(--primary)}}
.nav{{display:flex;flex-direction:column;gap:4px}}
.nav-item{{padding:12px 16px;border-radius:8px;cursor:pointer;transition:.2s;font-size:14px;text-decoration:none;color:var(--fg);display:block}}
.nav-item:hover{{background:var(--muted);opacity:0.8}}
.nav-item.active{{background:linear-gradient(90deg,var(--muted),transparent);color:var(--accent);border-left:2px solid var(--accent)}}
.main{{flex:1;padding:32px}}
.title{{font-size:28px;font-weight:600;margin-bottom:4px}}
.subtitle{{font-size:14px;opacity:.6;margin-bottom:24px}}
.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:32px}}
.stat-card{{background:linear-gradient(135deg,var(--muted),transparent);border:1px solid var(--muted);border-radius:{r};padding:24px}}
.stat-value{{font-size:32px;font-weight:700;color:var(--accent);font-family:'JetBrains Mono',monospace}}
.stat-label{{font-size:12px;opacity:.6;text-transform:uppercase;margin-top:8px}}
.card{{background:linear-gradient(135deg,rgba(255,255,255,.02),rgba(0,0,0,.2));border:1px solid var(--muted);border-radius:{r};overflow:hidden}}
.card-header{{padding:20px 24px;border-bottom:1px solid var(--muted);font-weight:600}}
.table{{width:100%;border-collapse:collapse}}
.table th{{padding:16px 24px;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:1px;opacity:.5;border-bottom:1px solid var(--muted)}}
.table td{{padding:16px 24px;border-bottom:1px solid var(--muted)}}
.table tr:hover{{background:rgba(255,255,255,.02)}}
.tier{{display:inline-block;padding:4px 10px;border-radius:20px;font-size:11px;font-weight:500}}
.tier-sensory{{background:rgba(15,197,237,.2);color:#0FC5ED}}
.tier-working{{background:rgba(162,119,255,.2);color:#a277ff}}
.tier-episodic{{background:rgba(71,255,156,.2);color:#47FF9C}}
.tier-semantic{{background:rgba(255,224,115,.2);color:#FFE073}}
.tier-procedural{{background:rgba(229,46,46,.2);color:#E52E2E}}
.metric{{font-family:'JetBrains Mono',monospace;font-size:14px}}
.empty{{padding:48px;text-align:center;opacity:.5}}
code{{font-family:'JetBrains Mono',monospace;background:var(--muted);padding:2px 8px;border-radius:4px;font-size:12px}}
"#,
            w = self.sidebar_width,
            r = self.border_radius
        )
    }
}

pub fn get_theme(name: &str) -> Theme {
    match name {
        "aura" => Theme::aura(),
        "onyx" => Theme::onyx(),
        "midnight" => Theme::midnight(),
        _ => Theme::onyx(), // Defaulting to the premium onyx theme
    }
}

