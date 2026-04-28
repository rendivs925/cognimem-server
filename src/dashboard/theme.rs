use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct Theme {
    pub name: String,
    pub bg: String,
    pub fg: String,
    pub accent: String,
    pub primary: String,
    pub surface: String,
    pub border: String,
    pub hover: String,
    pub active_bg: String,
    pub sidebar_bg: String,
    pub warning: String,
    pub error: String,
    pub success: String,
    pub radius: String,
    pub sidebar_width: String,
}

impl Theme {
    pub fn aura() -> Self {
        Self {
            name: "aura".into(),
            bg: "#011423".into(),
            fg: "#CBE0F0".into(),
            accent: "#47FF9C".into(),
            primary: "#0FC5ED".into(),
            surface: "#0A1E2F".into(),
            border: "#1A3A5C".into(),
            hover: "rgba(255,255,255,0.04)".into(),
            active_bg: "rgba(71,255,156,0.1)".into(),
            sidebar_bg: "#011423".into(),
            warning: "#FFE073".into(),
            error: "#E52E2E".into(),
            success: "#44FFB1".into(),
            radius: "12px".into(),
            sidebar_width: "240px".into(),
        }
    }

    pub fn onyx() -> Self {
        Self {
            name: "onyx".into(),
            bg: "#0B0C10".into(),
            fg: "#E0E2E6".into(),
            accent: "#6366F1".into(),
            primary: "#818CF8".into(),
            surface: "#14151A".into(),
            border: "#1F2937".into(),
            hover: "rgba(255,255,255,0.03)".into(),
            active_bg: "rgba(99,102,241,0.12)".into(),
            sidebar_bg: "#0B0C10".into(),
            warning: "#FBBF24".into(),
            error: "#F43F5E".into(),
            success: "#10B981".into(),
            radius: "10px".into(),
            sidebar_width: "240px".into(),
        }
    }

    pub fn midnight() -> Self {
        Self {
            name: "midnight".into(),
            bg: "#0F0A19".into(),
            fg: "#E2D8F0".into(),
            accent: "#A855F7".into(),
            primary: "#C084FC".into(),
            surface: "#1A1028".into(),
            border: "#2D1B4E".into(),
            hover: "rgba(255,255,255,0.04)".into(),
            active_bg: "rgba(168,85,247,0.12)".into(),
            sidebar_bg: "#0F0A19".into(),
            warning: "#F59E0B".into(),
            error: "#EF4444".into(),
            success: "#14B8A6".into(),
            radius: "10px".into(),
            sidebar_width: "240px".into(),
        }
    }

    pub fn to_css_vars(&self) -> String {
        format!(
            "--bg:{};--fg:{};--accent:{};--primary:{};--surface:{};--border:{};--hover:{};--active-bg:{};--sidebar-bg:{};--warning:{};--error:{};--success:{};--radius:{};--sidebar-width:{}",
            self.bg, self.fg, self.accent, self.primary, self.surface, self.border, self.hover, self.active_bg, self.sidebar_bg, self.warning, self.error, self.success, self.radius, self.sidebar_width
        )
    }
}

pub fn get_theme(name: &str) -> Theme {
    match name {
        "aura" => Theme::aura(),
        "midnight" => Theme::midnight(),
        _ => Theme::onyx(),
    }
}
