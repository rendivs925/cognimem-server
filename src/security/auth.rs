use crate::security::crypto::{hash_password, verify_password};

#[derive(Clone)]
pub struct AuthMiddleware {
    require_auth: bool,
    password_hash: Option<String>,
}

impl AuthMiddleware {
    pub fn new(require_auth: bool, password: Option<String>) -> Self {
        let password_hash = password.as_deref().map(hash_password);
        Self {
            require_auth,
            password_hash,
        }
    }

    pub fn authenticate(&self, token: Option<&str>) -> bool {
        if !self.require_auth {
            return true;
        }
        match (&self.password_hash, token) {
            (Some(hash), Some(t)) => verify_password(t, hash),
            _ => false,
        }
    }
}

pub struct TlsConfig {
    pub cert_path: String,
    pub key_path: String,
}

impl TlsConfig {
    pub fn new(cert: &str, key: &str) -> Self {
        Self {
            cert_path: cert.to_string(),
            key_path: key.to_string(),
        }
    }
}