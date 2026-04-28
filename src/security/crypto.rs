use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};
use bcrypt::{hash, verify, DEFAULT_COST};
use rand::Rng;
use serde::{Deserialize, Serialize};

const KEY_SIZE: usize = 32;

#[derive(Serialize, Deserialize, Clone)]
pub struct EncryptedData {
    pub nonce: Vec<u8>,
    pub ciphertext: Vec<u8>,
}

pub struct Crypto {
    key: [u8; KEY_SIZE],
}

impl Crypto {
    pub fn new(password: &str) -> Self {
        let hashed = hash(password, DEFAULT_COST).unwrap_or_else(|_| {
            let mut key = [0u8; KEY_SIZE];
            rand::rngs::OsRng.fill(&mut key);
            String::from_utf8(key.to_vec()).unwrap_or_default()
        });
        let mut key = [0u8; KEY_SIZE];
        let len = KEY_SIZE.min(hashed.as_bytes().len());
        key[..len].copy_from_slice(&hashed.as_bytes()[..len]);
        Self { key }
    }

    pub fn from_key(key: [u8; KEY_SIZE]) -> Self {
        Self { key }
    }

    pub fn encrypt(&self, plaintext: &[u8]) -> EncryptedData {
        let cipher = Aes256Gcm::new(self.key.as_ref().into());
        let mut nonce_bytes = [0u8; 12];
        rand::rngs::OsRng.fill(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);
        let ciphertext = cipher.encrypt(nonce, plaintext).expect("encryption failed");
        EncryptedData {
            nonce: nonce_bytes.to_vec(),
            ciphertext,
        }
    }

    pub fn decrypt(&self, data: &EncryptedData) -> Option<Vec<u8>> {
        let cipher = Aes256Gcm::new(self.key.as_ref().into());
        let nonce = Nonce::from_slice(&data.nonce);
        cipher.decrypt(nonce, data.ciphertext.as_ref()).ok()
    }

    pub fn encrypt_str(&self, plaintext: &str) -> EncryptedData {
        self.encrypt(plaintext.as_bytes())
    }

    pub fn decrypt_str(&self, data: &EncryptedData) -> Option<String> {
        let decrypted = self.decrypt(data)?;
        String::from_utf8(decrypted).ok()
    }
}

pub fn hash_password(password: &str) -> String {
    hash(password, DEFAULT_COST).unwrap_or_default()
}

pub fn verify_password(password: &str, hash: &str) -> bool {
    verify(password, hash).is_ok()
}