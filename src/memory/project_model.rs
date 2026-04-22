use super::graph::MemoryGraph;
use super::types::{CognitiveMemoryUnit, MemoryScope, ProjectConvention, ProjectModel};
use std::collections::HashMap;

pub struct ProjectModelManager {
    models: HashMap<String, ProjectModel>,
}

impl Default for ProjectModelManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ProjectModelManager {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }

    pub fn get_or_create(&mut self, project_path: &str) -> &mut ProjectModel {
        self.models
            .entry(project_path.to_string())
            .or_insert_with(|| ProjectModel::new(project_path.to_string()))
    }

    pub fn get(&self, project_path: &str) -> Option<&ProjectModel> {
        self.models.get(project_path)
    }

    pub fn get_all(&self) -> Vec<&ProjectModel> {
        self.models.values().collect()
    }

    pub fn extract_conventions(&mut self, graph: &MemoryGraph, project_path: &str) {
        let model = match self.models.get_mut(project_path) {
            Some(m) => m,
            None => return,
        };

        let test_patterns = ["test", "spec", "__tests__"];
        let config_patterns = ["config", ".json", ".yaml", ".toml"];
        let src_patterns = ["src/", "lib/", "app/"];

        let mut test_convention_found = false;
        let mut config_convention_found = false;
        let mut src_convention_found = false;

        for mem in graph.get_all_memories() {
            let scope_matches = match &mem.scope {
                MemoryScope::Project { project_path: p } => p == project_path,
                _ => false,
            };
            if !scope_matches {
                continue;
            }

            let content_lower = mem.content.to_lowercase();
            let id = mem.id;

            if !test_convention_found {
                for pattern in &test_patterns {
                    if content_lower.contains(pattern) {
                        model.add_convention(
                            "Test location convention".to_string(),
                            format!("Tests are located using '{}' pattern", pattern),
                            vec![id],
                        );
                        test_convention_found = true;
                        break;
                    }
                }
            }

            if !config_convention_found {
                for pattern in &config_patterns {
                    if content_lower.contains(pattern) {
                        model.add_convention(
                            "Config file convention".to_string(),
                            format!("Configuration files use '{}' format", pattern),
                            vec![id],
                        );
                        config_convention_found = true;
                        break;
                    }
                }
            }

            if !src_convention_found {
                for pattern in &src_patterns {
                    if content_lower.contains(pattern) {
                        model.add_convention(
                            "Source code convention".to_string(),
                            format!("Source code is in '{}' directory", pattern),
                            vec![id],
                        );
                        src_convention_found = true;
                        break;
                    }
                }
            }
        }

        model.touch();
    }

    pub fn suggest_conventions(&self, project_path: &str) -> Vec<String> {
        match self.models.get(project_path) {
            Some(model) => model
                .conventions
                .iter()
                .map(|c| format!("{}: {}", c.name, c.description))
                .collect(),
            None => Vec::new(),
        }
    }
}

pub fn detect_convention_patterns(memories: &[&CognitiveMemoryUnit]) -> Vec<ProjectConvention> {
    let mut conventions = Vec::new();
    let mut seen_names = std::collections::HashSet::new();

    let patterns = [
        (
            "Test pattern",
            vec!["test", "spec", "__tests__", ".test.", ".spec."],
        ),
        (
            "Config format",
            vec!["config", ".json", ".yaml", ".yml", ".toml", ".config"],
        ),
        (
            "Build tool",
            vec!["cargo", "npm", "make", "cmake", "gradle", "webpack"],
        ),
        (
            "Package manager",
            vec!["package.json", "Cargo.toml", "requirements.txt", "go.mod"],
        ),
        (
            "Code style",
            vec!["prettier", "eslint", "rustfmt", "gofmt", "black"],
        ),
        (
            "Test framework",
            vec!["jest", "pytest", "rspec", "junit", "mocha"],
        ),
    ];

    for mem in memories {
        let content_lower = mem.content.to_lowercase();
        let id = mem.id;

        for (name, keywords) in &patterns {
            if !seen_names.contains(*name) {
                for keyword in keywords {
                    if content_lower.contains(keyword) {
                        seen_names.insert(*name);
                        conventions.push(ProjectConvention {
                            name: name.to_string(),
                            description: format!(
                                "Project uses '{}' for {}",
                                keyword,
                                name.to_lowercase()
                            ),
                            source_memory_ids: vec![id],
                            confidence: 0.7,
                        });
                        break;
                    }
                }
            }
        }
    }

    conventions
}
