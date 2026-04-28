use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use uuid::Uuid;

static NEXT_VERSION: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum CodeNodeKind {
    Project,
    Module,
    File,
    Function,
    Struct,
    Trait,
    Impl,
    Enum,
    TypeAlias,
    Constant,
    Macro,
}

impl std::fmt::Display for CodeNodeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodeNodeKind::Project => write!(f, "project"),
            CodeNodeKind::Module => write!(f, "module"),
            CodeNodeKind::File => write!(f, "file"),
            CodeNodeKind::Function => write!(f, "function"),
            CodeNodeKind::Struct => write!(f, "struct"),
            CodeNodeKind::Trait => write!(f, "trait"),
            CodeNodeKind::Impl => write!(f, "impl"),
            CodeNodeKind::Enum => write!(f, "enum"),
            CodeNodeKind::TypeAlias => write!(f, "type_alias"),
            CodeNodeKind::Constant => write!(f, "constant"),
            CodeNodeKind::Macro => write!(f, "macro"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodeRelation {
    Contains,
    Calls,
    Imports,
    Implements,
    DependsOn,
}

impl std::fmt::Display for CodeRelation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodeRelation::Contains => write!(f, "contains"),
            CodeRelation::Calls => write!(f, "calls"),
            CodeRelation::Imports => write!(f, "imports"),
            CodeRelation::Implements => write!(f, "implements"),
            CodeRelation::DependsOn => write!(f, "depends_on"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeNode {
    pub id: Uuid,
    pub kind: CodeNodeKind,
    pub name: String,
    pub file_path: PathBuf,
    pub line_start: usize,
    pub line_end: usize,
    pub version: u64,
    pub summary: Option<String>,
    pub children: Vec<Uuid>,
}

impl CodeNode {
    pub fn new(kind: CodeNodeKind, name: String, file_path: PathBuf, line_start: usize, line_end: usize) -> Self {
        Self {
            id: Uuid::new_v4(),
            kind,
            name,
            file_path,
            line_start,
            line_end,
            version: NEXT_VERSION.fetch_add(1, Ordering::Relaxed),
            summary: None,
            children: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeEdge {
    pub from: Uuid,
    pub to: Uuid,
    pub relation: CodeRelation,
}

pub struct CodeGraph {
    nodes: HashMap<Uuid, CodeNode>,
    edges: Vec<CodeEdge>,
    file_index: HashMap<PathBuf, Vec<Uuid>>,
    name_index: HashMap<String, Vec<Uuid>>,
}

impl CodeGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            file_index: HashMap::new(),
            name_index: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, node: CodeNode) -> Uuid {
        let id = node.id;
        let file = node.file_path.clone();
        let name = node.name.clone();
        self.file_index.entry(file).or_default().push(id);
        self.name_index.entry(name).or_default().push(id);
        self.nodes.insert(id, node);
        id
    }

    pub fn add_edge(&mut self, from: Uuid, to: Uuid, relation: CodeRelation) {
        self.edges.push(CodeEdge { from, to, relation });
    }

    pub fn get_node(&self, id: &Uuid) -> Option<&CodeNode> {
        self.nodes.get(id)
    }

    pub fn get_node_mut(&mut self, id: &Uuid) -> Option<&mut CodeNode> {
        self.nodes.get_mut(id)
    }

    pub fn get_children(&self, id: &Uuid) -> Vec<&CodeNode> {
        let node = match self.nodes.get(id) {
            Some(n) => n,
            None => return Vec::new(),
        };
        node.children
            .iter()
            .filter_map(|cid| self.nodes.get(cid))
            .collect()
    }

    pub fn get_nodes_in_file(&self, path: &Path) -> Vec<&CodeNode> {
        self.file_index
            .get(path)
            .map(|ids| ids.iter().filter_map(|id| self.nodes.get(id)).collect())
            .unwrap_or_default()
    }

    pub fn search_by_name(&self, query: &str) -> Vec<&CodeNode> {
        let lower = query.to_lowercase();
        self.nodes
            .values()
            .filter(|n| n.name.to_lowercase().contains(&lower))
            .collect()
    }

    pub fn get_related_nodes(&self, id: &Uuid, relation: &CodeRelation, hops: usize) -> Vec<&CodeNode> {
        let mut visited = HashSet::new();
        let mut result = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back((*id, 0));
        visited.insert(*id);

        while let Some((current, depth)) = queue.pop_front() {
            if depth >= hops {
                continue;
            }
            for edge in &self.edges {
                if edge.from == current && edge.relation == *relation {
                    if visited.insert(edge.to) {
                        if let Some(node) = self.nodes.get(&edge.to) {
                            result.push(node);
                            queue.push_back((edge.to, depth + 1));
                        }
                    }
                }
                if edge.to == current && edge.relation == *relation {
                    if visited.insert(edge.from) {
                        if let Some(node) = self.nodes.get(&edge.from) {
                            result.push(node);
                            queue.push_back((edge.from, depth + 1));
                        }
                    }
                }
            }
        }
        result
    }

    pub fn all_files(&self) -> Vec<&PathBuf> {
        self.file_index.keys().collect()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

fn detect_language(path: &Path) -> Option<&'static str> {
    match path.extension()?.to_str()? {
        "rs" => Some("rust"),
        "py" => Some("python"),
        "js" | "jsx" => Some("javascript"),
        "ts" | "tsx" => Some("typescript"),
        _ => None,
    }
}

pub fn parse_file(path: &Path, content: &str) -> Vec<CodeNode> {
    let lang = match detect_language(path) {
        Some(l) => l,
        None => return Vec::new(),
    };

    match lang {
        "rust" => parse_rust(path, content),
        "python" => parse_python(path, content),
        "javascript" | "typescript" => parse_generic(path, content, lang),
        _ => Vec::new(),
    }
}

pub fn discover_project(root: &Path, graph: &mut CodeGraph) -> Result<usize, String> {
    let mut count = 0;
    let supported = ["rs", "py", "js", "jsx", "ts", "tsx"];

    for entry in walkdir::WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_str().unwrap_or("");
            !name.starts_with('.') && name != "node_modules" && name != "target"
        })
    {
        let entry = entry.map_err(|e| e.to_string())?;
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !supported.contains(&ext) {
            continue;
        }
        let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let nodes = parse_file(path, &content);
        for node in nodes {
            graph.add_node(node);
            count += 1;
        }
    }
    Ok(count)
}

fn make_file_node(path: &Path) -> CodeNode {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string_lossy().to_string());
    CodeNode::new(CodeNodeKind::File, name, path.to_path_buf(), 1, 1)
}

fn parse_rust(path: &Path, content: &str) -> Vec<CodeNode> {
    use tree_sitter::Parser;

    let mut parser = Parser::new();
    let language = tree_sitter_rust::LANGUAGE.into();
    parser.set_language(&language).ok();

    let tree = match parser.parse(content, None) {
        Some(t) => t,
        None => return vec![make_file_node(path)],
    };

    let mut nodes = vec![make_file_node(path)];
    let cursor = &mut tree.walk();

    let mut stack: Vec<(Uuid, usize)> = Vec::new();
    let root = nodes[0].id;

    loop {
        let node = cursor.node();
        let kind = node.kind();

        let code_node = match kind {
            "function_item" | "function" => {
                let name = node.child_by_field_name("name")
                    .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                    .unwrap_or("anon");
                Some(CodeNode::new(
                    CodeNodeKind::Function,
                    name.to_string(),
                    path.to_path_buf(),
                    node.start_position().row + 1,
                    node.end_position().row + 1,
                ))
            }
            "struct_item" => {
                let name = node.child_by_field_name("name")
                    .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                    .unwrap_or("anon");
                Some(CodeNode::new(
                    CodeNodeKind::Struct,
                    name.to_string(),
                    path.to_path_buf(),
                    node.start_position().row + 1,
                    node.end_position().row + 1,
                ))
            }
            "trait_item" => {
                let name = node.child_by_field_name("name")
                    .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                    .unwrap_or("anon");
                Some(CodeNode::new(
                    CodeNodeKind::Trait,
                    name.to_string(),
                    path.to_path_buf(),
                    node.start_position().row + 1,
                    node.end_position().row + 1,
                ))
            }
            "impl_item" => {
                let name = node.child_by_field_name("trait")
                    .or_else(|| node.child_by_field_name("type"))
                    .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                    .unwrap_or("impl");
                Some(CodeNode::new(
                    CodeNodeKind::Impl,
                    format!("impl {}", name),
                    path.to_path_buf(),
                    node.start_position().row + 1,
                    node.end_position().row + 1,
                ))
            }
            "enum_item" => {
                let name = node.child_by_field_name("name")
                    .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                    .unwrap_or("anon");
                Some(CodeNode::new(
                    CodeNodeKind::Enum,
                    name.to_string(),
                    path.to_path_buf(),
                    node.start_position().row + 1,
                    node.end_position().row + 1,
                ))
            }
            "type_item" => {
                let name = node.child_by_field_name("name")
                    .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                    .unwrap_or("anon");
                Some(CodeNode::new(
                    CodeNodeKind::TypeAlias,
                    name.to_string(),
                    path.to_path_buf(),
                    node.start_position().row + 1,
                    node.end_position().row + 1,
                ))
            }
            "const_item" => {
                let name = node.child_by_field_name("name")
                    .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                    .unwrap_or("anon");
                Some(CodeNode::new(
                    CodeNodeKind::Constant,
                    name.to_string(),
                    path.to_path_buf(),
                    node.start_position().row + 1,
                    node.end_position().row + 1,
                ))
            }
            "macro_definition" | "macro_invocation" => {
                let name = node.child_by_field_name("name")
                    .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                    .unwrap_or("macro");
                Some(CodeNode::new(
                    CodeNodeKind::Macro,
                    name.to_string(),
                    path.to_path_buf(),
                    node.start_position().row + 1,
                    node.end_position().row + 1,
                ))
            }
            "mod_item" | "module" => {
                let name = node.child_by_field_name("name")
                    .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                    .unwrap_or("mod");
                Some(CodeNode::new(
                    CodeNodeKind::Module,
                    name.to_string(),
                    path.to_path_buf(),
                    node.start_position().row + 1,
                    node.end_position().row + 1,
                ))
            }
            _ => None,
        };

        if let Some(cn) = code_node {
            let id = cn.id;

            let parent_id = stack.last().map(|(pid, _)| *pid).unwrap_or(root);
            if let Some(parent) = nodes.iter_mut().find(|n| n.id == parent_id) {
                parent.children.push(id);
            }
            nodes.push(cn);
            if node.child_count() > 0 {
                stack.push((id, node.start_position().row));
            }
        }

        if cursor.goto_first_child() {
            continue;
        }
        if cursor.goto_next_sibling() {
            continue;
        }
        loop {
            if !cursor.goto_parent() {
                break;
            }
            if let Some(last) = stack.last() {
                if last.1 <= cursor.node().start_position().row {
                    stack.pop();
                }
            }
            if cursor.goto_next_sibling() {
                break;
            }
        }
        if stack.is_empty() {
            break;
        }
    }

    nodes
}

fn parse_python(path: &Path, content: &str) -> Vec<CodeNode> {
    use tree_sitter::Parser;

    let mut parser = Parser::new();
    let language = tree_sitter_python::LANGUAGE.into();
    parser.set_language(&language).ok();

    let tree = match parser.parse(content, None) {
        Some(t) => t,
        None => return vec![make_file_node(path)],
    };

    let mut nodes = vec![make_file_node(path)];
    let cursor = &mut tree.walk();
    let root = nodes[0].id;

    loop {
        let node = cursor.node();
        let kind = node.kind();

        let code_node = match kind {
            "function_definition" => {
                let name = node.child_by_field_name("name")
                    .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                    .unwrap_or("anon");
                Some(CodeNode::new(
                    CodeNodeKind::Function,
                    name.to_string(),
                    path.to_path_buf(),
                    node.start_position().row + 1,
                    node.end_position().row + 1,
                ))
            }
            "class_definition" => {
                let name = node.child_by_field_name("name")
                    .and_then(|n| n.utf8_text(content.as_bytes()).ok())
                    .unwrap_or("anon");
                Some(CodeNode::new(
                    CodeNodeKind::Struct,
                    name.to_string(),
                    path.to_path_buf(),
                    node.start_position().row + 1,
                    node.end_position().row + 1,
                ))
            }
            _ => None,
        };

        if let Some(cn) = code_node {
            let parent_id = nodes.last().filter(|n| n.kind != CodeNodeKind::File)
                .map(|n| n.id)
                .unwrap_or(root);
            if let Some(parent) = nodes.iter_mut().find(|n| n.id == parent_id) {
                parent.children.push(cn.id);
            }
            nodes.push(cn);
        }

        if cursor.goto_first_child() { continue; }
        if cursor.goto_next_sibling() { continue; }
        loop {
            if !cursor.goto_parent() { break; }
            if cursor.goto_next_sibling() { break; }
        }
        if cursor.node().kind() == "module" && !cursor.goto_next_sibling() {
            break;
        }
    }

    nodes
}

fn parse_generic(path: &Path, content: &str, _lang: &str) -> Vec<CodeNode> {
    let mut nodes = vec![make_file_node(path)];
    for (lineno, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("function ") || trimmed.starts_with("export function ") {
            let name = trimmed
                .trim_start_matches("export ")
                .trim_start_matches("function ")
                .split(|c: char| c == '(' || c == '<' || c == ' ')
                .next()
                .unwrap_or("anon");
            nodes.push(CodeNode::new(
                CodeNodeKind::Function,
                name.to_string(),
                path.to_path_buf(),
                lineno + 1,
                lineno + 1,
            ));
        } else if trimmed.starts_with("class ") || trimmed.starts_with("export class ") {
            let name = trimmed
                .trim_start_matches("export ")
                .trim_start_matches("class ")
                .split(|c: char| c == '{' || c == ' ' || c == '<')
                .next()
                .unwrap_or("anon");
            nodes.push(CodeNode::new(
                CodeNodeKind::Struct,
                name.to_string(),
                path.to_path_buf(),
                lineno + 1,
                lineno + 1,
            ));
        }
    }
    nodes
}
