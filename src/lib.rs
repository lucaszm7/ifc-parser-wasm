use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use bimifc_model::{SpatialNode, EntityResolver};

pub mod viewer;

#[derive(Serialize)]
pub struct IfcMetadataResponse {
    pub success: bool,
    pub error: Option<String>,
    pub root: Option<SpatialNodeDto>,
}

#[derive(Serialize)]
pub struct SpatialNodeDto {
    pub id: u32,
    pub node_type: String,
    pub name: String,
    pub entity_type: String,
    pub elevation: Option<f32>,
    pub has_geometry: bool,
    pub attributes: String,
    pub children: Vec<SpatialNodeDto>,
}

fn map_spatial_node(node: &SpatialNode, resolver: &dyn EntityResolver) -> SpatialNodeDto {
    let attributes = resolver.get(node.id)
        .map(|e| format!("{:?}", e.attributes))
        .unwrap_or_default();
        
    SpatialNodeDto {
        id: node.id.0,
        node_type: node.node_type.display_name().to_string(),
        name: node.name.clone(),
        entity_type: node.entity_type.clone(),
        elevation: node.elevation,
        has_geometry: node.has_geometry,
        attributes,
        children: node.children.iter().map(|c| map_spatial_node(c, resolver)).collect(),
    }
}

#[wasm_bindgen]
pub fn parse_ifc_metadata(data: &str) -> String {
    let result = parse_internal(data);
    match result {
        Ok(root) => {
            serde_json::to_string(&IfcMetadataResponse {
                success: true,
                error: None,
                root,
            }).unwrap_or_else(|_| "{}".to_string())
        }
        Err(e) => {
            serde_json::to_string(&IfcMetadataResponse {
                success: false,
                error: Some(e.to_string()),
                root: None,
            }).unwrap_or_else(|_| "{}".to_string())
        }
    }
}

fn parse_internal(data: &str) -> Result<Option<SpatialNodeDto>, String> {
    let model = bimifc_parser::parse(data).map_err(|e| e.to_string())?;
    let resolver = model.resolver();
    let spatial = model.spatial();
    
    let root = spatial.spatial_tree().map(|tree| map_spatial_node(tree, resolver));
    Ok(root)
}
