use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};

pub mod viewer;

#[derive(Serialize)]
pub struct IfcMetadataResponse {
    pub success: bool,
    pub error: Option<String>,
    pub elements: Vec<IfcElementDto>,
}

#[derive(Serialize)]
pub struct IfcElementDto {
    pub id: u32,
    pub entity_type: String,
    pub attributes: String,
}

#[wasm_bindgen]
pub fn parse_ifc_metadata(data: &str) -> String {
    let result = parse_internal(data);
    match result {
        Ok(elements) => {
            serde_json::to_string(&IfcMetadataResponse {
                success: true,
                error: None,
                elements,
            }).unwrap_or_else(|_| "{}".to_string())
        }
        Err(e) => {
            serde_json::to_string(&IfcMetadataResponse {
                success: false,
                error: Some(e.to_string()),
                elements: vec![],
            }).unwrap_or_else(|_| "{}".to_string())
        }
    }
}

fn parse_internal(data: &str) -> Result<Vec<IfcElementDto>, String> {
    let model = bimifc_parser::parse(data).map_err(|e| e.to_string())?;
    let resolver = model.resolver();
    
    let mut elements = Vec::new();
    for id in resolver.all_ids() {
        if let Some(entity) = resolver.get(id) {
            let type_name = format!("{:?}", entity.ifc_type);
            elements.push(IfcElementDto {
                id: id.0,
                entity_type: type_name,
                attributes: format!("{:?}", entity.attributes),
            });
        }
    }
    
    Ok(elements)
}
