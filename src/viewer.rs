use bimifc_geometry::GeometryRouter;
use bimifc_geometry::transform::{resolve_cartesian_point, resolve_direction};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 4],
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

#[wasm_bindgen]
pub struct IfcViewer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: wgpu::Extent3d,
    render_pipeline: wgpu::RenderPipeline,
    depth_texture_view: wgpu::TextureView,

    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    num_indices: u32,

    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    // camera controls
    target: Vec3,
    distance: f32,
    yaw: f32,   // angle around Y axis
    pitch: f32, // angle above XZ plane
}

#[wasm_bindgen]
impl IfcViewer {
    #[wasm_bindgen]
    pub async fn create(canvas_id: String) -> Result<IfcViewer, JsValue> {
        let window = web_sys::window().unwrap();
        let document = window.document().unwrap();
        let canvas = document
            .get_element_by_id(&canvas_id)
            .expect("Canvas not found")
            .dyn_into::<HtmlCanvasElement>()?;

        let width = canvas.width();
        let height = canvas.height();

        let instance = wgpu::Instance::default();

        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to create WebGPU adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .expect("Failed to request WebGPU device");

        let config = surface.get_default_config(&adapter, width, height).unwrap();
        surface.configure(&device, &config);

        // Camera Uniforms Setup
        let camera_uniform = CameraUniform {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
        };
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        // Pipeline Setup
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[Some(&camera_bind_group_layout)],
                immediate_size: 0,
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            multiview_mask: None,
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"), // Updated per `wgpu` v0.20 APIs (now optional, but passing string works fine)
                compilation_options: Default::default(),
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // No backface culling since standard IFC geometries might be open
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            cache: None,
        });

        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_texture_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            depth_texture_view,
            vertex_buffer: None,
            index_buffer: None,
            num_indices: 0,
            camera_buffer,
            camera_bind_group,
            target: Vec3::ZERO,
            distance: 50.0,
            yaw: 0.785, // 45 degrees
            pitch: 0.5,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.size.width = width;
            self.size.height = height;
            self.surface.configure(&self.device, &self.config);

            let depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Depth Texture"),
                size: self.size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.depth_texture_view =
                depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        }
    }

    pub fn update_camera(&mut self) {
        let eye = self.target
            + Vec3::new(
                self.distance * self.yaw.sin() * self.pitch.cos(),
                self.distance * self.pitch.sin(),
                self.distance * self.yaw.cos() * self.pitch.cos(),
            );
        let view = Mat4::look_at_rh(eye, self.target, Vec3::Y);

        // Scale near/far with distance so large buildings are never clipped
        let near = (self.distance * 0.001).max(0.01);
        let far = self.distance * 10.0 + 1000.0;

        let proj = Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_4,
            self.config.width as f32 / self.config.height.max(1) as f32,
            near,
            far,
        );

        let view_proj = proj * view;
        let camera_uniform = CameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
        };
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[camera_uniform]),
        );
    }

    pub fn orbit_camera(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.01;
        self.pitch += dy * 0.01;
        // Clamp pitch to avoid gimbal lock and looking upside down
        self.pitch = self.pitch.clamp(
            -std::f32::consts::FRAC_PI_2 + 0.1,
            std::f32::consts::FRAC_PI_2 - 0.1,
        );
        self.update_camera();
    }

    pub fn zoom_camera(&mut self, scroll: f32) {
        self.distance *= 1.0 + (scroll * 0.001);
        self.distance = self.distance.clamp(1.0, 1000.0);
        self.update_camera();
    }

    pub fn load_ifc_geometry(&mut self, data: &str) -> Result<(), JsValue> {
        let model = bimifc_parser::parse(data).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let resolver = model.resolver();
        let router = GeometryRouter::with_default_processors_and_unit_scale(model.unit_scale());

        let mut all_vertices = Vec::new();
        let mut all_indices = Vec::new();
        let mut custom_mapped_item_cache = std::collections::HashMap::new();

        for id in 1..=50 {
            if let Some(entity) = resolver.get(bimifc_model::EntityId(id)) {
                if entity.ifc_type == bimifc_model::IfcType::IfcDirection
                    || entity.ifc_type == bimifc_model::IfcType::IfcCartesianPoint
                {
                    web_sys::console::log_1(&JsValue::from_str(&format!(
                        "DIR/POINT: id={} type={:?} attrs={:?}",
                        id, entity.ifc_type, entity.attributes
                    )));
                }
            }
        }

        if let Some(entity) = resolver.get(bimifc_model::EntityId(5527)) {
            web_sys::console::log_1(&JsValue::from_str(&format!(
                "WALL LOCATION 5527: attrs={:?}",
                entity.attributes
            )));
        }

        web_sys::console::log_1(&JsValue::from_str(&format!(
            "MODEL UNIT SCALE: {:?}",
            model.unit_scale()
        )));

        let mut logged = 0;
        for id in resolver.all_ids() {
            if let Some(entity) = resolver.get(id) {
                if logged < 15
                    && (entity.ifc_type == bimifc_model::IfcType::IfcWall
                        || entity.ifc_type == bimifc_model::IfcType::IfcWallStandardCase
                        || entity.ifc_type == bimifc_model::IfcType::IfcSlab
                        || entity.ifc_type == bimifc_model::IfcType::IfcRoof)
                {
                    web_sys::console::log_1(&JsValue::from_str(&format!(
                        "LOG ENTITY: id={:?} type={:?} attrs={:?}",
                        entity.id, entity.ifc_type, entity.attributes
                    )));
                    if let Some(placement_id) = entity.get_ref(5) {
                        log_placement_chain(placement_id, resolver, "  ");
                    }
                    logged += 1;
                }

                if let Ok(mesh) = custom_process_element(
                    &entity,
                    resolver,
                    &router,
                    &mut custom_mapped_item_cache,
                ) {
                    if mesh.is_empty() {
                        continue;
                    }

                    let color = bimifc_model::geometry::get_default_color(&entity.ifc_type);
                    let start_index = all_vertices.len() as u32;

                    for i in 0..(mesh.positions.len() / 3) {
                        // WGPU expects position, normals and color arrays.
                        // Wait, check if mesh.normals hasn't same size
                        let (nx, ny, nz) = if mesh.normals.len() >= (i * 3 + 3) {
                            (
                                mesh.normals[i * 3],
                                mesh.normals[i * 3 + 1],
                                mesh.normals[i * 3 + 2],
                            )
                        } else {
                            (0.0, 1.0, 0.0) // fallback
                        };

                        // bimifc typically yields Z as UP. For 3D viewers typically Y is UP. Let's fix axes:
                        all_vertices.push(Vertex {
                            position: [
                                mesh.positions[i * 3],
                                mesh.positions[i * 3 + 2],
                                -mesh.positions[i * 3 + 1],
                            ],
                            normal: [nx, nz, -ny],
                            color,
                        });
                    }

                    for idx in mesh.indices {
                        all_indices.push(start_index + idx);
                    }
                }
            }
        }

        if !all_vertices.is_empty() {
            // ── Compute bounding box ─────────────────────────────────────────
            let mut min = Vec3::splat(f32::MAX);
            let mut max = Vec3::splat(f32::MIN);
            for v in &all_vertices {
                let p = Vec3::from(v.position);
                min = min.min(p);
                max = max.max(p);
            }
            let center = (min + max) * 0.5;
            let half_diag = (max - min).length() * 0.5;

            // ── Fit camera to bounding sphere ───────────────────────────────
            // Use FOV = 45° (PI/4). Distance so sphere fills ~70% of screen.
            let fov_half = std::f32::consts::FRAC_PI_4 * 0.5; // 22.5°
            let fit_distance = (half_diag / fov_half.tan()) * 1.4;

            self.target = center;
            self.distance = fit_distance.max(0.5);
            self.yaw = 0.785; // 45° isometric angle
            self.pitch = 0.35; // slightly above horizon
            // Extend far plane to cover the whole model
            // (update_camera reads self.config.width/height for aspect ratio)

            self.vertex_buffer = Some(self.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("IFC Vertex Buffer"),
                    contents: bytemuck::cast_slice(&all_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                },
            ));
            self.index_buffer = Some(self.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("IFC Index Buffer"),
                    contents: bytemuck::cast_slice(&all_indices),
                    usage: wgpu::BufferUsages::INDEX,
                },
            ));
            self.num_indices = all_indices.len() as u32;
        }

        self.update_camera();
        Ok(())
    }

    pub fn render(&mut self) -> Result<(), JsValue> {
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) => t,
            wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => return Err(JsValue::from_str("Failed to get surface texture")),
        };

        let view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.05,
                            b: 0.05,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            if let (Some(v_buf), Some(i_buf)) = (&self.vertex_buffer, &self.index_buffer) {
                render_pass.set_pipeline(&self.render_pipeline);
                render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                render_pass.set_vertex_buffer(0, v_buf.slice(..));
                render_pass.set_index_buffer(i_buf.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        surface_texture.present();

        Ok(())
    }

    pub fn update_shader(&mut self, source: String) -> Result<(), JsValue> {
        let camera_bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                    label: Some("camera_bind_group_layout"),
                });

        let render_pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Render Pipeline Layout"),
                    bind_group_layouts: &[Some(&camera_bind_group_layout)],
                    immediate_size: 0,
                });

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Shader"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });

        let render_pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Render Pipeline"),
                multiview_mask: None,
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[Vertex::desc()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.config.format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::SrcAlpha,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent::OVER,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                cache: None,
            });

        self.render_pipeline = render_pipeline;
        Ok(())
    }
}

// Custom placement and transform resolution logic to override buggy bimifc-geometry implementation
fn custom_resolve_axis_placement(
    placement_id: bimifc_model::EntityId,
    resolver: &dyn bimifc_model::EntityResolver,
) -> Option<nalgebra::Matrix4<f64>> {
    let placement = resolver.get(placement_id)?;

    if placement.ifc_type != bimifc_model::IfcType::IfcAxis2Placement3D {
        return None;
    }

    // Location (index 0)
    let location = resolve_cartesian_point(placement.get_ref(0)?, resolver)?;

    // Axis (index 1) - Z direction, optional
    let axis = placement
        .get_ref(1)
        .and_then(|id| resolve_direction(id, resolver))
        .unwrap_or_else(|| nalgebra::Vector3::new(0.0, 0.0, 1.0));

    // RefDirection (index 2) - X direction, optional
    let ref_dir = placement
        .get_ref(2)
        .and_then(|id| resolve_direction(id, resolver))
        .unwrap_or_else(|| nalgebra::Vector3::new(1.0, 0.0, 0.0));

    // Build orthonormal basis
    let z = axis.normalize();
    let x = ref_dir.normalize();
    let mut y = z.cross(&x);
    if y.norm_squared() < 1e-6 {
        y = z.cross(&nalgebra::Vector3::new(0.0, 1.0, 0.0));
        if y.norm_squared() < 1e-6 {
            y = z.cross(&nalgebra::Vector3::new(0.0, 0.0, 1.0));
        }
    }
    let y = y.normalize();
    let x = y.cross(&z).normalize();

    Some(nalgebra::Matrix4::new(
        x.x, y.x, z.x, location.x, x.y, y.y, z.y, location.y, x.z, y.z, z.z, location.z, 0.0, 0.0,
        0.0, 1.0,
    ))
}

fn custom_resolve_transformation_operator(
    op_id: bimifc_model::EntityId,
    resolver: &dyn bimifc_model::EntityResolver,
) -> Option<nalgebra::Matrix4<f64>> {
    let op = resolver.get(op_id)?;

    // Axis1 (index 0) - X direction, optional
    let axis1 = op
        .get_ref(0)
        .and_then(|id| resolve_direction(id, resolver))
        .unwrap_or_else(|| nalgebra::Vector3::new(1.0, 0.0, 0.0));

    // Axis2 (index 1) - Y direction, optional
    let _axis2 = op
        .get_ref(1)
        .and_then(|id| resolve_direction(id, resolver))
        .unwrap_or_else(|| nalgebra::Vector3::new(0.0, 1.0, 0.0));

    // LocalOrigin (index 2)
    let origin = op
        .get_ref(2)
        .and_then(|id| resolve_cartesian_point(id, resolver))
        .unwrap_or_else(|| nalgebra::Point3::new(0.0, 0.0, 0.0));

    // Scale (index 3) - uniform scale, optional, default 1.0
    let scale = op.get_float(3).unwrap_or(1.0);

    // Axis3 (index 4) - Z direction, optional
    let axis3 = op
        .get_ref(4)
        .and_then(|id| resolve_direction(id, resolver))
        .unwrap_or_else(|| nalgebra::Vector3::new(0.0, 0.0, 1.0));

    // Scale2 (index 5) and Scale3 (index 6) for non-uniform operators
    let scale2 =
        if op.ifc_type == bimifc_model::IfcType::IfcCartesianTransformationOperator3DnonUniform {
            op.get_float(5).unwrap_or(scale)
        } else {
            scale
        };
    let scale3 =
        if op.ifc_type == bimifc_model::IfcType::IfcCartesianTransformationOperator3DnonUniform {
            op.get_float(6).unwrap_or(scale)
        } else {
            scale
        };

    // Build orthonormal basis
    let z = axis3.normalize();
    let x = axis1.normalize();
    let mut y = z.cross(&x);
    if y.norm_squared() < 1e-6 {
        y = z.cross(&nalgebra::Vector3::new(0.0, 1.0, 0.0));
        if y.norm_squared() < 1e-6 {
            y = z.cross(&nalgebra::Vector3::new(0.0, 0.0, 1.0));
        }
    }
    let y = y.normalize();
    let x = y.cross(&z).normalize();

    // Apply scale to the basis vectors
    let col0 = x * scale;
    let col1 = y * scale2;
    let col2 = z * scale3;

    Some(nalgebra::Matrix4::new(
        col0.x, col1.x, col2.x, origin.x, col0.y, col1.y, col2.y, origin.y, col0.z, col1.z, col2.z,
        origin.z, 0.0, 0.0, 0.0, 1.0,
    ))
}

fn custom_resolve_placement(
    placement_id: bimifc_model::EntityId,
    resolver: &dyn bimifc_model::EntityResolver,
) -> Option<nalgebra::Matrix4<f64>> {
    let placement = resolver.get(placement_id)?;

    match placement.ifc_type {
        bimifc_model::IfcType::IfcLocalPlacement => {
            // Recursively resolve parent placement (attribute 0: PlacementRelTo)
            let parent_transform = placement
                .get_ref(0)
                .and_then(|parent_id| custom_resolve_placement(parent_id, resolver))
                .unwrap_or_else(nalgebra::Matrix4::identity);

            // Resolve local transform (attribute 1: RelativePlacement)
            let local_transform = placement
                .get_ref(1)
                .and_then(|rel_id| custom_resolve_placement(rel_id, resolver))
                .unwrap_or_else(nalgebra::Matrix4::identity);

            Some(parent_transform * local_transform)
        }
        bimifc_model::IfcType::IfcAxis2Placement3D => {
            custom_resolve_axis_placement(placement_id, resolver)
        }
        bimifc_model::IfcType::IfcCartesianTransformationOperator3D
        | bimifc_model::IfcType::IfcCartesianTransformationOperator3DnonUniform => {
            custom_resolve_transformation_operator(placement_id, resolver)
        }
        _ => None,
    }
}

fn custom_process_mapped_item(
    entity: &bimifc_model::DecodedEntity,
    resolver: &dyn bimifc_model::EntityResolver,
    router: &bimifc_geometry::GeometryRouter,
    cache: &mut std::collections::HashMap<u32, bimifc_geometry::Mesh>,
) -> Result<Option<bimifc_geometry::Mesh>, bimifc_geometry::error::Error> {
    // Extract source_id (MappingSource -> RepresentationMap -> MappedRepresentation)
    let source_id = entity
        .get_ref(0)
        .and_then(|map_id| resolver.get(map_id))
        .and_then(|rep_map| rep_map.get_ref(1))
        .map(|id| id.0);

    if let Some(sid) = source_id {
        if let Some(cached) = cache.get(&sid) {
            let mut mesh = cached.clone();
            if let Some(transform_id) = entity.get_ref(1) {
                if let Some(transform) = custom_resolve_placement(transform_id, resolver) {
                    bimifc_geometry::extrusion::apply_transform(&mut mesh, &transform);
                }
            }
            custom_scale_mesh(&mut mesh, router.unit_scale());
            return Ok(Some(mesh));
        }
    }

    // Not cached - resolve the source representation and process its items
    if let Some(map_id) = entity.get_ref(0) {
        if let Some(rep_map) = resolver.get(map_id) {
            if let Some(shape_rep_id) = rep_map.get_ref(1) {
                if let Some(shape_rep) = resolver.get(shape_rep_id) {
                    if let Some(mut mesh) =
                        custom_process_shape_representation(&shape_rep, resolver, router, cache)?
                    {
                        if let Some(sid) = source_id {
                            cache.insert(sid, mesh.clone());
                        }
                        if let Some(transform_id) = entity.get_ref(1) {
                            if let Some(transform) =
                                custom_resolve_placement(transform_id, resolver)
                            {
                                bimifc_geometry::extrusion::apply_transform(&mut mesh, &transform);
                            }
                        }
                        custom_scale_mesh(&mut mesh, router.unit_scale());
                        return Ok(Some(mesh));
                    }
                }
            }
        }
    }

    Ok(None)
}

fn custom_scale_mesh(mesh: &mut bimifc_geometry::Mesh, unit_scale: f64) {
    if unit_scale != 1.0 {
        let scale = unit_scale as f32;
        for pos in mesh.positions.iter_mut() {
            *pos *= scale;
        }
    }
}

fn resolve_axis2_placement_2d(
    placement_id: bimifc_model::EntityId,
    resolver: &dyn bimifc_model::EntityResolver,
) -> Option<nalgebra::Matrix4<f64>> {
    let placement = resolver.get(placement_id)?;
    if placement.ifc_type != bimifc_model::IfcType::IfcAxis2Placement2D {
        return None;
    }

    let location_id = placement.get_ref(0)?;
    let location_entity = resolver.get(location_id)?;
    let coords = location_entity.get(0)?.as_list()?;
    let x = coords.first().and_then(|v| v.as_float()).unwrap_or(0.0);
    let y = coords.get(1).and_then(|v| v.as_float()).unwrap_or(0.0);

    let ref_dir = placement
        .get_ref(1)
        .and_then(|id| {
            let dir_entity = resolver.get(id)?;
            let ratios = dir_entity.get(0)?.as_list()?;
            let dx = ratios.first().and_then(|v| v.as_float()).unwrap_or(1.0);
            let dy = ratios.get(1).and_then(|v| v.as_float()).unwrap_or(0.0);
            Some(nalgebra::Vector2::new(dx, dy).normalize())
        })
        .unwrap_or_else(|| nalgebra::Vector2::new(1.0, 0.0));

    let u = ref_dir;
    let v = nalgebra::Vector2::new(-u.y, u.x);

    Some(nalgebra::Matrix4::new(
        u.x, v.x, 0.0, x, u.y, v.y, 0.0, y, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ))
}

fn custom_process_shape_representation(
    shape_rep: &bimifc_model::DecodedEntity,
    resolver: &dyn bimifc_model::EntityResolver,
    router: &bimifc_geometry::GeometryRouter,
    cache: &mut std::collections::HashMap<u32, bimifc_geometry::Mesh>,
) -> Result<Option<bimifc_geometry::Mesh>, bimifc_geometry::error::Error> {
    // Get Items (index 3 in IfcShapeRepresentation)
    let items = match shape_rep.get(3) {
        Some(bimifc_model::AttributeValue::List(list)) => list,
        _ => return Ok(None),
    };

    let mut combined = bimifc_geometry::Mesh::new();

    for item_ref in items {
        if let Some(item_id) = item_ref.as_entity_ref() {
            if let Some(item) = resolver.get(item_id) {
                if item.ifc_type == bimifc_model::IfcType::IfcMappedItem {
                    if let Some(mesh) = custom_process_mapped_item(&item, resolver, router, cache)?
                    {
                        combined.merge(&mesh);
                    }
                } else if item.ifc_type == bimifc_model::IfcType::IfcExtrudedAreaSolid
                    || item.ifc_type == bimifc_model::IfcType::IfcRevolvedAreaSolid
                {
                    let mut modified_item = (*item).clone();
                    if modified_item.attributes.len() > 1 {
                        modified_item.attributes[1] = bimifc_model::AttributeValue::Null;
                    }
                    if let Ok(mut mesh) =
                        router.process_representation_item(&modified_item, resolver)
                    {
                        let scale = router.unit_scale();

                        // 1. Resolve and apply profile's Position transform (index 2 of SweptArea)
                        if let Some(swept_id) = item.get_ref(0) {
                            if let Some(swept) = resolver.get(swept_id) {
                                if let Some(prof_pos_id) = swept.get_ref(2) {
                                    if let Some(mut prof_pos_transform) =
                                        resolve_axis2_placement_2d(prof_pos_id, resolver)
                                    {
                                        if scale != 1.0 {
                                            prof_pos_transform[(0, 3)] *= scale;
                                            prof_pos_transform[(1, 3)] *= scale;
                                        }
                                        bimifc_geometry::extrusion::apply_transform(
                                            &mut mesh,
                                            &prof_pos_transform,
                                        );
                                    }
                                }
                            }
                        }

                        // 2. Resolve and apply solid's original Position transform (index 1 of solid)
                        if let Some(pos_id) = item.get_ref(1) {
                            if let Some(mut pos_transform) =
                                custom_resolve_placement(pos_id, resolver)
                            {
                                if scale != 1.0 {
                                    pos_transform[(0, 3)] *= scale;
                                    pos_transform[(1, 3)] *= scale;
                                    pos_transform[(2, 3)] *= scale;
                                }
                                bimifc_geometry::extrusion::apply_transform(
                                    &mut mesh,
                                    &pos_transform,
                                );
                            }
                        }

                        combined.merge(&mesh);
                    }
                } else {
                    match router.process_representation_item(&item, resolver) {
                        Ok(mesh) => combined.merge(&mesh),
                        Err(_) => continue,
                    }
                }
            }
        }
    }

    if combined.is_empty() {
        Ok(None)
    } else {
        Ok(Some(combined))
    }
}

fn custom_process_element(
    element: &bimifc_model::DecodedEntity,
    resolver: &dyn bimifc_model::EntityResolver,
    router: &bimifc_geometry::GeometryRouter,
    cache: &mut std::collections::HashMap<u32, bimifc_geometry::Mesh>,
) -> Result<bimifc_geometry::Mesh, bimifc_geometry::error::Error> {
    let mut combined_mesh = bimifc_geometry::Mesh::new();

    // Get Representation (typically at index 6 for products)
    let rep_id = match element.get_ref(6) {
        Some(id) => id,
        None => return Ok(combined_mesh), // No representation
    };

    let representation = match resolver.get(rep_id) {
        Some(rep) => rep,
        None => return Ok(combined_mesh),
    };

    // Get Representations list (index 2 in IfcProductDefinitionShape)
    // IfcProductRepresentation: 0=Name, 1=Description, 2=Representations
    let reps = match representation.get(2) {
        Some(bimifc_model::AttributeValue::List(list)) => list,
        _ => return Ok(combined_mesh),
    };

    // Process each shape representation
    for rep_ref in reps {
        if let Some(shape_rep_id) = rep_ref.as_entity_ref() {
            if let Some(shape_rep) = resolver.get(shape_rep_id) {
                // Filter: skip non-geometry representations (e.g., "Axis", "Box", "FootPrint")
                // Attribute 1: RepresentationIdentifier (e.g., "Body", "Facetation")
                if let Some(rep_id_str) = shape_rep.get_string(1) {
                    if !matches!(
                        rep_id_str,
                        "Body" | "Facetation" | "Reference" | "MappedRepresentation"
                    ) {
                        continue;
                    }
                }

                // Process representation items
                if let Some(mesh) =
                    custom_process_shape_representation(&shape_rep, resolver, router, cache)?
                {
                    combined_mesh.merge(&mesh);
                }
            }
        }
    }

    // Apply object placement transform
    if let Some(placement_id) = element.get_ref(5) {
        if let Some(mut transform) = custom_resolve_placement(placement_id, resolver) {
            if element.id.0 == 5548 {
                web_sys::console::log_1(&wasm_bindgen::JsValue::from_str(&format!(
                    "WALL 5548 LOCAL BOUNDS: {:?}",
                    combined_mesh.bounds()
                )));
                web_sys::console::log_1(&wasm_bindgen::JsValue::from_str(&format!(
                    "WALL 5548 TRANSFORM MATRIX: {:?}",
                    transform
                )));
            }
            // Scale translation components from file units to meters
            let scale = router.unit_scale();
            if scale != 1.0 {
                transform[(0, 3)] *= scale;
                transform[(1, 3)] *= scale;
                transform[(2, 3)] *= scale;
            }
            bimifc_geometry::extrusion::apply_transform(&mut combined_mesh, &transform);
            if element.id.0 == 5548 {
                web_sys::console::log_1(&wasm_bindgen::JsValue::from_str(&format!(
                    "WALL 5548 TRANSFORMED BOUNDS: {:?}",
                    combined_mesh.bounds()
                )));
            }
        }
    }

    Ok(combined_mesh)
}

fn log_placement_chain(
    placement_id: bimifc_model::EntityId,
    resolver: &dyn bimifc_model::EntityResolver,
    indent: &str,
) {
    if let Some(placement) = resolver.get(placement_id) {
        web_sys::console::log_1(&wasm_bindgen::JsValue::from_str(&format!(
            "{}PLACEMENT: id={:?} type={:?} attrs={:?}",
            indent, placement_id, placement.ifc_type, placement.attributes
        )));
        if placement.ifc_type == bimifc_model::IfcType::IfcLocalPlacement {
            // RelTo (index 0)
            if let Some(rel_to_id) = placement.get_ref(0) {
                log_placement_chain(rel_to_id, resolver, &format!("{}  ", indent));
            }
            // Relative (index 1)
            if let Some(rel_id) = placement.get_ref(1) {
                if let Some(rel) = resolver.get(rel_id) {
                    web_sys::console::log_1(&wasm_bindgen::JsValue::from_str(&format!(
                        "{}  RELATIVE: id={:?} type={:?} attrs={:?}",
                        indent, rel_id, rel.ifc_type, rel.attributes
                    )));
                }
            }
        }
    }
}
