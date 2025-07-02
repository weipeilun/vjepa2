import sys
import os
import yaml
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_SOLID
from OCC.Core.TopoDS import topods
from OCC.Core.BRep import BRep_Tool

# USD imports
from pxr import Usd, UsdGeom, Gf, Sdf, UsdShade, UsdPhysics

def validate_config(config):
    """Validate the configuration structure."""
    if not isinstance(config, dict):
        return False, "Configuration must be a dictionary"
    
    # Check required fields
    if 'convert_unlisted' not in config:
        return False, "Missing required field 'convert_unlisted'"
    
    if not isinstance(config['convert_unlisted'], bool):
        return False, "'convert_unlisted' must be a boolean"
    
    # Check solids configuration if present
    if 'solids' in config:
        if not isinstance(config['solids'], dict):
            return False, "'solids' must be a dictionary"
        
        # Validate each solid's configuration
        for solid_name, solid_config in config['solids'].items():
            if not isinstance(solid_config, dict):
                return False, f"Configuration for solid '{solid_name}' must be a dictionary"
            
            # Validate convert flag
            if 'convert' in solid_config and not isinstance(solid_config['convert'], bool):
                return False, f"'convert' flag for solid '{solid_name}' must be a boolean"
            
            # Validate visuals flag
            if 'visuals' in solid_config and not isinstance(solid_config['visuals'], bool):
                return False, f"'visuals' flag for solid '{solid_name}' must be a boolean"
            
            # Validate collisions flag
            if 'collisions' in solid_config and not isinstance(solid_config['collisions'], bool):
                return False, f"'collisions' flag for solid '{solid_name}' must be a boolean"
            
            # Validate joint configuration
            if 'joint' in solid_config:
                joint_config = solid_config['joint']
                if not isinstance(joint_config, dict):
                    return False, f"Joint configuration for solid '{solid_name}' must be a dictionary"
                
                # Check required joint fields
                if 'type' not in joint_config:
                    return False, f"Joint configuration for solid '{solid_name}' missing required field 'type'"
                
                if joint_config['type'] not in ['prismatic']:  # Add more joint types as needed
                    return False, f"Invalid joint type '{joint_config['type']}' for solid '{solid_name}'"
                
                if 'body0' not in joint_config or 'body1' not in joint_config:
                    return False, f"Joint configuration for solid '{solid_name}' missing required fields 'body0' and/or 'body1'"
    
    return True, None

def load_conversion_config(config_path):
    """Load and validate the YAML configuration file for conversion."""
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found. Using default configuration.")
        return {
            'convert_unlisted': True,
            'solids': {}
        }
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Validate configuration
        is_valid, error_msg = validate_config(config)
        if not is_valid:
            print(f"Error in configuration file: {error_msg}")
            print("Using default configuration.")
            return {
                'convert_unlisted': True,
                'solids': {}
            }
        
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        print("Using default configuration.")
        return {
            'convert_unlisted': True,
            'solids': {}
        }
    except Exception as e:
        print(f"Unexpected error loading config file: {e}")
        print("Using default configuration.")
        return {
            'convert_unlisted': True,
            'solids': {}
        }

def should_convert_solid(solid_name, config):
    """Determine if a solid should be converted based on configuration."""
    if not config:
        return True
    
    # Extract solid number from the name (e.g., "solid_42" -> "42")
    try:
        solid_number = str(int(solid_name.split('_')[1]))  # Convert to string for dict lookup
    except (IndexError, ValueError):
        # If we can't parse the number, use default behavior
        return config.get('convert_unlisted', True)
    
    # Get the solids configuration
    solids_config = config.get('solids', {})
    
    # Check if this solid has specific configuration
    if solid_number in solids_config:
        return solids_config[solid_number].get('convert', True)
    
    # Return the default behavior for unlisted solids
    return config.get('convert_unlisted', True)

def get_solid_config(solid_name, config):
    """Get the configuration for a specific solid."""
    if not config:
        return {}
    
    try:
        solid_number = str(int(solid_name.split('_')[1]))
    except (IndexError, ValueError):
        return {}
    
    solids_config = config.get('solids', {})
    return solids_config.get(solid_number, {})

def create_solid_xform(stage, solid_name, parent_path="/Model"):
    """Create an Xform for the solid."""
    xform_path = f"{parent_path}/{solid_name}"
    xform = UsdGeom.Xform.Define(stage, xform_path)
    
    # Add transform with scale (1,1,1)
    xform.AddTranslateOp().Set((0.0, 0.0, 0.0))
    xform.AddRotateXYZOp().Set((0.0, 0.0, 0.0))
    xform.AddScaleOp().Set((1.0, 1.0, 1.0))
    
    return xform

def create_material(stage, material_path="/Model/Looks/material_191919"):
    """Create a OmniPBR material."""
    # Create the material with OmniPBR type
    material = UsdShade.Material.Define(stage, material_path)
    
    # Create the shader
    shader_path = material_path + "/Shader"
    shader = UsdShade.Shader.Define(stage, shader_path)

    # 设置Shader的实现来源为MDL文件
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)

    # 指定MDL文件名和类型
    shader.SetSourceAsset("OmniPBR.mdl", "mdl")

    # 指定MDL文件中的子材质名称
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")

    # 连接Material的surface、displacement、volume输出到Shader的输出
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    material.CreateDisplacementOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    material.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    
    return material

def create_visual_mesh(stage, solid_name, vertices, faces, parent_path="/Model"):
    """Create a visual mesh for the solid."""
    mesh_path = f"{parent_path}/{solid_name}/visual"
    mesh_prim = UsdGeom.Mesh.Define(stage, mesh_path)
    
    # Add transform with scale (1,1,1)
    mesh_prim.AddTranslateOp().Set((0.0, 0.0, 0.0))
    mesh_prim.AddRotateXYZOp().Set((0.0, 0.0, 0.0))
    mesh_prim.AddScaleOp().Set((1.0, 1.0, 1.0))
    
    # Set mesh data
    mesh_prim.CreatePointsAttr().Set(vertices)
    face_vertex_counts = [3] * (len(faces) // 3)
    mesh_prim.CreateFaceVertexCountsAttr().Set(face_vertex_counts)
    mesh_prim.CreateFaceVertexIndicesAttr().Set(faces)
    mesh_prim.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    
    # Bind material
    material_path = "/Model/Looks/material_191919"
    if not stage.GetPrimAtPath(material_path):
        material = create_material(stage)
    material_binding = UsdShade.MaterialBindingAPI(mesh_prim)
    material_binding.Bind(UsdShade.Material.Get(stage, material_path))
    
    return mesh_prim

def create_collision_mesh(stage, solid_name, vertices, faces, parent_path="/Model"):
    """Create a collision mesh for the solid."""
    mesh_path = f"{parent_path}/{solid_name}/collision"
    mesh_prim = UsdGeom.Mesh.Define(stage, mesh_path)
    
    # Add transform with scale (1,1,1)
    mesh_prim.AddTranslateOp().Set((0.0, 0.0, 0.0))
    mesh_prim.AddRotateXYZOp().Set((0.0, 0.0, 0.0))
    mesh_prim.AddScaleOp().Set((1.0, 1.0, 1.0))
    
    # Set mesh data
    mesh_prim.CreatePointsAttr().Set(vertices)
    face_vertex_counts = [3] * (len(faces) // 3)
    mesh_prim.CreateFaceVertexCountsAttr().Set(face_vertex_counts)
    mesh_prim.CreateFaceVertexIndicesAttr().Set(faces)
    mesh_prim.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    
    # Get the prim to apply APIs
    prim = mesh_prim.GetPrim()
    
    # Apply collision API first
    if UsdPhysics.CollisionAPI.CanApply(prim):
        collision_api = UsdPhysics.CollisionAPI.Apply(prim)
        collision_api.CreateCollisionEnabledAttr().Set(True)
    else:
        print(f"Warning: Cannot apply CollisionAPI to {mesh_path}")
        return mesh_prim
    
    # Apply mesh collision API for triangle mesh collision
    if UsdPhysics.MeshCollisionAPI.CanApply(prim):
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        # Use string value for mesh approximation (valid values: "none", "convexHull", "convexDecomposition", "boundingSphere", "boundingCube", "meshSimplification")
        mesh_collision_api.CreateApproximationAttr().Set("meshSimplification")
    else:
        print(f"Warning: Cannot apply MeshCollisionAPI to {mesh_path}")
    
    # Set purpose to guide for collision geometry (makes it invisible for rendering)
    mesh_prim.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
    
    return mesh_prim

def create_prismatic_joint(stage, joint_config, solid_name, parent_path="/Model"):
    """Create a prismatic joint between two bodies."""
    if not joint_config or 'body0' not in joint_config or 'body1' not in joint_config:
        return None
        
    body0 = joint_config['body0']
    body1 = joint_config['body1']
    
    # Create joint path under the solid
    joint_path = Sdf.Path(f"{parent_path}/{solid_name}/PrismaticJoint")
    
    # Create prismatic joint using USD Physics API
    prismatic_joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
    
    # Set body0 and body1 paths
    body0_path = Sdf.Path(f"{parent_path}/solid_{body0}")
    body1_path = Sdf.Path(f"{parent_path}/solid_{body1}")
    
    # Set body relationships
    prismatic_joint.CreateBody0Rel().SetTargets([body0_path])
    prismatic_joint.CreateBody1Rel().SetTargets([body1_path])
    
    # Set joint axis (default to X axis, can be configured later)
    # prismatic_joint.CreateAxisAttr().Set((1.0, 0.0, 0.0))
    
    return prismatic_joint

def step_to_usd(step_path, usd_path, config_path=None, linear_deflection=0.1, angular_deflection=0.5):
    # Load configuration
    config = load_conversion_config(config_path) if config_path else None
    
    # 读取STEP文件
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(step_path)
    if status != IFSelect_RetDone:
        print("Error: Cannot read STEP file.")
        return

    step_reader.TransferRoots()
    shape = step_reader.Shape()

    # 网格化
    mesh = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
    mesh.Perform()

    # 创建USD Stage
    stage = Usd.Stage.CreateNew(usd_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    
    # 设置单位为厘米 (1厘米 = 0.01米)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    
    # 创建根节点
    model_root = UsdGeom.Xform.Define(stage, "/Model")
    
    # Add transform with scale (1,1,1)
    model_root.AddTranslateOp().Set((0.0, 0.0, 0.0))
    model_root.AddRotateXYZOp().Set((0.0, 0.0, 0.0))
    model_root.AddScaleOp().Set((1.0, 1.0, 1.0))
    
    # 设置模型单位信息的自定义metadata
    model_prim = model_root.GetPrim()
    
    # Create Looks scope for materials
    UsdGeom.Scope.Define(stage, "/Model/Looks")

    # 首先尝试按solid级别探索
    solid_exp = TopExp_Explorer(shape, TopAbs_SOLID)
    solid_count = 0
    converted_count = 0
    
    while solid_exp.More():
        solid = topods.Solid(solid_exp.Current())
        solid_name = f"solid_{solid_count}"
        
        # Check if this solid should be converted
        if not should_convert_solid(solid_name, config):
            solid_count += 1
            solid_exp.Next()
            continue

        # 为当前solid收集所有面的顶点和索引
        solid_vertices = []
        solid_faces = []
        vert_offset = 0
        
        # 在当前solid中探索所有面
        face_exp = TopExp_Explorer(solid, TopAbs_FACE)
        while face_exp.More():
            face = topods.Face(face_exp.Current())
            location = face.Location()
            triangulation = BRep_Tool.Triangulation(face, location)
            if triangulation is None:
                face_exp.Next()
                continue

            # 使用新的API访问节点和三角形
            nb_nodes = triangulation.NbNodes()
            nb_triangles = triangulation.NbTriangles()

            # 记录当前面的顶点（手动转换米到厘米，除以100）
            for i in range(1, nb_nodes + 1):
                pnt = triangulation.Node(i)
                solid_vertices.append(Gf.Vec3f(pnt.X() / 100.0, pnt.Y() / 100.0, pnt.Z() / 100.0))

            # 记录当前面的三角形索引
            for i in range(1, nb_triangles + 1):
                tri = triangulation.Triangle(i)
                idxs = [tri.Value(1), tri.Value(2), tri.Value(3)]
                # USD索引从0开始，调整索引并加上偏移量
                solid_faces.extend([j - 1 + vert_offset for j in idxs])
            
            vert_offset += nb_nodes
            face_exp.Next()

        # Get solid specific configuration
        solid_config = get_solid_config(solid_name, config)
        
        # Create Xform for this solid
        create_solid_xform(stage, solid_name)
        
        # Create visual mesh if needed
        if solid_config.get('visuals', True):
            create_visual_mesh(stage, solid_name, solid_vertices, solid_faces)
            
        # Create collision mesh if needed
        if solid_config.get('collisions', False):
            create_collision_mesh(stage, solid_name, solid_vertices, solid_faces)
            
        # Create joint if specified
        if 'joint' in solid_config:
            create_prismatic_joint(stage, solid_config['joint'], solid_name)
            
        converted_count += 1
        solid_count += 1
        solid_exp.Next()

    # 保存USD文件
    stage.GetRootLayer().Save()
    if converted_count > 0:
        print(f"Converted {step_path} to {usd_path} with {converted_count} solid meshes (out of {solid_count} total solids)")
    else:
        print(f"No solids were converted from {step_path} to {usd_path}")


if __name__ == "__main__":
    if len(sys.argv) not in [4, 5]:
        print("Usage: python convert.py base_dir input.step output.usd [config.yaml]")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    input_file = os.path.join(base_dir, sys.argv[2])
    output_file = os.path.join(base_dir, sys.argv[3])
    config_path = os.path.join(base_dir, sys.argv[4]) if len(sys.argv) == 5 else None
    
    # Ensure input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    step_to_usd(input_file, output_file, config_path)