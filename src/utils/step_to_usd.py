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
from pxr import Usd, UsdGeom, Gf

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
        for solid_name, should_convert in config['solids'].items():
            if not isinstance(should_convert, bool):
                return False, f"Configuration for solid '{solid_name}' must be a boolean"
    
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
    
    # Get the solids configuration, defaulting to empty dict if not present
    solids_config = config.get('solids', {})
    
    # Get the default behavior for unlisted solids
    convert_unlisted = config.get('convert_unlisted', True)
    
    # Check if this solid is explicitly configured
    if solid_name in solids_config:
        return solids_config[solid_name]
    
    # Return the default behavior for unlisted solids
    return convert_unlisted

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
    
    # 创建根节点
    model_root = UsdGeom.Xform.Define(stage, "/Model")

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

            # 记录当前面的顶点
            for i in range(1, nb_nodes + 1):
                pnt = triangulation.Node(i)
                solid_vertices.append(Gf.Vec3f(pnt.X(), pnt.Y(), pnt.Z()))

            # 记录当前面的三角形索引
            for i in range(1, nb_triangles + 1):
                tri = triangulation.Triangle(i)
                idxs = [tri.Value(1), tri.Value(2), tri.Value(3)]
                # USD索引从0开始，调整索引并加上偏移量
                solid_faces.extend([j - 1 + vert_offset for j in idxs])
            
            vert_offset += nb_nodes
            face_exp.Next()

        # 为当前solid创建一个USD Mesh（包含其所有面）
        if solid_vertices and solid_faces:
            mesh_prim = UsdGeom.Mesh.Define(stage, f"/Model/{solid_name}")
            
            # 设置顶点
            mesh_prim.CreatePointsAttr().Set(solid_vertices)
            
            # 设置面数据 - USD需要面的顶点数量和顶点索引
            face_vertex_counts = [3] * (len(solid_faces) // 3)  # 每个三角形有3个顶点
            mesh_prim.CreateFaceVertexCountsAttr().Set(face_vertex_counts)
            mesh_prim.CreateFaceVertexIndicesAttr().Set(solid_faces)
            
            # 设置细分方案为无细分（保持三角形）
            mesh_prim.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
            converted_count += 1

        solid_exp.Next()
        solid_count += 1
    
    # 如果没有找到solid或没有转换任何solid，则按原来的face级别处理
    if converted_count == 0:
        face_vertices = []
        face_indices = []
        vert_offset = 0
        
        face_exp = TopExp_Explorer(shape, TopAbs_FACE)
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

            # 记录当前面的顶点
            for i in range(1, nb_nodes + 1):
                pnt = triangulation.Node(i)
                face_vertices.append(Gf.Vec3f(pnt.X(), pnt.Y(), pnt.Z()))

            # 记录当前面的三角形索引
            for i in range(1, nb_triangles + 1):
                tri = triangulation.Triangle(i)
                idxs = [tri.Value(1), tri.Value(2), tri.Value(3)]
                # USD索引从0开始，调整索引并加上偏移量
                face_indices.extend([j - 1 + vert_offset for j in idxs])
            
            vert_offset += nb_nodes
            face_exp.Next()

        # 创建单个mesh包含所有面
        if face_vertices and face_indices:
            mesh_prim = UsdGeom.Mesh.Define(stage, "/Model/all_faces")
            
            # 设置顶点
            mesh_prim.CreatePointsAttr().Set(face_vertices)
            
            # 设置面数据 - USD需要面的顶点数量和顶点索引
            face_vertex_counts = [3] * (len(face_indices) // 3)  # 每个三角形有3个顶点
            mesh_prim.CreateFaceVertexCountsAttr().Set(face_vertex_counts)
            mesh_prim.CreateFaceVertexIndicesAttr().Set(face_indices)
            
            # 设置细分方案为无细分（保持三角形）
            mesh_prim.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

    # 保存USD文件
    stage.GetRootLayer().Save()
    if converted_count > 0:
        print(f"Converted {step_path} to {usd_path} with {converted_count} solid meshes (out of {solid_count} total solids)")
    else:
        print(f"Converted {step_path} to {usd_path} with 1 combined mesh")


if __name__ == "__main__":
    if len(sys.argv) not in [3, 4]:
        print("Usage: python convert.py input.step output.usd [config.yaml]")
        sys.exit(1)
    
    config_path = sys.argv[3] if len(sys.argv) == 4 else None
    step_to_usd(sys.argv[1], sys.argv[2], config_path)