import sys
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_SOLID
from OCC.Core.TopoDS import topods
from OCC.Core.BRep import BRep_Tool

# USD imports
from pxr import Usd, UsdGeom, Gf

def step_to_usd(step_path, usd_path, linear_deflection=0.1, angular_deflection=0.5):
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
    
    while solid_exp.More():
        solid = topods.Solid(solid_exp.Current())
        
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
            mesh_prim = UsdGeom.Mesh.Define(stage, f"/Model/solid_{solid_count}")
            
            # 设置顶点
            mesh_prim.CreatePointsAttr().Set(solid_vertices)
            
            # 设置面数据 - USD需要面的顶点数量和顶点索引
            face_vertex_counts = [3] * (len(solid_faces) // 3)  # 每个三角形有3个顶点
            mesh_prim.CreateFaceVertexCountsAttr().Set(face_vertex_counts)
            mesh_prim.CreateFaceVertexIndicesAttr().Set(solid_faces)
            
            # 设置细分方案为无细分（保持三角形）
            mesh_prim.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

        solid_exp.Next()
        solid_count += 1
    
    # 如果没有找到solid，则按原来的face级别处理
    if solid_count == 0:
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
    if solid_count > 0:
        print(f"Converted {step_path} to {usd_path} with {solid_count} solid meshes")
    else:
        print(f"Converted {step_path} to {usd_path} with 1 combined mesh")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert.py input.step output.usd")
        sys.exit(1)
    step_to_usd(sys.argv[1], sys.argv[2])