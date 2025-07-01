import sys
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
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
    
    vertices = []
    faces = []
    vert_idx = 0

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    face_count = 0
    
    while exp.More():
        face = topods.Face(exp.Current())
        location = face.Location()
        triangulation = BRep_Tool.Triangulation(face, location)
        if triangulation is None:
            exp.Next()
            continue

        # 使用新的API访问节点和三角形
        nb_nodes = triangulation.NbNodes()
        nb_triangles = triangulation.NbTriangles()

        # 记录顶点
        face_vertices = []
        for i in range(1, nb_nodes + 1):
            pnt = triangulation.Node(i)
            vertices.append(Gf.Vec3f(pnt.X(), pnt.Y(), pnt.Z()))
            face_vertices.append(vert_idx)
            vert_idx += 1

        # 记录面
        for i in range(1, nb_triangles + 1):
            tri = triangulation.Triangle(i)
            idxs = [tri.Value(1), tri.Value(2), tri.Value(3)]
            # USD索引从0开始，调整索引
            faces.extend([face_vertices[j - 1] for j in idxs])

        exp.Next()
        face_count += 1

    # 创建USD Mesh
    if vertices and faces:
        mesh_prim = UsdGeom.Mesh.Define(stage, "/Model/mesh")
        
        # 设置顶点
        mesh_prim.CreatePointsAttr().Set(vertices)
        
        # 设置面数据 - USD需要面的顶点数量和顶点索引
        face_vertex_counts = [3] * (len(faces) // 3)  # 每个三角形有3个顶点
        mesh_prim.CreateFaceVertexCountsAttr().Set(face_vertex_counts)
        mesh_prim.CreateFaceVertexIndicesAttr().Set(faces)
        
        # 设置细分方案为无细分（保持三角形）
        mesh_prim.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

    # 保存USD文件
    stage.GetRootLayer().Save()
    print(f"Converted {step_path} to {usd_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert.py input.step output.usd")
        sys.exit(1)
    step_to_usd(sys.argv[1], sys.argv[2])