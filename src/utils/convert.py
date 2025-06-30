import sys
import os
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods_Face
from OCC.Core.BRep import BRep_Tool
import numpy as np

def step_to_obj(step_path, obj_path, linear_deflection=0.1, angular_deflection=0.5):
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

    vertices = []
    faces = []
    vert_idx = 1

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods_Face(exp.Current())
        location = face.Location()
        triangulation = BRep_Tool.Triangulation(face, location)
        if triangulation is None:
            exp.Next()
            continue

        nodes = triangulation.Nodes()
        triangles = triangulation.Triangles()

        # 记录顶点
        face_vertices = []
        for i in range(1, nodes.Size() + 1):
            pnt = nodes.Value(i)
            vertices.append((pnt.X(), pnt.Y(), pnt.Z()))
            face_vertices.append(vert_idx)
            vert_idx += 1

        # 记录面
        for i in range(1, triangles.Size() + 1):
            tri = triangles.Value(i)
            idxs = [tri.Value(1), tri.Value(2), tri.Value(3)]
            # OBJ索引从1开始
            faces.append([face_vertices[j-1] for j in idxs])

        exp.Next()

    # 写OBJ文件
    with open(obj_path, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    print(f"Converted {step_path} to {obj_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert.py input.step output.obj")
        sys.exit(1)
    step_to_obj(sys.argv[1], sys.argv[2])