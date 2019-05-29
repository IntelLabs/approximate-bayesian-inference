import numpy as np
import pywavefront as pyobj
from pyViewer.viewer import CGeometry


def extract_vertex_data(mat, out_format="V3F_N3F_T2F_C4F", scale=1.0):
    in_format = mat.vertex_format

    # Possible format string sizes 3, 7, 11, 15
    formats = []
    formats_len = []
    line_length = 0
    if len(in_format) >= 3:
        formats.append(in_format[0:3])
        formats_len.append(int(formats[-1][1]))
        line_length = line_length + int(formats[-1][1])
    if len(in_format) >= 7:
        formats.append(in_format[4:7])
        formats_len.append(int(formats[-1][1]))
        line_length = line_length + int(formats[-1][1])
    if len(in_format) >= 11:
        formats.append(in_format[8:11])
        formats_len.append(int(formats[-1][1]))
        line_length = line_length + int(formats[-1][1])
    if len(in_format) >= 15:
        formats.append(in_format[12:])
        formats_len.append(int(formats[-1][1]))
        line_length = line_length + int(formats[-1][1])

    vertices = []
    normals = []
    texture = []
    colors = []
    in_data = np.array(mat.vertices, np.float32).reshape(-1, line_length)
    values_idx = 0
    for idx, ft in enumerate(formats):
        if ft[0] == "V":
            v0 = values_idx
            v1 = v0 + formats_len[idx]
            vertices = in_data[:, v0:v1] * scale
        if ft[0] == "N":
            v0 = values_idx
            v1 = v0 + formats_len[idx]
            normals = in_data[:, v0:v1]
        if ft[0] == "T":
            v0 = values_idx
            v1 = v0 + formats_len[idx]
            texture = in_data[:, v0:v1]
        if ft[0] == "C":
            v0 = values_idx
            v1 = v0 + formats_len[idx]
            colors = in_data[:, v0:v1]
        values_idx = values_idx + formats_len[idx]

    assert len(vertices) != 0

    if len(texture) == 0:
        texture = np.zeros((vertices.shape[0], 3), np.float32)
    if len(normals) == 0:
        normals = np.ones((vertices.shape[0], 3), np.float32)
    if len(colors) == 0:
        if mat.texture is not None:
            colors = np.zeros((vertices.shape[0], 4), np.float32)
        else:
            colors = np.ones((vertices.shape[0], 4), np.float32) * np.array(mat.diffuse, np.float32)
    if len(colors[0]) == 3:
        colors = np.hstack([colors, np.ones(len(colors), np.float32).reshape(-1, 1)])
    if len(texture[0]) == 2:
        padding = np.zeros(len(texture), np.float32).reshape(-1,1)
        texture = np.hstack([texture, padding])

    vert_data = np.hstack([vertices, normals, texture, colors])
    return vert_data


def make_mesh(ctx, filename, scale=1.0):
    geom = CGeometry(ctx)
    geom.tex_id = None

    if isinstance(filename, type(b'')):
        mesh_filename = filename.decode("utf-8")
    else:
        mesh_filename = filename

    meshes = pyobj.Wavefront(mesh_filename, collect_faces=True)
    # Iterate vertex data collected in each material
    data = None
    for name, material in meshes.materials.items():
        m_vertex_data = extract_vertex_data(material, scale=scale)
        if material.texture is not None:
            geom.set_texture(material.texture.path)
        if data is None:
            data = m_vertex_data
        else:
            data = np.vstack([data, m_vertex_data])

    geom.set_data(data.astype(np.float32).reshape(-1).tobytes())

    return geom
