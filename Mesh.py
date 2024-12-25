import pygmsh
import matplotlib.pyplot as plt
import numpy as np


# 创建几何对象并生成网格
# with pygmsh.geo.Geometry() as geom:
#     circle = geom.add_circle([0.0, 0.0], 1.0, mesh_size=0.05)
#     # Rect = geom.add_polygon([[0, 0], [2, 0], [2, 1],
#     #                          [1, 1], [0.5, 3], [0, 3]], mesh_size=0.1)
#     mesh = geom.generate_mesh(dim=2, verbose=False)  # 生成二维网格
#     mesh.write("test1.vtk")  # 保存网格文件（可选）

# with pygmsh.occ.Geometry() as geom:
#     geom.characteristic_length_max = 0.1
#     disks = [
#         geom.add_disk([-0.5 * np.cos(7 / 6 * np.pi), -0.25], 1.0),
#         geom.add_disk([+0.5 * np.cos(7 / 6 * np.pi), -0.25], 1.0),
#         geom.add_disk([0.0, 0.5], 1.0),
#     ]
#     geom.boolean_intersection(disks)
#     mesh = geom.generate_mesh()

with pygmsh.occ.Geometry() as geom:
    geom.characteristic_length_min = 0.1
    geom.characteristic_length_max = 0.1
    disks = [
        geom.add_rectangle([0, 0, 0], 4, 4),
        # geom.add_polygon([[0, 0], [4, 0], [4, 4], [0, 4]]),
        geom.add_disk([2, 2, 0], 1.0),
    ]
    geom.boolean_difference(disks[0], disks[1])
    mesh = geom.generate_mesh()

# 提取节点坐标
nodes = mesh.points

# 提取三角形单元
if "triangle" in mesh.cells_dict:
    triangles = mesh.cells_dict["triangle"]  # 三角形单元连接信息
else:
    raise ValueError("生成的网格不包含三角形单元！")

# 提取单元节点索引并去重
unique_node_indices = np.unique(triangles)

# 筛选节点坐标
nodes = nodes[unique_node_indices, :]

# 构建节点索引映射表
node_index_map = {old_index: new_index for new_index,
                  old_index in enumerate(unique_node_indices)}

# 使用 np.vectorize 更新单元信息
vectorized_map = np.vectorize(lambda x: node_index_map[x])
triangles = vectorized_map(triangles)

# 计算每个单元的形心坐标，面积和质量，预分配数组
centroids = np.zeros(triangles.shape)
areas = np.zeros(len(triangles))
quality = np.zeros(len(triangles))

for i, triangle in enumerate(triangles):
    # 提取单元的节点坐标
    cell_nodes = nodes[triangle]
    # 计算形心坐标
    centroids[i] = np.mean(cell_nodes, axis=0)
    # 计算单元面积
    a, b, c = cell_nodes
    areas[i] = 0.5 * np.abs(a[0]*(b[1]-c[1]) + b[0] *
                            (c[1]-a[1]) + c[0]*(a[1]-b[1]))
    quality[i] = 4*np.sqrt(3)*areas[i]/(np.dot(a-b, a-b) +
                                        np.dot(b-c, b-c)+np.dot(c-a, c-a))
print("总面积:", areas.sum())
print("平均质量:", quality.mean())

# 绘制网格
plt.figure(figsize=(13, 6))
plt.subplot(1, 2, 1)
plt.gca().set_aspect('equal')  # 设置坐标轴比例相同
plt.triplot(nodes[:, 0], nodes[:, 1], triangles,
            color='k', linewidth=0.5)  # 绘制三角形网格
plt.scatter(nodes[:, 0], nodes[:, 1], color='r', s=3)  # 绘制节点
plt.scatter(centroids[:, 0], centroids[:, 1], color='g', s=6)  # 绘制形心
plt.title("Circle")
plt.xlabel("X")
plt.ylabel("Y")


# 绘制网格质量热图
plt.subplot(1, 2, 2)
plt.gca().set_aspect('equal')
plt.tripcolor(nodes[:, 0], nodes[:, 1], triangles, quality, cmap='viridis')
plt.colorbar(label="quality")
plt.title("quality")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
