import numpy as np
import trimesh
import matplotlib.pyplot as plt
from xml.etree import ElementTree as ET
from scipy.ndimage import maximum_filter, minimum_filter, gaussian_filter
from PIL import Image


class Map:
    def __init__(self):
        self.map = None
        self.horizontal_scale = None
        self.plot_process = None
        self.origin = np.array([0,0])
    
    def get_size(self):
        if self.map is None:
            return 0, 0
        width = self.map.shape[0] * self.horizontal_scale
        height = self.map.shape[1] * self.horizontal_scale
        return width, height
    
    def get_value(self, x, y):
        x_gird = int((x - self.origin[0]) // self.horizontal_scale)
        y_gird = int((y - self.origin[1]) // self.horizontal_scale)
        return self.map[y_gird, x_gird]

        
    def plot_map(self):
        fig, ax = plt.subplots(figsize=(8, 6))

        im = None
        cbar = None

        height_field = self.map.copy()  # 避免绘制过程中被其他线程修改
        h, w = height_field.shape
        x = np.arange(w) * self.horizontal_scale
        y = np.arange(h) * self.horizontal_scale
        # 首次绘图
        im = ax.imshow(
            height_field,
            cmap='viridis',
            origin='lower',
            interpolation='none',
            extent=[x[0], x[-1], y[0], y[-1]]
        )
        ax.set_title("Height Map")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        cbar = plt.colorbar(im, ax=ax, label='Height (m)')
        plt.show()
    
    def save_png(
        self,
        path,
        z_min=None,
        z_max=None,
        flip_y=True
    ):
        assert self.map is not None, "map is None"

        height = self.map.copy().astype(np.float32)

        if z_min is None:
            z_min = float(height.min())
        if z_max is None:
            z_max = float(height.max())

        if z_max <= z_min:
            raise ValueError("z_max must be greater than z_min")

        # 归一化到 0–255
        height = np.clip(height, z_min, z_max)
        height = (height - z_min) / (z_max - z_min)
        height = (height * 255.0).astype(np.uint8)

        # MuJoCo 通常不需要 flip；如果你发现上下反了再开
        if flip_y:
            height = np.flipud(height)

        Image.fromarray(height, mode="L").save(path)
        print(f"[Map] heightfield png saved to: {path}")
        print(f"      z_min={z_min:.3f}, z_max={z_max:.3f}")
        
class HeightMap(Map):
    def __init__(self, origin, horizontal_scale, height_field_raw, scale=1):
        super().__init__()
        self.origin = origin
        self.horizontal_scale = horizontal_scale / scale
        self.map = np.repeat(np.repeat(height_field_raw, scale, axis=0), scale, axis=1)  # 转换为实际高度
        
class Box:
    def __init__(self, x, y, length, width, height):
        self.x = x
        self.y = y
        self.length = length
        self.width = width
        self.height = height
    
    def is_inside(self, x, y):
        """
        判断点 (x, y) 相对于盒子在二维平面的位置
        返回:
            1 -> 在内部
            0 -> 在边界上
           -1 -> 在外部
        """
        if (self.x < x < self.x + self.length) and (self.y < y < self.y + self.width):
            return 1
        elif ((x == self.x or x == self.x + self.length) and (self.y <= y <= self.y + self.width)) or \
             ((y == self.y or y == self.y + self.width) and (self.x <= x <= self.x + self.length)):
            return 0
        else:
            return -1


def show_heightmap(Z, resolution):
    """
    显示 2D 高程图灰度图
    参数:
        Z: numpy 2D 数组，高度值
        resolution: 每像素对应的实际米数（仅用于坐标标注）
    """
    H, W = Z.shape
    extent = [0, W * resolution, 0, H * resolution]

    plt.imshow(Z, cmap='gray', origin='lower', extent=extent)
    plt.colorbar(label='Height')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Heightmap (2D view)')
    plt.show()

def add_noise_grid_to_map(hmap:HeightMap, x, y, length, width,
                          scale=0.15, roughness=0.2, smoothness=0.0,
                          blend_ratio=0.0, seed=None, type="grid", method="add"):
    """""
    给 HeightMap 指定区域添加连续随机噪声，并与周围平滑融合。
    注意：hmap.map 的索引为 [row(y), col(x)]，horizontal_scale 表示实际坐标到格子的缩放比例。
    """

    # ---- 坐标转换（列/行）与边界限制 ----
    col_start = int(x // hmap.horizontal_scale)
    row_start = int(y // hmap.horizontal_scale)
    col_end = int((x + length) // hmap.horizontal_scale)
    row_end = int((y + width) // hmap.horizontal_scale)

    # clamp 到数组范围（shape: [rows, cols]）
    n_rows, n_cols = hmap.map.shape[0], hmap.map.shape[1]
    col_start = max(0, col_start)
    row_start = max(0, row_start)
    col_end = min(n_cols, col_end)
    row_end = min(n_rows, row_end)

    region_h = row_end - row_start  # 行数（height）
    region_w = col_end - col_start  # 列数（width）
    if region_h <= 0 or region_w <= 0:
        return
    rng = np.random.RandomState(seed) if seed is not None else np.random
    if type == "random":

        # ---- 生成噪声 ----
        noise = rng.randn(region_h, region_w)
        if smoothness > 0:
            noise = gaussian_filter(noise, sigma=smoothness)

        # 归一化到 [-1, 1]
        noise_min = noise.min()
        noise_max = noise.max()
        noise = noise - noise_min
        noise = noise / (noise_max - noise_min + 1e-12)
        noise = 2.0 * (noise - 0.5)
    elif type == "grid":
        grid_n_x = int(max(1, width/roughness))  # grid 划分数（至少 1x1）
        grid_n_y = int(max(1, length/roughness))  # grid 划分数（至少 1x1）
        sub_h = region_h // grid_n_x
        sub_w = region_w // grid_n_y

        noise = np.zeros((region_h, region_w), dtype=np.float32)

        for i in range(grid_n_x):
            for j in range(grid_n_y):
                # 每个小格的随机高度（范围约 [-scale, scale]）
                height_val = rng.uniform(-scale, scale)
                row_s = i * sub_h
                col_s = j * sub_w
                row_e = region_h if i == grid_n_x - 1 else (i + 1) * sub_h
                col_e = region_w if j == grid_n_y - 1 else (j + 1) * sub_w
                noise[row_s:row_e, col_s:col_e] = height_val
        # 通过 Gaussian 平滑让块之间过渡自然
        if smoothness > 0:
            noise = gaussian_filter(noise, sigma=smoothness)
    else:
        raise ValueError("Invalid noise type!")

    
    # ---- 构建边缘融合权重 mask ----
    yy, xx = np.mgrid[0:region_h, 0:region_w]
    dist_x = np.minimum(xx, region_w - 1 - xx)
    dist_y = np.minimum(yy, region_h - 1 - yy)
    dist_edge = np.minimum(dist_x, dist_y).astype(float)

    blend_w = int(min(region_h, region_w) * blend_ratio)
    if blend_w > 0:
        mask = np.clip(dist_edge / blend_w, 0.0, 1.0)
    else:
        mask = np.ones_like(noise)

    # 使用 cosine 提升过渡平滑度
    mask = 0.5 - 0.5 * np.cos(mask * np.pi)

    # ---- 应用噪声并融合（注意行/列顺序） ----
    blended = noise * scale * mask
    if method == "add":
        hmap.map[row_start:row_end, col_start:col_end] += blended
    else:
        hmap.map[row_start:row_end, col_start:col_end] = blended

def add_stairs_to_map(map:HeightMap, x, y, length, width, height, dir, number, method="set"): #right left up down
    boxs = []
    height_base = 0
    for i in range(number):
        box = Box(x, y, length, width, height + height_base)
        height_base = height_base + height
        if dir == "right":
            x = x + length
        elif dir == "left":
            x = x - length
        elif dir == "up":
            y = y + width
        elif dir == "down":
            y = y - width
        else:
            raise ValueError(f'Invalid dir input {dir}. Valid input are right, left, up, down.')
        boxs.append(box)
    
    add_boxs_to_map(map, boxs, method=method)
        
def add_boxs_to_map(map:HeightMap, boxs, edge_ratio_x=0.0, edge_ratio_y=0.0, method="set"):
    """
    在 heightmap 上叠加一系列 Box，可选择边缘倒角（斜坡平滑过渡）。

    参数:
        map (HeightMap): 含有 map.map (np.ndarray) 和 horizontal_scale 的高度图对象
        boxs (list[Box]): Box 列表，每个 Box 需包含 x, y, length, width, height
        method (str): "add" 或 "set"
        edge_ratio_x (float): X方向倒角相对长度比例（相对于box.length）
        edge_ratio_y (float): Y方向倒角相对宽度比例（相对于box.width）

    返回:
        None: 直接修改 map.map
    """
    Z = map.map
    resolution = map.horizontal_scale
    height, width = Z.shape

    for box in boxs:
        # 将物理坐标转换为像素索引
        x_start = int(box.x / resolution)
        y_start = int(box.y / resolution)
        x_end   = int((box.x + box.length) / resolution)
        y_end   = int((box.y + box.width) / resolution)

        x_start = max(0, min(width - 1, x_start))
        y_start = max(0, min(height - 1, y_start))
        x_end   = max(0, min(width, x_end))
        y_end   = max(0, min(height, y_end))

        # 分别计算 X、Y 方向的倒角宽度（像素）
        blend_px_x = int((edge_ratio_x * box.length) / resolution)
        blend_px_y = int((edge_ratio_y * box.width) / resolution)

        # 遍历区域并根据距离边缘的距离计算高度
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                # 距离最近边界的距离（像素）
                dx = min(x - x_start, x_end - 1 - x)
                dy = min(y - y_start, y_end - 1 - y)

                # 分别计算x、y方向平滑权重（0在边缘，1在中心）
                wx = min(1.0, dx / blend_px_x) if blend_px_x > 0 else 1.0
                wy = min(1.0, dy / blend_px_y) if blend_px_y > 0 else 1.0

                # 综合两个方向的权重，取较小值会形成四角过渡（类似圆角）
                w = min(wx, wy)

                # 最终高度
                h = box.height * w

                if method == "add":
                    Z[y, x] += h
                elif method == "set":
                    Z[y, x] = h
                else:
                    raise ValueError("unrecognized method! only 'add' and 'set' are supported")

    map.map = Z
    
def heightmap_to_dae(map, output_file="terrain.dae"):
    """
    将 numpy 高程图 (Z) 转为带双面材质的 DAE (Collada) 模型。
    
    参数:
        map: HeightMap 对象，包含 map.map (2D np.ndarray) 和 map.horizontal_scale
        output_file: 输出 DAE 文件名
    """
    xy_scale = map.horizontal_scale
    z_scale = 1
    Z = map.map
    H, W = Z.shape

    # 生成顶点
    xs = np.linspace(0, (W - 1) * xy_scale, W)
    ys = np.linspace(0, (H - 1) * xy_scale, H)
    X, Y = np.meshgrid(xs, ys)
    z_offset = 0.0  # 你希望的高程基准
    vertices = np.column_stack([X.flatten(), Y.flatten(), (Z - z_offset).flatten() * z_scale])

    # 生成三角形
    faces = []
    for y in range(H - 1):
        for x in range(W - 1):
            i0 = y * W + x
            i1 = i0 + 1
            i2 = i0 + W
            i3 = i2 + 1
            faces.append([i0, i1, i2])
            faces.append([i1, i3, i2])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=False)
    # 导出初步 DAE
    mesh.export(output_file)

    return mesh


def create_empty_map(origin, size, resolution):
    width = round(size[0] / resolution )
    height = round(size[1] / resolution )
    Z = np.zeros((height, width), dtype=float)  # 平地
    map = HeightMap(origin, resolution, Z)
    return map

def generate_normal(map):
    Z = map
    dzdx = np.gradient(Z, axis=1)  # x 方向
    dzdy = np.gradient(Z, axis=0)  # y 方向

    # 生成法线向量 (nx, ny, nz)
    nx = -dzdx
    ny = -dzdy
    nz = np.ones_like(Z)
    n = np.stack((nx, ny, nz), axis=-1)
    norm = np.linalg.norm(n, axis=2, keepdims=True)
    n /= norm  # 单位化

    # 映射到 RGB 0~255
    n_rgb = ((n + 1) / 2 * 255).astype(np.uint8)
    return n_rgb

def heightmap_to_pcd(map:HeightMap, output_file="terrain.pcd", edge_only=True, step=None):
    """
    将 heightmap 转为点云（支持体积或仅边缘+顶底面模式）
    """
    if step is None:
        step = map.horizontal_scale
    Z = map.map
    resolution = map.horizontal_scale
    H, W = Z.shape

    xs = np.arange(W) * resolution
    ys = np.arange(H) * resolution
    X, Y = np.meshgrid(xs, ys)

    points = []
    max_neigh = maximum_filter(Z, size=3)
    min_neigh = minimum_filter(Z, size=3)
    if edge_only:
        # === 边缘检测 ===
        
        edge_mask = (max_neigh - min_neigh) > step

        # === 顶面 ===
        top_points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
        points.append(top_points)

        # === 底面 ===
        # bottom_points = np.column_stack((X.flatten(), Y.flatten(), np.full_like(Z.flatten(), base_height)))
        # points.append(bottom_points)

        # === 边缘竖直面 ===
        edge_points = []
        for y in range(H):
            for x in range(W):
                if not edge_mask[y, x]:
                    continue
                z_top = Z[y, x]
                z_vals = np.arange(min_neigh[y,x], z_top + step, step)
                for z in z_vals:
                    edge_points.append([X[y, x], Y[y, x], z])
        if edge_points:
            points.append(np.array(edge_points))
    else:
        # === 体积填充（整个柱体）===
        volume_points = []
        for y in range(H):
            for x in range(W):
                z_top = Z[y, x]
                z_vals = np.arange(min_neigh[y,x], z_top + step, step)
                for z in z_vals:
                    volume_points.append([X[y, x], Y[y, x], z])
        points.append(np.array(volume_points))

    # === 合并所有点 ===
    points = np.vstack(points)

    # === 写入 PCD 文件 ===
    with open(output_file, "w") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {points.shape[0]}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {points.shape[0]}\n")
        f.write("DATA ascii\n")
        np.savetxt(f, points, fmt="%.6f %.6f %.6f")

    print(f"✅ PCD saved to {output_file} (points={points.shape[0]}, edge_only={edge_only})")
    return points

def heightmap_to_obj(map, output_file="terrain.obj", make_double_sided=False):
    """
    将 numpy 高程图 (Z) 转为 OBJ 模型。

    参数:
        map: HeightMap 对象，包含 map.map (2D np.ndarray) 和 map.horizontal_scale
        output_file: 输出 OBJ 文件名（建议以 .obj 结尾）
        make_double_sided: 是否通过“复制反向面”实现双面（OBJ本身没双面属性）
    """
    xy_scale = float(map.horizontal_scale)
    z_scale = 1.0
    Z = np.asarray(map.map)
    H, W = Z.shape

    # 生成顶点
    xs = np.linspace(0, (W - 1) * xy_scale, W, dtype=np.float64)
    ys = np.linspace(0, (H - 1) * xy_scale, H, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    z_offset = 0.0
    vertices = np.column_stack([
        X.reshape(-1),
        Y.reshape(-1),
        (Z.reshape(-1) - z_offset) * z_scale
    ]).astype(np.float64)

    # 生成三角形面
    faces = np.empty(((H - 1) * (W - 1) * 2, 3), dtype=np.int64)
    k = 0
    for y in range(H - 1):
        row = y * W
        row_next = (y + 1) * W
        for x in range(W - 1):
            i0 = row + x
            i1 = i0 + 1
            i2 = row_next + x
            i3 = i2 + 1

            faces[k] = [i0, i1, i2]
            k += 1
            faces[k] = [i1, i3, i2]
            k += 1

    if make_double_sided:
        # 复制一份反向面（把三角形顶点顺序反过来）
        faces = np.vstack([faces, faces[:, ::-1]])

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # 1) 去除退化三角形
    mesh.update_faces(mesh.nondegenerate_faces())

    # 2) 去除重复三角形
    mesh.update_faces(mesh.unique_faces())

    # 3) 清理孤立顶点
    mesh.remove_unreferenced_vertices()
    mesh.rezero()
    # 导出 OBJ（扩展名为 .obj 时 file_type 可省略，但写上更稳）
    mesh.export(output_file, file_type="obj")

    return mesh

if __name__ == "__main__":
    SAVE_FILES = True
    resolution = 0.05
    origin = np.array([0.0, 0.0])
    size = np.array([8, 8])
    map = create_empty_map(origin,size, resolution)
    
    add_noise_grid_to_map(map, 0.1, 0.1, 8.0, 7.8, seed=24, scale=0.1, smoothness=18, type="random", blend_ratio=0.05, method="set") #global noise
    add_noise_grid_to_map(map, 0.1, 0.1, 8.0, 7.8, seed=24, scale=0.02, smoothness=4, type="random", blend_ratio=0.05, method="add") #global noise
    
    box1 = Box(5.2, 0.2, 0.8, 6.3, -1.0)
    add_boxs_to_map(map, [box1]) # trench
    
    box1 = Box(6.5, 6.5, 1, 1, 0.0)
    add_boxs_to_map(map, [box1]) # spawn area
    
    box1 = Box(2, 6.5, 4.0, 1, 0.9)
    add_boxs_to_map(map, [box1]) # major platform
    
    add_stairs_to_map(map, 2, 4.7, 1.0, 0.31, 0.15, "up", 6) # stair to major platform
    
    add_stairs_to_map(map, 5.2, 3.0, 0.11, 1.0, 0.1, "right",4)
    add_stairs_to_map(map, 5.9, 3.0, 0.11, 1.0, 0.1, "left",4) # stair bridge
    
    
    box1 = Box(4, 7.5, 2.0, 0.5, -1.0)
    add_boxs_to_map(map, [box1], 0.2) # upper trench
    
    box1 = Box(5.2, 1.0, 0.8, 0.8, 1.0)
    add_boxs_to_map(map, [box1], method="add")  # lower bridge, traversable
    
    box1 = Box(5.2, 5.0, 0.8, 0.8, 0.5)
    add_boxs_to_map(map, [box1], 0.2) # upper bridge(not traversable)
    
    box1 = Box(6.8, 4.0, 1.0, 2.0, 0.3)
    add_boxs_to_map(map, [box1], 0.2, 0.2) # uper right small hill
    
    add_noise_grid_to_map(map, 6.2, 2, 0.8, 1.2, seed=24, scale=0.4, smoothness=0.15, roughness=0.15,type="grid",method="set") # right grid noise
    
    box1 = Box(6.8, 0.2, 1.0, 1.0, -0.3)
    add_boxs_to_map(map, [box1], 0.02, 0.02) # lower right pitfall
    
    box1 = Box(3.8, 3.0, 0.2, 2.0, 0.25)
    add_boxs_to_map(map, [box1], 0.05, 0.02) # door sill
    
    box1 = Box(5.1, 6.6, 0.8, 0.8, -0.02)
    add_boxs_to_map(map, [box1], method="add") # small pit square up
    
    boxs = []
    boxs.append(Box(0.4, 6.5, 1.6, 1.0, 0.9))
    boxs.append(Box(0.4, 2.3, 1.0, 4.3, 0.9))
    boxs.append(Box(0.4, 0.6, 1.0, 2.0, 1.0))
    boxs.append(Box(1.4, 0.6, 2.0, 1.0, 1.1))
    boxs.append(Box(3.2, 0.6, 1.0, 2.9, 1.2))
    boxs.append(Box(0.4, 2.5, 2.81, 1.0, 0.9))
    add_boxs_to_map(map, boxs, 0.0, 0.0) # sub platform
    
    box1 = Box(3.3, 0.7, 0.8, 0.8, -0.02)
    add_boxs_to_map(map, [box1], method="add") # small pit square down
    
    box1 = Box(3.3, 2.6, 0.8, 0.8, -0.02)
    add_boxs_to_map(map, [box1], method="add") # small pit square mid
    
    
    # map.plot_map()
    map.save_png("/root/unitree_mujoco/terrain_tool/terrain.png")
    heightmap_to_obj(map, "/root/unitree_mujoco/terrain_tool/terrain.obj")
    # if SAVE_FILES:
    #     rospack = rospkg.RosPack()
    #     dae_path = rospack.get_path("terrain_aware") + "/meshes/" + "terrain.dae"
    #     heightmap_to_dae(map, dae_path)
    #     pcd_path = rospack.get_path("terrain_aware") + "/pcds/" + "terrain.pcd"
    #     heightmap_to_pcd(map, pcd_path)