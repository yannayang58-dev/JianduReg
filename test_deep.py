import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import json
import time
from scipy.spatial.transform import Rotation
import os

# ====================== 路径配置 ======================
SOURCE_PLY = r"C:\Users\27412\PycharmProjects\sys_dataset\Jiandu_chip\output_oblique_mid\point_clouds\two_step_faces\oblique_final_face_1_fractureA_first_complete_0049.ply"
TARGET_PLY = r"C:\Users\27412\PycharmProjects\sys_dataset\Jiandu_chip\output_oblique_mid\point_clouds\two_step_faces\oblique_final_face_4_fractureB_continuous_0049.ply"
GT_JSON = r"C:\Users\27412\PycharmProjects\3Dproject\mydcp\fracture_dataset\labels\sample_00017.json"

RPMNET_WEIGHT = r"C:\Users\27412\PycharmProjects\3Dproject\mydcp\RPMNet\partial-trained.pth"
RIENET_WEIGHT = r"C:\Users\27412\PycharmProjects\3Dproject\mydcp\RIENet\pretrained\modelnet\model.best.t7"
PRNET_WEIGHT = r"C:\Users\27412\PycharmProjects\3Dproject\mydcp\prnet\checkpoints\latest.pth"

NUM_POINTS = 2048
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ====================== 工具函数 ======================
def load_ply(path, num_points=NUM_POINTS):
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    if len(pts) > num_points:
        pcd = pcd.farthest_point_down_sample(num_points)
        pts = np.asarray(pcd.points, dtype=np.float32)
    else:
        idx = np.random.choice(len(pts), num_points, replace=True)
        pts = pts[idx]

    centroid = pts.mean(axis=0)
    pts -= centroid
    scale = np.max(np.linalg.norm(pts, axis=1)) + 1e-6
    print(f"[{os.path.basename(path)}] 中心: {centroid.round(4)}, 原始缩放: {scale:.4f}")
    return pts, centroid, scale


def load_gt(json_path, src_cent, tgt_cent, common_scale):
    with open(json_path, 'r') as f:
        data = json.load(f)
    if "relative_transform" in data:
        T = np.array(data["relative_transform"], dtype=np.float32)
    else:
        R = np.array(data["rotation_matrix"], dtype=np.float32)
        t = np.array(data["translation_vector"], dtype=np.float32)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = t
    R_gt = T[:3, :3]
    t_gt = T[:3, 3]

    t_gt_norm = (t_gt - tgt_cent + R_gt @ src_cent) / common_scale
    print(f"[GT] 归一化后 t_norm = {np.linalg.norm(t_gt_norm):.4f}")
    return R_gt, t_gt_norm


def to_tensor(pts):
    return torch.tensor(pts, dtype=torch.float32).unsqueeze(0).to(DEVICE)


def rotation_error(R_pred, R_gt):
    R_diff = R_pred @ R_gt.T
    trace = np.clip((np.trace(R_diff) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(trace))


def translation_error(t_pred, t_gt):
    return np.linalg.norm(t_pred - t_gt)


def icp_refine(src_np, tgt_np, R_init=None, t_init=None):
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(src_np)
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(tgt_np)

    tgt.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.12, max_nn=30))
    src.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.12, max_nn=30))

    if R_init is None: R_init = np.eye(3, dtype=np.float64)
    if t_init is None: t_init = np.zeros(3, dtype=np.float64)

    trans_init = np.eye(4, dtype=np.float64)
    trans_init[:3, :3] = R_init
    trans_init[:3, 3] = t_init

    diameter = max(np.max(np.linalg.norm(src_np, axis=1)), np.max(np.linalg.norm(tgt_np, axis=1))) * 2
    max_dist = max(0.05, diameter * 0.06)

    reg = o3d.pipelines.registration.registration_icp(
        src, tgt, max_dist, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=120)
    )
    print(f"   ICP → Fitness={reg.fitness:.4f} | RMSE={reg.inlier_rmse:.6f}")
    return reg.transformation[:3, :3], reg.transformation[:3, 3]


# ====================== 可视化函数 ======================

# ====================== 可视化函数（按第二版重写）======================

def visualize_registration(source, target, T_est, method_name,
                           color_source=[0.97, 0.77, 0.02],
                           color_target=[0.01, 0.73, 0.97]):
    """
    可视化配准结果（第二版风格）
    - source / target : open3d.geometry.PointCloud
    - T_est           : (4, 4) numpy 变换矩阵
    """
    # 复制点云，避免修改原始数据
    src_vis = o3d.geometry.PointCloud(source)
    tgt_vis = o3d.geometry.PointCloud(target)

    # 统一设置颜色
    src_vis.colors = o3d.utility.Vector3dVector(
        np.tile(color_source, (len(src_vis.points), 1))
    )
    tgt_vis.colors = o3d.utility.Vector3dVector(
        np.tile(color_target, (len(tgt_vis.points), 1))
    )

    # 用 4x4 矩阵直接变换
    src_vis.transform(T_est)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"{method_name} 配准结果", width=1200, height=800)
    vis.add_geometry(src_vis)
    vis.add_geometry(tgt_vis)

    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)

    vis.run()
    vis.destroy_window()


def visualize_all_results(source, target, results_dict, num_trials=1):
    """
    可视化所有方法的配准结果
    - source / target : open3d.geometry.PointCloud（原始完整点云）
    - results_dict    : {method_name: [(R_np, t_np), ...]}
    """
    color_source = [0.97, 0.77, 0.02]  # 黄色
    color_target = [0.01, 0.73, 0.97]  # 青色

    print("\n" + "=" * 70)
    print("  【开始可视化所有配准结果】")
    print(f"  黄色 = 变换后的源点云  |  青色 = 目标点云")
    print("=" * 70 + "\n")

    total = len(results_dict)
    for idx, (method_name, T_list) in enumerate(sorted(results_dict.items()), 1):
        if len(T_list) == 0:
            print(f"  [{idx}/{total}] {method_name} 无结果，跳过")
            continue

        trial_idx = min(num_trials - 1, len(T_list) - 1)
        R_np, t_np = T_list[trial_idx]   # 拆包 (R, t)

        # 组装 4x4 矩阵
        T_est = np.eye(4, dtype=np.float64)
        T_est[:3, :3] = R_np.astype(np.float64)
        T_est[:3, 3]  = t_np.astype(np.float64)

        print(f"  [{idx}/{total}] 显示 {method_name} 配准效果... (关闭窗口后显示下一个)")
        visualize_registration(source, target, T_est, method_name,
                               color_source, color_target)

    print("\n" + "=" * 70)
    print("  【可视化全部完成】")
    print("=" * 70 + "\n")

"""
def visualize_all_results(src_np, tgt_np, results_dict):
    
    依次可视化所有方法的配准结果

    参数：
    - src_np: 源点云 numpy 数组 (N, 3)（扰动后的）
    - tgt_np: 目标点云 numpy 数组 (N, 3)
    - results_dict: {method_name: (R_est, t_est)} 字典
    
    color_source = [0.97, 0.77, 0.02]  # 黄色
    color_target = [0.01, 0.73, 0.97]  # 青色

    total = len(results_dict)
    print("\n" + "=" * 70)
    print("  🎨 【开始可视化所有配准结果】")
    print(f"  黄色 = 变换后的源点云  |  青色 = 目标点云")
    print("=" * 70 + "\n")

    for idx, (method_name, (R_est, t_est)) in enumerate(results_dict.items(), 1):
        print(f"  [{idx}/{total}] 显示 {method_name} 配准效果... (关闭窗口后显示下一个)")
        visualize_registration(src_np, tgt_np, R_est, t_est, method_name,
                               color_source, color_target)

    print("\n" + "=" * 70)
    print("  ✅ 【可视化全部完成】")
    print("=" * 70 + "\n")
"""

# ====================== compute_rigid_transform ======================
def compute_rigid_transform(a, b, weights):
    B, N, _ = a.shape
    weights = weights.view(B, N)
    weights_normalized = weights.unsqueeze(-1) / (weights.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-5)

    centroid_a = (a * weights_normalized).sum(dim=1)
    centroid_b = (b * weights_normalized).sum(dim=1)

    a_centered = a - centroid_a.unsqueeze(1)
    b_centered = b - centroid_b.unsqueeze(1)

    cov = a_centered.transpose(-2, -1) @ (b_centered * weights_normalized)

    u, s, v = torch.svd(cov)
    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[:, :, 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    rot_mat = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)

    translation = -rot_mat @ centroid_a.unsqueeze(-1) + centroid_b.unsqueeze(-1)
    translation = translation.squeeze(-1)

    return rot_mat, translation

# ====================== 模型定义 ======================
class RPMNet(nn.Module):
    def __init__(self, num_iters=3):
        super().__init__()
        self.num_iters = num_iters
        self.feat_net = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU()
        )
        self.weight_net = nn.Sequential(
            nn.Conv1d(512, 256, 1), nn.ReLU(),
            nn.Conv1d(256, 128, 1), nn.ReLU(),
            nn.Conv1d(128, 1, 1), nn.Sigmoid()
        )

    def forward(self, src, tgt):
        B, N, _ = src.shape
        src_curr = src.clone()
        R_accum = torch.eye(3, device=src.device).unsqueeze(0).repeat(B, 1, 1)
        t_accum = torch.zeros(B, 3, device=src.device)

        for _ in range(self.num_iters):
            f_src = self.feat_net(src_curr.transpose(1, 2))
            f_tgt = self.feat_net(tgt.transpose(1, 2))
            sim = torch.bmm(f_src.transpose(1, 2), f_tgt) / (256 ** 0.5)
            soft_corr = F.softmax(sim, dim=-1)
            tgt_corr = torch.bmm(soft_corr, tgt)

            feat_cat = torch.cat([f_src, torch.bmm(f_tgt, soft_corr.transpose(1, 2))], dim=1)
            weights = self.weight_net(feat_cat).squeeze(1)

            R, t = compute_rigid_transform(src_curr, tgt_corr, weights)
            src_curr = (src_curr @ R.transpose(1, 2)) + t.unsqueeze(1)
            R_accum = R @ R_accum
            t_accum = (R @ t_accum.unsqueeze(-1)).squeeze(-1) + t

        return R_accum, t_accum


class RIENet(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_nn = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 1)
        )
        self.weight_fn = nn.Sequential(
            nn.Conv1d(256, 128, 1), nn.ReLU(),
            nn.Conv1d(128, 64, 1), nn.ReLU(),
            nn.Conv1d(64, 1, 1), nn.Sigmoid()
        )

    def forward(self, src, tgt):
        f_src = self.emb_nn(src.transpose(1, 2))
        f_tgt = self.emb_nn(tgt.transpose(1, 2))
        sim = torch.bmm(f_src.transpose(1, 2), f_tgt) / (256 ** 0.5)
        attn = F.softmax(sim, dim=-1)
        tgt_corr = torch.bmm(attn, tgt)
        weights = self.weight_fn(f_src).squeeze(1)
        R, t = compute_rigid_transform(src, tgt_corr, weights)
        return R, t


class PRNet(nn.Module):
    def __init__(self, num_iters=3):
        super().__init__()
        self.num_iters = num_iters
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 1)
        )

    def forward(self, src, tgt):
        B, N, _ = src.shape
        src_curr = src.clone()
        R_accum = torch.eye(3, device=src.device).unsqueeze(0).repeat(B, 1, 1)
        t_accum = torch.zeros(B, 3, device=src.device)

        for _ in range(self.num_iters):
            f = self.encoder(src_curr.transpose(1, 2))
            sim = torch.bmm(f.transpose(1, 2), f) / (256 ** 0.5)
            attn = F.softmax(sim, dim=-1)
            tgt_corr = torch.bmm(attn, tgt)
            weights = torch.ones(B, N, device=src.device)
            R, t = compute_rigid_transform(src_curr, tgt_corr, weights)
            src_curr = (src_curr @ R.transpose(1, 2)) + t.unsqueeze(1)
            R_accum = R @ R_accum
            t_accum = (R @ t_accum.unsqueeze(-1)).squeeze(-1) + t
        return R_accum, t_accum


# ====================== 权重加载 ======================
def load_model(model_class, weight_path, *args, **kwargs):
    model = model_class(*args, **kwargs).to(DEVICE)
    if not os.path.exists(weight_path):
        print(f"❌ 权重不存在 → 随机初始化")
        return model
    try:
        state = torch.load(weight_path, map_location=DEVICE, weights_only=False)
        if isinstance(state, dict):
            for k in ['model', 'state_dict', 'prnet', 'module']:
                if k in state:
                    state = state[k]
                    break
            state = {k.replace('module.', ''): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"✅ {model_class.__name__} 加载完成 (缺失:{len(missing)}  多余:{len(unexpected)})")
        return model
    except Exception as e:
        print(f"❌ 加载失败: {str(e)[:120]}")
        return model
def load_ply_full(path):
    """加载完整点云，不下采样，仅做归一化"""
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    centroid = pts.mean(axis=0)
    pts -= centroid
    scale = np.max(np.linalg.norm(pts, axis=1)) + 1e-6
    print(f"[FULL {os.path.basename(path)}] 点数: {len(pts)}, 缩放: {scale:.4f}")
    return pts, centroid, scale

def compute_rmse_fitness(src, tgt, R, t):
    src_trans = (R @ src.T).T + t
    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src_trans)

    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(tgt)

    eval = o3d.pipelines.registration.evaluate_registration(
        src_pcd, tgt_pcd, max_correspondence_distance=0.05
    )
    return eval.inlier_rmse, eval.fitness


# ====================== 主程序（已添加可视化） ======================
def main():
    np.random.seed(42)
    print(f"🎯 使用设备: {DEVICE}\n")

    print("=" * 70)
    print("📁 加载点云 + 统一尺度")
    print("=" * 70)
    src_raw, src_cent, src_scale = load_ply(SOURCE_PLY)
    tgt_raw, tgt_cent, tgt_scale = load_ply(TARGET_PLY)

    common_scale = max(src_scale, tgt_scale)
    src_raw /= common_scale
    tgt_raw /= common_scale
    print(f"✅ 统一缩放因子: {common_scale:.4f}")

    print("\n" + "=" * 70)
    print("📋 加载 GT")
    print("=" * 70)
    R_gt = np.eye(3, dtype=np.float32)
    t_gt = np.zeros(3, dtype=np.float32)

    print("\n" + "=" * 70)
    print("🔀 极小随机扰动（仅用于测试，最终可设为0）")
    print("=" * 70)
    angle = np.random.uniform(-2, 2)
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis) + 1e-8
    R_rand = Rotation.from_rotvec(angle * np.pi / 180 * axis).as_matrix().astype(np.float32)
    t_rand = np.random.uniform(-0.03, 0.03, 3).astype(np.float32)
    src_perturbed = (R_rand @ src_raw.T).T + t_rand
    print(f"扰动角度 ≈ ±{angle:.2f}°")

    src_t = to_tensor(src_perturbed)
    tgt_t = to_tensor(tgt_raw)

    methods = [
        ("RPMNet", RPMNet, RPMNET_WEIGHT, {"num_iters": 10}),
        ("RIENet", RIENet, RIENET_WEIGHT, {}),
        ("PRNet", PRNet, PRNET_WEIGHT, {"num_iters": 10}),
    ]

    print("\n" + "=" * 70)
    print("🚀 只使用模型原始输出（已移除ICP）")
    print("=" * 70)

    vis_results = {}

    for name, cls, wpath, kwargs in methods:
        print(f"\n{'=' * 70}")
        print(f"🚀 运行 {name}（num_iters=10）")
        print(f"{'=' * 70}")

        model = load_model(cls, wpath, **kwargs)
        model.eval()

        t0 = time.time()
        with torch.no_grad():
            R_pred, t_pred = model(src_t, tgt_t)
        elapsed = time.time() - t0

        R_np = R_pred[0].cpu().numpy()
        t_np = t_pred[0].cpu().numpy()

        r_err = rotation_error(R_np, R_gt)
        t_err = translation_error(t_np, t_gt)

        print(f"   📍 模型原始输出 → R误差 {r_err:.3f}° | t误差 {t_err:.4f} | 耗时 {elapsed:.3f}s")
        rmse, fitness = compute_rmse_fitness(src_raw, tgt_raw, R_np, t_np)
        print(f"   {name}: RMSE = {rmse:.6f}, Fitness = {fitness:.4f}")

        vis_results[name] = [(R_np, t_np)]

    print("\n✅ 运行完成！")

    # ⭐ 重新读取原始完整PLY用于可视化（不下采样）
    source_full = o3d.io.read_point_cloud(SOURCE_PLY)
    target_full = o3d.io.read_point_cloud(TARGET_PLY)

    # 统一归一化（与推理时保持同一坐标系）
    src_full_pts = np.asarray(source_full.points, dtype=np.float32)
    tgt_full_pts = np.asarray(target_full.points, dtype=np.float32)

    src_full_pts -= src_cent
    tgt_full_pts -= tgt_cent
    src_full_pts /= common_scale
    tgt_full_pts /= common_scale

    # 对完整源点云施加同样的随机扰动
    src_full_pts = (R_rand @ src_full_pts.T).T + t_rand

    # 写回 Open3D 点云对象
    source_full.points = o3d.utility.Vector3dVector(src_full_pts)
    target_full.points = o3d.utility.Vector3dVector(tgt_full_pts)

    # ⭐ 可视化（传 Open3D 对象）
    visualize_all_results(source_full, target_full, vis_results, num_trials=1)


if __name__ == "__main__":
    main()