import numpy as np
import trimesh
import torch
import torch.nn.functional as F
import numpy as np
import trimesh
# torch version: torch-2.5.1 + cu118

def build_sdf_grid(mesh: trimesh.Trimesh, resolution: int = 64):
    bounds_min = mesh.bounds[0].copy()
    bounds_max = mesh.bounds[1].copy()
    padding = (bounds_max - bounds_min) * 0.05
    bounds_min -= padding
    bounds_max += padding

    lin = [np.linspace(bounds_min[i], bounds_max[i], resolution) for i in range(3)]
    xx, yy, zz = np.meshgrid(*lin, indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)

    chunk_size = 10_000
    distances = np.empty(len(pts), dtype=np.float32)
    signs     = np.empty(len(pts), dtype=np.float32)
    print("len")
    print(len(pts))
    for start in range(0, len(pts), chunk_size):
        end   = min(start + chunk_size, len(pts))
        print(end)
        chunk = pts[start:end]
        _, d, _ = trimesh.proximity.closest_point(mesh, chunk)
        distances[start:end] = d.astype(np.float32)
        signs[start:end]     = np.where(mesh.contains(chunk), -1.0, 1.0)

    sdf = (distances * signs).reshape(resolution, resolution, resolution)
    return torch.tensor(sdf), bounds_min, bounds_max


def query_sdf(vertices: torch.Tensor, sdf_grid: torch.Tensor,
              bounds_min: np.ndarray, bounds_max: np.ndarray) -> torch.Tensor:
    """
    Differentiable SDF query via trilinear interpolation (torch grid_sample).
    Returns SDF value at each vertex position; negative = inside, positive = outside.
    """
    bmin = torch.tensor(bounds_min, dtype=torch.float32, device=vertices.device)
    bmax = torch.tensor(bounds_max, dtype=torch.float32, device=vertices.device)

    # Normalize vertex positions to [-1, 1] for grid_sample
    coords = 2.0 * (vertices - bmin) / (bmax - bmin) - 1.0  # (V, 3)

    # grid_sample 3D expects input  (N, C, D, H, W)  with axes (z, y, x)
    # sdf_grid is (R_x, R_y, R_z)  →  permute to (R_z, R_y, R_x)
    sdf_in = sdf_grid.permute(2, 1, 0).unsqueeze(0).unsqueeze(0).to(vertices.device)
    grid   = coords.view(1, 1, 1, -1, 3)   # (1, 1, 1, V, xyz)

    out = F.grid_sample(sdf_in, grid, mode="bilinear",
                        align_corners=True, padding_mode="border")
    return out.view(-1)  # (V,)

def build_laplacian(mesh: trimesh.Trimesh) -> torch.Tensor:
    n = len(mesh.vertices)
    edges = mesh.edges_unique           # (E, 2), undirected

    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    degree = np.bincount(src, minlength=n).astype(np.float32)

    off_diag_vals = -1.0 / degree[src]  # -1/deg for each off-diagonal entry
    diag_vals     = np.ones(n, dtype=np.float32)

    rows = np.concatenate([src, np.arange(n)])
    cols = np.concatenate([dst, np.arange(n)])
    vals = np.concatenate([off_diag_vals, diag_vals])

    idx = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
    val = torch.tensor(vals, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()

def mesh_volume(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    return torch.abs((v0 * torch.cross(v1, v2, dim=1)).sum() / 6.0)


def optimize_mesh(
    original_mesh: trimesh.Trimesh,
    n_iters:       int   = 1000,
    lr:            float = 1e-3,
    w_smooth:      float = 1.0,    # weight: Laplacian smoothness
    w_inside:      float = 100.0,  # weight: stay inside original mesh
    w_volume:      float = 0.1,    # weight: maximise volume
    sdf_resolution: int  = 64,     # higher = tighter inside constraint
) -> trimesh.Trimesh:
    """
    Optimises the vertex positions of `original_mesh` to:
      1. Be smoother (minimise Laplacian energy)
      2. Stay strictly inside the original mesh (SDF penalty)
      3. Maximise enclosed volume
    Topology (faces) is kept fixed throughout.
    """
    # --- Precompute ---
    print("Building SDF grid …")
    sdf_grid, bounds_min, bounds_max = build_sdf_grid(original_mesh, sdf_resolution)

    print("Building Laplacian …")
    L = build_laplacian(original_mesh)

    faces = torch.tensor(original_mesh.faces, dtype=torch.long)
    verts = torch.tensor(original_mesh.vertices.copy(),
                         dtype=torch.float32, requires_grad=True)

    optimizer = torch.optim.Adam([verts], lr=lr)
    # Gradually cool the learning rate so fine details settle
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_iters)

    initial_vol = mesh_volume(verts.detach(), faces).item()
    print(f"Initial volume : {initial_vol:.6f}")

    # --- Optimisation loop ---
    for i in range(n_iters):
        optimizer.zero_grad()

        # 1. Smoothness loss  ── penalise each vertex deviating from neighbour mean
        Lv          = torch.sparse.mm(L, verts)         # (V, 3)
        smooth_loss = (Lv ** 2).mean()

        # 2. Inside-mesh loss ── penalise vertices that have drifted outside
        sdf_vals    = query_sdf(verts, sdf_grid, bounds_min, bounds_max)
        inside_loss = F.relu(sdf_vals).pow(2).mean()    # zero when fully inside

        # 3. Volume loss      ── maximise → minimise negative volume
        #    Normalised by initial volume so the scale is ~1 at the start
        vol         = mesh_volume(verts, faces)
        vol_loss    = -(vol / initial_vol)

        loss = w_smooth * smooth_loss + w_inside * inside_loss + w_volume * vol_loss
        loss.backward()

        optimizer.step()
        scheduler.step()

        if i % 100 == 0:
            print(f"[{i:5d}/{n_iters}]  "
                  f"loss={loss.item():9.5f}  "
                  f"smooth={smooth_loss.item():.5f}  "
                  f"inside={inside_loss.item():.5f}  "
                  f"vol={vol.item():.5f}")

    # --- Return result as a new trimesh ---
    final_verts = verts.detach().cpu().numpy()
    result = trimesh.Trimesh(vertices=final_verts,
                             faces=original_mesh.faces,
                             process=False)

    print(f"Final volume   : {mesh_volume(verts.detach(), faces).item():.6f}")
    return result


# def meshVolume(vertices: torch.Tensor, faces: torch.Tensor):
#     v0 = vertices[faces[:, 0]]
#     v1 = vertices[faces[:, 1]]
#     v2 = vertices[faces[:, 2]]
#     volume = torch.sum(v0 * torch.cross(v1, v2, dim=1)) / 6.0
#     volume = torch.sum(v0 * torch.cross(v1, v2, dim=1)) / 6.0
#     return torch.abs(volume)

# def triangleAngles(opposites, left, right):
#     cos = (left**2 + right**2 - opposites**2) / (2 * left * right)
#     return torch.acos(cos.clamp(-1.0, 1.0))



# def triangleAngles(opposites, left, right):
#     cos = (left**2 + right**2 - opposites**2) / (2 * left * right)
#     return torch.acos(cos.clamp(-1.0, 1.0))



# class ColorSpaceOptimizer:
#     def __init__(self, trimesh: trimesh.Trimesh):
#         self.mesh = trimesh
#         self.original_mesh = trimesh.copy()
#         # print(len(self.mesh.vertices))
#         self.iterations = 5

#         # Containment:
#         self.containment_weight = 1.0

#         # Curvature:
#         self.curvature_weight = 1.0
#         self.elliptic_penalty = 2.0
#         self.hyperbolic_penalty = 1.0
#         self.elliptic_threshold = 1.0
#         self.hyperbolic_threshold = 1.0
#         self.threshold_penalty = 1.0

#         # Volume:
#         self.volume_weight = 1.0

#     def containment(self, mesh: trimesh.Trimesh):
#         # loss = 0
#         all_containment = self.original_mesh.contains(mesh.vertices).astype(np.float64)
#         all_containment = 1.0 - all_containment
#         # for vert in mesh.vertices:
#             # loss += 1.0 - float(insideMesh(vert, self.original_mesh))

#         # why is this immediately high 
#         return np.sum(all_containment)
    
#     def curvature(self, mesh: trimesh.Trimesh):
#         # elliptic_penalty and hyperbolic_penalty are hyperparameters; elliptic_penalty should be larger than hyperbolic_penalty
#         # threshold is a hyperparameter representing the maximum curvature to allow
#         # should threshold be imposed before or after penalty? could incorporate relative difference here
#         # could also add a separate average penalty outside curvature weight
#         curvatures = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, 0)
#         elliptic_max = -1 * np.min(curvatures)
#         hyperbolic_max = np.max(curvatures)
#         curvatures[curvatures < 0] *= self.elliptic_penalty
#         curvatures[curvatures > 0] *= self.hyperbolic_penalty
#         total_curvature = np.sum(curvatures ** 2)
#         if elliptic_max > self.elliptic_threshold or hyperbolic_max > self.hyperbolic_threshold:
#             total_curvature += self.threshold_penalty
#         return total_curvature 
            
    
#     def volume(self, mesh: trimesh.Trimesh):
#         return mesh.volume
    
#     def loss_fn(self, vertices):
#         # print(self.containment(mesh))
#         # print(self.curvature(mesh))
#         # print(self.volume(mesh))
#         mesh = trimesh.Trimesh(vertices=vertices.reshape(-1, 3), faces=self.mesh.faces)
#         loss = (self.containment_weight * self.containment(mesh) + self.curvature_weight * self.curvature(mesh) - self.volume_weight * self.volume(mesh))
#         # print(loss)
#         return loss
    
#     def optimizeMesh(self):
#         res = minimize(fun=self.loss_fn, x0=self.mesh.vertices.flatten(), method="Powell", options={"maxiter": self.iterations, "disp": True})

#         final_mesh = trimesh.Trimesh(vertices=res.x.reshape(-1, 3), faces=self.mesh.faces)
#         self.original_mesh.show()
#         final_mesh.show()
#         return final_mesh
    
# class ColorSpaceTorchOptimizer:
#     def __init__(self, mesh: trimesh.Trimesh, device="cuda", 
#                  containment_weight=2000.0, curvature_weight=3000.0, volume_weight=0.0001,
#                  save_intermediate_meshes=True, print_updates=True,
#                  iterations=5000):
#         self.device = device

#         self.vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
#         self.faces = torch.tensor(mesh.faces, dtype=torch.int64, device=device)

#         self.original_vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
#         self.iterations = iterations

#         self.deform_verts = torch.zeros_like(self.vertices, device=self.device, requires_grad=True)
         
#         # with torch.no_grad():
#             # self.deform_verts[:, 0] = 25
#         self.optimizer = torch.optim.SGD([self.deform_verts], lr=1e-5, momentum=0.75)
        
#         # Containment:
#         self.containment_weight = containment_weight

#         # Curvature:
#         self.curvature_weight = curvature_weight

#         # Volume:
#         self.volume_weight = volume_weight

#         self.save_intermediate_meshes = save_intermediate_meshes
#         self.print_updates = print_updates

#         self.intermediate_meshes = []
#         self.curvatures = []
#         self.colors = mesh.visual.vertex_colors

#     def gaussian_curvature(self, vertices: torch.Tensor, faces: torch.Tensor):
#         v0 = vertices[faces[:, 0]]
#         v1 = vertices[faces[:, 1]]
#         v2 = vertices[faces[:, 2]]

#         e1 = (v1 - v2).norm(dim=1)
#         e2 = (v2 - v0).norm(dim=1)
#         e3 = (v0 - v1).norm(dim=1)

#         mesh_angles = torch.stack([triangleAngles(e1, e2, e3), triangleAngles(e2, e3, e1), triangleAngles(e3, e1, e2)], dim=1)
#         vertex_angles = torch.full((vertices.shape[0],), 2*torch.pi, device=self.device)

#         for i in range(3):
#             vertex_angles = vertex_angles.index_add(0, faces[:, i], -mesh_angles[:, i])
        
#         return vertex_angles
    
#     def build_adjacency(self, num_vertices, faces):
#         neighbors = [[] for _ in range(num_vertices)]
#         for f in faces:
#             i, j, k = f
#             neighbors[i] += [j, k]
#             neighbors[j] += [i, k]
#             neighbors[k] += [i, j]
#         return neighbors

#     def laplacian_loss(self, V, neighbors):
#         loss = 0.0
#         for i, nbrs in enumerate(neighbors):
#             if len(nbrs) == 0:
#                 continue
#             mean_neighbor = V[nbrs].mean(dim=0)
#             loss += torch.sum((V[i] - mean_neighbor) ** 2)
#         return loss / len(neighbors)

#     def containment(self, vertices, faces):
#         verts = torch.unsqueeze(vertices, dim=0)
#         original_verts = torch.unsqueeze(self.original_vertices, dim=0)
#         distances, _, _ = kal.metrics.trianglemesh.point_to_mesh_distance(verts, kal.ops.mesh.index_vertices_by_faces(original_verts, faces))
#         containment = kal.ops.mesh.check_sign(original_verts, faces, verts)[0].float()
#         c = torch.mean(distances * (1.0 - containment))
#         # print("con")
#         # print(c)
#         return c
    
#     def curvature_loss(self, vertices, faces, neighbors):
#         v0 = vertices[faces[:, 0]]
#         v1 = vertices[faces[:, 1]]
#         v2 = vertices[faces[:, 2]]

#         e0 = v2 - v1   
#         e1 = v0 - v2   
#         e2 = v1 - v0   

#         def cot(a, b):
#             dot = (a * b).sum(dim=1)
#             cross = torch.linalg.norm(torch.cross(a, b), dim=1)
#             cross = torch.clamp(cross, min=1e-6)
#             return dot / (cross)

#         cot0 = cot(e1, e2)   
#         cot1 = cot(e2, e0)  
#         cot2 = cot(e0, e1)   

#         V = vertices
#         L = torch.zeros_like(V)

#         i0 = faces[:, 0]
#         i1 = faces[:, 1]
#         i2 = faces[:, 2]

#         w = cot0.unsqueeze(1)
#         L.index_add_(0, i1, w * (V[i1] - V[i2]))
#         L.index_add_(0, i2, w * (V[i2] - V[i1]))

#         w = cot1.unsqueeze(1)
#         L.index_add_(0, i2, w * (V[i2] - V[i0]))
#         L.index_add_(0, i0, w * (V[i0] - V[i2]))

#         w = cot2.unsqueeze(1)
#         L.index_add_(0, i0, w * (V[i0] - V[i1]))
#         L.index_add_(0, i1, w * (V[i1] - V[i0]))

#         face_area = 0.5 * torch.linalg.norm(torch.cross(v1 - v0, v2 - v0), dim=1)
#         vertex_area = torch.zeros((vertices.shape[0],), device=vertices.device)
#         vertex_area.index_add_(0, i0, face_area / 3)
#         vertex_area.index_add_(0, i1, face_area / 3)
#         vertex_area.index_add_(0, i2, face_area / 3)

#         vertex_area = vertex_area.unsqueeze(1)
#         vertex_area = torch.clamp(vertex_area, min=1e-6)  # avoid blow-up

#         # mean curvature vector
#         H = L / (2.0 * vertex_area)

#         # final curvature energy
#         loss = (H ** 2).sum()

#         return loss
    
#     def curvature(self, vertices, faces, neighbors):
#         # laplacian_loss = mesh_laplacian_smoothing(mesh, method="uniform")
#         # edge_loss = mesh_edge_loss(mesh)
#         # normal_loss = mesh_normal_consistency(mesh)
#         # curve = 0.1 * laplacian_loss + edge_loss + 0.01 * normal_loss
#         # print("curve")
#         # print(curve)
#         # print(torch.max(curvature))
#         # print(outliers)
#         # print("cur")
#         # print(c)
#         # smoothed_verts = kal.metrics.trianglemesh.uniform_laplacian_smoothing(torch.unsqueeze(vertices, dim=1), faces)[0]
#         # distances = torch.norm(vertices - smoothed_verts, dim=1)
#         # c = distances.sum()
#         # print("curve")
#         # print(c)
#         threshold = 0.015
#         curvature = self.gaussian_curvature(vertices, faces)
#         outliers = (curvature > threshold).float().sum().item()
#         c = torch.sum(curvature ** 2)
#         return c
    
#         # return self.curvature_loss(vertices, faces, neighbors)

#         # loss = 0.0
#         # for i, nbrs in enumerate(neighbors):
#         #     if len(nbrs) == 0:
#         #         continue
#         #     mean_nbr = vertices[nbrs].mean(dim=0)
#         #     loss += ((vertices[i] - mean_nbr).norm())**2

#         # return loss / len(neighbors)
            
#     def volume(self, vertices, faces):
#         vol = meshVolume(vertices,faces)
#         # print("vol")
#         # vol = mesh.volume
#         # print(vol)
#         return -vol
    
#     def loss_fn(self, vertices, faces, neighbors):
#         return self.containment_weight * self.containment(vertices, faces) + self.curvature_weight * self.curvature(vertices, faces, neighbors) + self.volume_weight * self.volume(vertices, faces)
#         # return self.containment_weight* self.containment(vertices, faces)
#         # return self.curvature_weight * self.curvature(vertices, faces) + self.containment_weight * self.containment(vertices, faces)
    
#     def optimizeMesh(self, mesh: trimesh.Trimesh):
        
#         mesh_vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=self.device)
#         mesh_faces = torch.tensor(mesh.faces, dtype=torch.int64, device=self.device)
#         neighbors = self.build_adjacency(len(mesh_vertices), mesh_faces.tolist())
#         for i in tqdm(range(self.iterations)):
            
#             self.optimizer.zero_grad()
#             new_vertices = mesh_vertices + self.deform_verts
#             loss = self.loss_fn(new_vertices, mesh_faces, neighbors)
#             loss.backward()
#             self.optimizer.step()

#             if i % 10 == 0 and self.print_updates:
#                 print("iteration:")
#                 print(i)
#                 print("loss:")
#                 print(loss.item())
#                 if self.save_intermediate_meshes:
#                     v = (mesh_vertices + self.deform_verts).detach().cpu().numpy()
#                     intermediate_mesh = trimesh.Trimesh(vertices=v, faces=mesh_faces.detach().cpu().numpy())
#                     curvature = np.array(np.abs(trimesh.curvature.discrete_gaussian_curvature_measure(intermediate_mesh, v, 1.0)))            
#                     self.curvatures.append(curvature)
#                     self.intermediate_meshes.append(intermediate_mesh)

#         final_mesh = trimesh.Trimesh(vertices=(mesh_vertices + self.deform_verts).detach().cpu().numpy(), faces=mesh_faces.detach().cpu().numpy())
#         final_mesh.visual.vertex_colors = self.colors
#         return final_mesh
    
#     def getOriginalCurvature(self):
#         return self.curvatures[0]

#     def getFinalCurvature(self):
#         return self.curvatures[-1]

#     def getIntermediateMeshes(self):
#         curvature_min = np.min(np.array(self.curvatures))
#         curvature_max = np.max(np.array(self.curvatures))

#         gamma = 0.3
#         for mesh, curvature in zip(self.intermediate_meshes, self.curvatures):
#             colors = []
#             for c in curvature:
#                 # c = c / (1.0 + c)
#                 # if c < percentile5:
#                 # c = curvature_min
#                 c = (c - curvature_min) / (curvature_max - curvature_min)
#                 c = c ** gamma
#                 color = [c, c, c, 1.0]
#                 # print(c)
#                 colors.append(color)
#             mesh.visual.vertex_colors = np.array(colors)
#             # print(colors)
#         return self.intermediate_meshes
    
    

def save_single_view(mesh, rotation_matrix, filename):
    mesh_copy = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy())
    mesh_copy.visual.vertex_colors = mesh.visual.vertex_colors
    s = trimesh.Scene(mesh)
    s.apply_transform(rotation_matrix)
    # png = s.save_image(resolution=[800,800], visible=True)
    # Image.open(io.BytesIO(png)).save(filename + ".png")

def save_views(mesh: trimesh.Trimesh):
    r_quarter = trimesh.transformations.rotation_matrix(np.pi/2.0, [0, 1, 0])
    r_half = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])
    r_three_quarter = trimesh.transformations.rotation_matrix(3.0*np.pi/2.0, [0, 1, 0])

    save_single_view(mesh, r_quarter, "quarter_view")
    save_single_view(mesh, r_half, "half_view")
    save_single_view(mesh, r_three_quarter, "three_quarter_view")