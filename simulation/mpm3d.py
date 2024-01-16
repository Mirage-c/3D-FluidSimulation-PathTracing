export_path = "" # 输出路径，若为空字符串则不输出

#export_path = "results/" 

import numpy as np
import mcubes
import taichi as ti

ti.init(arch=ti.gpu)
dim, n_grid, steps, dt = 3, 32, 25, 4e-4

n_particles = n_grid**dim // 2 ** (dim - 1)
dx = 1 / n_grid

p_rho = 1
p_vol = (dx * 0.5) ** dim
p_mass = p_vol * p_rho
gravity = 9.8
bound = 3
E = 400

F_x = ti.Vector.field(dim, float, n_particles)
F_v = ti.Vector.field(dim, float, n_particles)
F_C = ti.Matrix.field(dim, dim, float, n_particles)
F_J = ti.field(float, n_particles)

F_grid_v = ti.Vector.field(dim, float, (n_grid,) * dim)
F_grid_m = ti.field(float, (n_grid,) * dim)

n_surface_grid = 32
F_grid_u = ti.field(float, (n_surface_grid,) * dim)

neighbour = (3,) * dim
 
interpolation_h = ti.field(ti.f64, shape=1)
C_pi = ti.field(ti.f64, shape=1)
cubic_spline = ti.field(ti.f64, shape=1)

C_pi[0] = np.pi
interpolation_h[0] = 1.0 # 归一化，用网格边长作为单位长度
interpolation_neighbour = (2 * 2 * int(interpolation_h[0]),) * dim

@ti.func
def local_coordinate(absolute_coordinate):
    normalized_coordinate = absolute_coordinate / dx
    base_grid = int(normalized_coordinate - 0.5)
    return base_grid, normalized_coordinate - base_grid


@ti.kernel
def substep():
    for I in ti.grouped(F_grid_m):
        F_grid_v[I] = ti.zero(F_grid_v[I])
        F_grid_m[I] = 0
    ti.loop_config(block_dim=n_grid)
    # [1] p2g
    for p in F_x:
        base, local_coord = local_coordinate(F_x[p])
        w = [0.5 * (1.5 - local_coord) ** 2, 0.75 - (local_coord - 1) ** 2, 0.5 * (local_coord - 0.5) ** 2]
        stress = -dt * 4 * E * p_vol * (F_J[p] - 1) / dx**2
        affine = ti.Matrix.identity(float, dim) * stress + p_mass * F_C[p]
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - local_coord) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            F_grid_v[base + offset] += weight * (p_mass * F_v[p] + affine @ dpos)
            F_grid_m[base + offset] += weight * p_mass
    # [2] 网格更新
    for I in ti.grouped(F_grid_m):
        if F_grid_m[I] > 0:
            F_grid_v[I] /= F_grid_m[I]
        F_grid_v[I][1] -= dt * gravity
        cond = (I < bound) & (F_grid_v[I] < 0) | (I > n_grid - bound) & (F_grid_v[I] > 0)
        F_grid_v[I] = ti.select(cond, 0, F_grid_v[I])
    # [3] g2p
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        base, local_coord = local_coordinate(F_x[p])
        w = [0.5 * (1.5 - local_coord) ** 2, 0.75 - (local_coord - 1) ** 2, 0.5 * (local_coord - 0.5) ** 2]
        new_v = ti.zero(F_v[p])
        new_C = ti.zero(F_C[p])
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - local_coord) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            g_v = F_grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
        F_v[p] = new_v
        F_x[p] += dt * F_v[p]
        F_J[p] *= 1 + dt * new_C.trace()
        F_C[p] = new_C


@ti.kernel
def init():
    for i in range(n_particles):
        F_x[i] = ti.Vector([ti.random() for i in range(dim)]) * 0.4 + 0.15
        F_J[i] = 1


def ViewProjection(a):
    phi, theta = np.radians(28), np.radians(32)

    a = a - 0.5
    x, y, z = a[:, 0], a[:, 1], a[:, 2]
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    x, z = x * cp + z * sp, z * cp - x * sp
    u, v = x, y * ct + z * st
    return np.array([u, v]).swapaxes(0, 1) + 0.5

@ti.func
def cubic_spline_kernel(r):
    q = r / interpolation_h[0]
    cubic_spline[0] = 3 / 2 / C_pi[0]
    if 0 <= q and q < 1:
        cubic_spline[0] *= 2 / 3 - q * q + 0.5 * q ** 3
    elif q < 2:
        cubic_spline[0] *= 1 / 6 * (2 - q) ** 3
    else:
        cubic_spline[0] *= 0
    return q / interpolation_h[0] ** 3 * cubic_spline[0]

@ti.kernel
def interpolation():
    ti.loop_config(block_dim=32)
    for I in ti.grouped(F_grid_u):
        F_grid_u[I] = 0

    
    Xp = F_x[0] * n_surface_grid
    base = int(Xp - 1)
    # print(Xp)
    # for offset in ti.grouped(ti.ndrange(*interpolation_neighbour)):
    #     grid_coord = base + offset
    #     r = (Xp - grid_coord).norm()
    #     F_grid_u[grid_coord] += cubic_spline_kernel(r)
    #     print(grid_coord,  r / interpolation_h[0], cubic_spline_kernel(r))

    for p in F_x:
        Xp = F_x[p] * n_surface_grid
        base = int(Xp - 1)
        for offset in ti.grouped(ti.ndrange(*interpolation_neighbour)):
            grid_coord = base + offset
            r = (Xp - grid_coord).norm()
            F_grid_u[grid_coord] += cubic_spline_kernel(r)


def marching_cube(scalar_field: np.array, frame_cnt):
    scalar_field = np.squeeze(scalar_field)
    # print(scalar_field)
    vertices, triangles = mcubes.marching_cubes(scalar_field, 0.05)
    mcubes.export_obj(vertices, triangles, export_path + "mpm3d_" + str(frame_cnt) +".obj")
    
def export_mesh(frame_cnt = 0):
    # pos = F_x.to_numpy()
    # writer = ti.tools.PLYWriter(num_vertices=n_particles)
    # writer.add_vertex_pos(pos[:, 0], pos[:, 1], pos[:, 2])
    # writer.export_frame(gui.frame, export_path)
    interpolation()
    marching_cube(F_grid_u.to_numpy(), frame_cnt)

def test():
    for s in range(steps):
        substep()
    pos = F_x.to_numpy()
    if export_path:
        export_mesh()


if __name__ == "__main__":
    init()
    # test()
    gui = ti.GUI("MPM3D", background_color=0x112F41)
    while gui.running and not gui.get_event(gui.ESCAPE):
        for s in range(steps):
            substep()
        pos = F_x.to_numpy()
        if export_path:
            export_mesh(gui.frame)
        gui.circles(ViewProjection(pos), radius=1.5, color=0x66CCFF)
        gui.show()
