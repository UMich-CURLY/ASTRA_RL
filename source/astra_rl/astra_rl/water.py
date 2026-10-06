import numpy as np
import warp as wp
import torch
import math

import omni.usd
from usdrt import Usd, UsdGeom, Vt, Sdf
from isaaclab.sim import SimulationContext
import omni.kit.commands


PROFILE_EXTENT = 410.0
PROFILE_RES = 8192
PROFILE_WAVENUM = 1000
MIN_WAVE_LENGTH = 0.1
MAX_WAVE_LENGTH = 250.0


wp.init()


@wp.func
def frac(a: float):
    return a - wp.floor(a)


@wp.func
def sqr(a: float):
    return a * a


@wp.func
def alpha_beta_spectrum(omega: float, peak_omega: float, alpha: float, beta: float, gravity: float):
    return (alpha * gravity * gravity / wp.pow(omega, 5.0)) * wp.exp(-beta * wp.pow(peak_omega / omega, 4.0))


@wp.func
def jonswap_peak_sharpening(omega: float, peak_omega: float, gamma: float):
    sigma = float(0.07)
    if omega > peak_omega:
        sigma = float(0.09)
    return wp.pow(gamma, wp.exp(-0.5 * sqr((omega - peak_omega) / (sigma * peak_omega))))


@wp.func
def jonswap_spectrum(omega: float, gravity: float, alpha: float, peak_omega: float, gamma: float):
    # https://www.wikiwaves.org/index.php/Ocean-Wave_Spectra#JONSWAP_Spectrum
    return jonswap_peak_sharpening(omega, peak_omega, gamma) * alpha_beta_spectrum(
        omega, peak_omega, alpha, 1.25, gravity
    )


@wp.func
def TMA_spectrum(omega: float, gravity: float, alpha: float, peak_omega: float, gamma: float, water_depth: float):
    # https://dl.acm.org/doi/10.1145/2791261.2791267
    omegaH = omega * wp.sqrt(water_depth / gravity)
    # omegaH = wp.max(0.0, wp.min(2.2, omegaH))
    # phi = 0.5 * omegaH * omegaH
    # if omegaH > 1.0:
    #     phi = 1.0 - 0.5 * sqr(2.0 - omegaH)
    if omegaH <= 1.0:
        phi = 0.5 * sqr(omegaH)
    elif omegaH < 2.0:
        phi = 1.0 - 0.5 * sqr(2.0 - omegaH)
    else:
        phi = float(1.0)
    return phi * jonswap_spectrum(omega, gravity, alpha, peak_omega, gamma)


@wp.kernel
def update_profile(
    profile: wp.array(dtype=wp.vec3),
    profile_res: int,
    profile_data_num: int,
    min_lambda: float,
    max_lambda: float,
    profile_extend: float,
    time: float,
    wind_speed: float,
    fetch_km: float,
    gamma: float,
    water_depth: float,
):
    x = wp.tid()
    randState = wp.rand_init(7)
    # sampling parameters
    omega0 = wp.sqrt(wp.tau * 9.80665 / min_lambda)
    omega1 = wp.sqrt(wp.tau * 9.80665 / max_lambda)
    omega_delta = wp.abs(omega1 - omega0) / float(profile_data_num)
    # we blend three displacements for seamless spatial profile tiling
    space_pos_1 = profile_extend * float(x) / float(profile_res)
    space_pos_2 = space_pos_1 + profile_extend
    space_pos_3 = space_pos_1 - profile_extend
    p1 = wp.vec2(0.0, 0.0)
    p2 = wp.vec2(0.0, 0.0)
    p3 = wp.vec2(0.0, 0.0)
    # CHANGE: compute alpha and peak_omega once per profile update
    fetch = 1000.0 * fetch_km
    alpha = 0.076 * wp.pow(wind_speed * wind_speed / (9.80665 * fetch), 0.22)
    peak_omega = 22.0 * wp.pow(wp.abs(9.80665 * 9.80665 / (wind_speed * fetch)), 1.0 / 3.0)
    for i in range(profile_data_num):
        omega = wp.abs(omega0 + (omega1 - omega0) * float(i) / float(profile_data_num))  # linear sampling of omega
        k = omega * omega / 9.80665
        phase = -time * omega + wp.randf(randState) * 2.0 * wp.pi
        amplitude = float(10000.0) * wp.sqrt(
            wp.abs(2.0 * omega_delta * TMA_spectrum(omega, 9.80665, alpha, peak_omega, gamma, water_depth))
        )
        p1 = wp.vec2(
            p1[0] + amplitude * wp.sin(phase + space_pos_1 * k), p1[1] - amplitude * wp.cos(phase + space_pos_1 * k)
        )
        p2 = wp.vec2(
            p2[0] + amplitude * wp.sin(phase + space_pos_2 * k), p2[1] - amplitude * wp.cos(phase + space_pos_2 * k)
        )
        p3 = wp.vec2(
            p3[0] + amplitude * wp.sin(phase + space_pos_3 * k), p3[1] - amplitude * wp.cos(phase + space_pos_3 * k)
        )
    # cubic blending coefficients
    s = float(float(x) / float(profile_res))
    c1 = float(2.0 * s * s * s - 3.0 * s * s + 1.0)
    c2 = float(-2.0 * s * s * s + 3.0 * s * s)
    disp_out = wp.vec3(
        (p1[0] + c1 * p2[0] + c2 * p3[0]) / float(profile_data_num),
        (p1[1] + c1 * p2[1] + c2 * p3[1]) / float(profile_data_num),
        0.0,
    )
    profile[x] = disp_out


@wp.kernel
def update_points(
    points: wp.array(dtype=wp.vec3),
    profile: wp.array(dtype=wp.vec3),
    profile_res: int,
    profile_extent: float,
    amplitude: float,
    center_pos: wp.vec3,
    clipmap_cell_size: float,
    direction_data: wp.array(dtype=wp.vec3), 
    direction_count: int,
    out_points: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    p_crd = wp.vec3(
        points[tid][0] + wp.floor(center_pos[0] / clipmap_cell_size) * clipmap_cell_size,
        points[tid][1],
        points[tid][2] + wp.floor(center_pos[2] / clipmap_cell_size) * clipmap_cell_size,
    )

    randState = wp.rand_init(7)
    disp_x = float(0.0)
    disp_y = float(0.0)
    disp_z = float(0.0)
    w_sum = float(0.0)
    for d in range(0, direction_count):
        dir_vec = direction_data[d]  # <- CHANGE: use precomputed direction_data kernel 
        dir_x = dir_vec[0]
        dir_y = dir_vec[1]
        dir_amp = dir_vec[2]
        rand_phase = wp.randf(randState)
        x_crd = (p_crd[0] * dir_x + p_crd[1] * dir_y) / profile_extent + rand_phase  # <- CHANGE: p_crd[2] to p_crd[1]
        pos_0 = int(wp.floor(x_crd * float(profile_res))) % profile_res
        if x_crd < 0.0:
            pos_0 = pos_0 + profile_res - 1
        pos_1 = int(pos_0 + 1) % profile_res
        p_disp_0 = profile[pos_0]
        p_disp_1 = profile[pos_1]
        w = frac(x_crd * float(profile_res))
        prof_height_x = dir_amp * float((1.0 - w) * p_disp_0[0] + w * p_disp_1[0])
        prof_height_y = dir_amp * float((1.0 - w) * p_disp_0[1] + w * p_disp_1[1])
        disp_x = disp_x + dir_x * prof_height_x
        disp_y = disp_y + dir_y * prof_height_x  # <- CHANGE: (prof_height_y) to (dir_y * prof_height_x)
        disp_z = disp_z + prof_height_y  # <- CHANGE: (dir_y * prof_height_x) to (prof_height_y)
        w_sum = w_sum + 1.0

    # write output vertex position
    out_points[tid] = wp.vec3(
        p_crd[0] + amplitude * disp_x / w_sum,
        p_crd[1] + amplitude * disp_y / w_sum,
        p_crd[2] + amplitude * disp_z / w_sum,
    )


@wp.kernel
def precompute_directions(
    direction_data: wp.array(dtype=wp.vec3),
    directionality: float,
    direction: float,
    direction_count: int
):
    d = wp.tid()
    r = float(d) * wp.tau / float(direction_count) + 0.02
    dir_x = wp.cos(r)
    dir_y = wp.sin(r)

    t = wp.abs(direction - r)
    if t > wp.pi:
        t = wp.tau - t
    t = t / wp.pi
    t = wp.pow(t, 1.2)

    dir_amp = (2.0 * t * t * t - 3.0 * t * t + 1.0) * 1.0 + (-2.0 * t * t * t + 3.0 * t * t) * (1.0 - directionality)
    dir_amp = dir_amp / (1.0 + 10.0 * directionality)

    direction_data[d] = wp.vec3(dir_x, dir_y, dir_amp)


@wp.kernel
def flat_mesh_kernel(points: wp.array(dtype=wp.vec3), out_points: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    out_points[i] = points[i]


class Water:
    device = None

    # USDRT
    stage = None
    water_path = "/World/Water"
    water_prim = None
    water_mesh = None

    # CPU mesh data
    orig_points = None

    # warp arrays
    wp_points = None
    wp_out_points = None
    grid_x = None
    grid_y = None
    grid_indices = None
    grid_shape = None
    height_field_np = None
    direction_data = None

    # 1D ocean profile for spectrum-based waves
    profile = None
    directions = 128
    wave_scale = 1.0
    directional_spreading = 0.75
    wind_direction_rad = 0.0
    wind_speed_ms = 5.0
    fetch_km = 100.0
    gamma = 3.3
    water_depth_m = 100.0

    # other wave vars
    t = 0.0
    is_waves_enabled = False
    is_initialized = False

    @staticmethod
    def initialize(device):
        Water.device = device

        # spawn the water
        omni.kit.commands.execute(
            "CreateMeshPrim",
            prim_type="Plane",
            half_scale=4096,
            u_verts_scale=256,
            v_verts_scale=256,
            prim_path="/World/Water"
        )

        # The commented out code beneath works to load a material saved as a USD
        # and apply it to the water. However, it doesn't load/look correct...
        # it might have to do with the material being hosted and loaded from AWS
        # (you cannot export NVIDIA base materials to MDL because they are on AWS)

        # # from pathlib import Path
        # stage = omni.usd.get_context().get_stage()
        # material_path = str(Path(__file__).parent / "Water_Material.usd")
        # prim_path = "/World/Looks/WaterMaterial"
        # stage.DefinePrim(prim_path).GetReferences().AddReference(material_path)

        # # bind material to the water
        # omni.kit.commands.execute(
        #     'BindMaterialCommand',
        #     prim_path="/World/Water",
        #     material_path="/World/Looks/WaterMaterial/Water",
        #     strength='weakerThanDescendants'
        # )

        # USDRT
        Water.stage = Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
        Water.water_path = Sdf.Path("/World/Water")
        Water.water_mesh = UsdGeom.Mesh(Water.stage.GetPrimAtPath(Water.water_path))

        # get points attribute from fabric cache
        orig_points_vt = Water.water_mesh.GetPointsAttr().Get()  # VtArray[GfVec3f]
        Water.orig_points = np.array(orig_points_vt, dtype=np.float32)

        # initialize warp arrays
        Water.wp_orig_points = wp.array(Water.orig_points, dtype=wp.vec3)
        Water.wp_points = wp.array(Water.orig_points, dtype=wp.vec3)
        Water.wp_out_points = wp.zeros(len(Water.orig_points), dtype=wp.vec3)
        Water.profile = wp.zeros(PROFILE_RES, dtype=wp.vec3)
        Water.direction_data = wp.zeros(Water.directions, dtype=wp.vec3)

        grid_x = np.unique(Water.orig_points[:, 0]).astype(np.float32)
        grid_y = np.unique(Water.orig_points[:, 1]).astype(np.float32)
        if grid_x.size * grid_y.size == len(Water.orig_points):
            Water.grid_x = grid_x
            Water.grid_y = grid_y
            Water.grid_shape = (grid_y.size, grid_x.size)
            Water.grid_indices = np.lexsort((Water.orig_points[:, 0], Water.orig_points[:, 1]))
            Water.height_field_np = None
        else:
            Water.grid_x = None
            Water.grid_y = None
            Water.grid_shape = None
            Water.grid_indices = None
            Water.height_field_np = None

        Water.t = 0.0
        Water.is_initialized = True


    @staticmethod
    def on_physics_step(dt):
        Water.t += dt

        # only update if waves are enabled
        if Water.is_waves_enabled:
            Water.update_mesh()


    @staticmethod
    def update_mesh():
        wp.launch(
            kernel=update_profile,
            dim=(PROFILE_RES,),
            inputs=[
                Water.profile,
                PROFILE_RES,
                PROFILE_WAVENUM,
                MIN_WAVE_LENGTH,
                MAX_WAVE_LENGTH,
                PROFILE_EXTENT,
                Water.t,
                Water.wind_speed_ms,
                Water.fetch_km,
                Water.gamma,
                Water.water_depth_m
            ]
        )

        wp.launch(
            kernel=precompute_directions,
            dim= Water.directions,
            inputs=[
                Water.direction_data,
                Water.directional_spreading,
                Water.wind_direction_rad,
                Water.directions,
            ]
        )

        wp.launch(
            kernel=update_points,
            dim=len(Water.orig_points),
            inputs=[
                Water.wp_points,
                Water.profile,
                PROFILE_RES,
                PROFILE_EXTENT,
                Water.wave_scale,
                wp.vec3(0.0, 0.0, 0.0),
                1.0,
                Water.direction_data,
                Water.directions,
                Water.wp_out_points,
            ]
        )

        pts = Water.wp_out_points.numpy().astype(np.float32)
        if Water.grid_indices is not None and Water.grid_shape is not None:
            Water.height_field_np = pts[Water.grid_indices, 2].reshape(Water.grid_shape)

        if SimulationContext.instance().render_mode == SimulationContext.RenderMode.FULL_RENDERING:
            verts = Vt.Vec3fArray(pts)
            Water.water_mesh.GetPointsAttr().Set(verts)


    @staticmethod
    def get_wave_z(X, Y, device):
        # enabled/valid check
        if not Water.is_waves_enabled or Water.height_field_np is None or Water.grid_x is None or Water.grid_y is None:
            if isinstance(X, torch.Tensor):
                return torch.zeros_like(X, device=device)
            return torch.tensor(0.0, device=device)

        x = X.clone().detach().to(device)
        y = Y.clone().detach().to(device)

        nx = int(Water.grid_shape[1])
        ny = int(Water.grid_shape[0])
        x_min = float(Water.grid_x[0])
        x_span = float(Water.grid_x[-1] - Water.grid_x[0])
        y_min = float(Water.grid_y[0])
        y_span = float(Water.grid_y[-1] - Water.grid_y[0])

        # normalize coordinates with tiling
        x_norm = torch.remainder((x - x_min) / x_span, 1.0) * float(nx)
        y_norm = torch.remainder((y - y_min) / y_span, 1.0) * float(ny)

        x0 = torch.floor(x_norm).to(torch.long) % nx
        x1 = (x0 + 1) % nx
        y0 = torch.floor(y_norm).to(torch.long) % ny
        y1 = (y0 + 1) % ny
        wx = x_norm - torch.floor(x_norm)
        wy = y_norm - torch.floor(y_norm)

        hf = torch.from_numpy(Water.height_field_np).to(device)

        # bilinear interpolation
        h00 = hf[y0, x0]
        h10 = hf[y0, x1]
        h01 = hf[y1, x0]
        h11 = hf[y1, x1]
        hx0 = (1.0 - wx) * h00 + wx * h10
        hx1 = (1.0 - wx) * h01 + wx * h11
        z = (1.0 - wy) * hx0 + wy * hx1

        return z.to(X.dtype)  # (N, 6)


    @staticmethod
    def get_nu_wave(root_pos_w, device):
        N = root_pos_w.shape[0]
        nu_wave = torch.zeros((N, 6), device=device, dtype=root_pos_w.dtype)

        if not Water.is_waves_enabled or Water.height_field_np is None:
            return nu_wave

        X = root_pos_w[:, 0]  # (N,)
        Y = root_pos_w[:, 1]  # (N,)
        Z = root_pos_w[:, 2]  # (N,)

        # https://en.wikipedia.org/wiki/Airy_wave_theory

        # compute spectrum properties dynamically
        gravity = 9.80665
        wind_speed = max(Water.wind_speed_ms, 1e-4)
        fetch = 1000.0 * max(Water.fetch_km, 1e-4)
        peak_omega = 22.0 * math.pow(abs(gravity * gravity / (wind_speed * fetch)), 1.0 / 3.0)
        k = peak_omega * peak_omega / gravity  # assume deep water, tanh(kh) -> 1
        phase_vel = peak_omega / k if k > 0 else 0.0

        # unit direction vector of the wind
        nx = math.cos(Water.wind_direction_rad)
        ny = math.sin(Water.wind_direction_rad)

        # base wave height at current XY position
        Z_surf = Water.get_wave_z(X, Y, device)

        # approximate spatial gradients using finite difference method
        delta = 0.1
        Z_x = Water.get_wave_z(X + delta, Y, device)
        Z_y = Water.get_wave_z(X, Y + delta, device)
        dZ_dx = (Z_x - Z_surf) / delta
        dZ_dy = (Z_y - Z_surf) / delta

        # depth decay
        Z_depth = Z - Z_surf
        decay = torch.exp(k * torch.clamp(Z_depth, max=0.0))

        # wave velocities
        Vx = nx * peak_omega * Z_surf * decay
        Vy = ny * peak_omega * Z_surf * decay
        Vz = phase_vel * (nx*dZ_dx + ny*dZ_dy) * decay

        zeros = torch.zeros_like(Vx)
        return torch.stack([Vx, Vy, Vz, zeros, zeros, zeros], dim=-1)  # (N, 6)
