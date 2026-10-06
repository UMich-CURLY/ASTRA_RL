import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from pathlib import Path


# Rigid Object: https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.assets.html#rigid-object
RIGID_OBJECT_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Robot",
    # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.sim.spawners.html#isaaclab.sim.spawners.from_files.UsdFileCfg
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(Path(__file__).parent / "BlueBoat.usd"),
        # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.sim.schemas.html#isaaclab.sim.schemas.RigidBodyPropertiesCfg
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,
            enable_gyroscopic_forces=False,
            retain_accelerations=True
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(
        #     collision_enabled=False,
        # )
    ),
)


HYDRODYNAMICS_CFG = {
    "COM_OFFSET_BODY": [-0.309, 0.0, -0.047],
    "MASS": 26.4,             # [kg]
    "RHO": 1000.0,            # water density
}


FOSSEN_CFG = {
    "mass": HYDRODYNAMICS_CFG["MASS"],
    "Iz": 6.306603,  # 4.8603,  # 4.962992,  # 6.306603,
    #linear drag coefficients
    "X_u": 0.593417,  # 0.457,  # 0.593420,  # 0.593417,
    "Y_v": 71.255440,  # 54.890,  # 71.275162,  # 71.255440,
    "Z_w": 0.0,  # 50.0
    "K_p": 0.0,  # 12.0
    "M_q": 0.0,  # 12.0
    "N_r": 4.478570,  # 6.384,  # 8.289728,  # 4.478570,
    #quadratic drag coefficients
    "X_uu": 6.9751547,  # 6.829,  # 8.867567,  # 6.975154,
    "Y_vv": 148.370300,  # 114.300,  # 148.418488,  # 148.370300,
    "Z_ww": 0.0,  # 50.0
    "K_pp": 0.0,  # 4.0
    "M_qq": 0.0,  # 4.0
    "N_rr": 13.159036,  # 18.757,  # 24.356270,  # 13.159036,
}


WIND_CFG = {
    "rho_air": 1.225,  # air density
    "C_x": 0.90,       # surge coefficient
    "C_y": 0.20,       # sway coefficient
    "C_n": 0.30,       # yaw coefficient
    "L_oa": 1.05,      # overall length
    "A_fw": 0.115000,  # frontal wind area (0.175*0.1*2 + 0.275*0.15 + 0.8*0.025 + 0.3*0.025 + 0.225*0.025*2)
    "A_lw": 0.163125,  # lateral wind area (1.05*0.1 + 0.225*0.125 + 0.95*0.025 + 0.025*0.25)
}


# starboard hull config
STBD_HULL_CFG = {
    "PATH": "{ENV_REGEX_NS}/Robot/BlueBoat/stbd_hull/stbd_hull", # USD path (mesh)
    "XFORM_OFFSET": [-0.23852, 0.33879, -0.15699],  # (x,y,z) offset from origin [m]
    "REST_VOLUME": 0.0285/2,       # submerged volume at rest [m^3]
    "FULL_VOLUME": 0.0565/2,       # total volume [m^3]
    "REST_WATERLINE_Z": -0.11973,  # at rest z offset (top_of_hull - waterline_at_rest) [m]
    "REST_CENTROID": [-0.30966,  0.32946, -0.21375], # center of buoyancy at rest [m]
}


# port hull config
PORT_HULL_CFG = {
    "PATH": "{ENV_REGEX_NS}/Robot/BlueBoat/port_hull/port_hull",
    "XFORM_OFFSET": [-0.23852, -0.33879, -0.15699],
    "REST_VOLUME": 0.0285/2,
    "FULL_VOLUME": 0.0565/2,
    "REST_WATERLINE_Z": -0.11973,
    "REST_CENTROID": [-0.30966, -0.32946, -0.21375],
}


# list of all defined hull configs
HULLS_CFG = [
    STBD_HULL_CFG,
    PORT_HULL_CFG,
]
