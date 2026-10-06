import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

class SysIdAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._env = env
        self._env_indices = torch.arange(self.num_envs, device=self.device)

        # sysID parameters
        self._env.num_of_sysID_params = cfg.num_of_sysID_params
        self._action_dim = cfg.num_of_sysID_params
        self._raw_actions = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        def get_range(base_val, deviation):
            dev_val = abs(base_val) * deviation
            return [base_val - dev_val, base_val + dev_val]

        ranges = [
            # [SYSID]
            get_range(self._env.sysid_params["Iz"], cfg.iz_deviation),
            get_range(self._env.sysid_params["X_u"], cfg.xu_deviation),
            get_range(self._env.sysid_params["Y_v"], cfg.yv_deviation),
            get_range(self._env.sysid_params["N_r"], cfg.nr_deviation),
            get_range(self._env.sysid_params["X_uu"], cfg.xuu_deviation),
            get_range(self._env.sysid_params["Y_vv"], cfg.yvv_deviation),
            get_range(self._env.sysid_params["N_rr"], cfg.nrr_deviation),
            get_range(self._env.sysid_params["c_rev"], cfg.c_rev_deviation),
            get_range(self._env.sysid_params["c_u"], cfg.c_u_deviation),
            # get_range(self._env.sysid_params["c_lag"], cfg.c_lag_deviation),
        ]
        self.param_min = torch.tensor([r[0] for r in ranges], device=self.device)
        self.param_max = torch.tensor([r[1] for r in ranges], device=self.device)


    @property
    def action_dim(self) -> int:
        return self._action_dim


    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions


    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions


    def process_actions(self, actions: torch.Tensor):
        self._raw_actions = actions
        norm_actions = torch.tanh(self._raw_actions[:, :self._env.num_of_sysID_params])
        self._processed_actions = self.param_min + 0.5 * (norm_actions + 1.0) * (self.param_max - self.param_min)
        self.apply_actions()


    def apply_actions(self):
        p = self._processed_actions
        # [SYSID]
        self._env.sysid_params["Iz"] = p[:, 0]
        self._env.sysid_params["X_u"] = p[:, 1]
        self._env.sysid_params["Y_v"] = p[:, 2]
        self._env.sysid_params["N_r"] = p[:, 3]
        self._env.sysid_params["X_uu"] = p[:, 4]
        self._env.sysid_params["Y_vv"] = p[:, 5]
        self._env.sysid_params["N_rr"] = p[:, 6]
        self._env.sysid_params["c_rev"] = p[:, 7]
        self._env.sysid_params["c_u"] = p[:, 8]
        # self._env.sysid_params["c_lag"] = p[:, 9]

    def reset(self, env_ids=None):
        if env_ids is None:
            self._raw_actions.zero_()
            self._processed_actions.zero_()
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0


@configclass
class SysIdActionCfg(ActionTermCfg):
    class_type = SysIdAction
    asset_name: str = "robot"

    # [SYSID]
    num_of_sysID_params = 9
    iz_deviation: float = 1.00
    xu_deviation: float = 1.00
    yv_deviation: float = 1.00
    nr_deviation: float = 1.00
    xuu_deviation: float = 1.00
    yvv_deviation: float = 1.00
    nrr_deviation: float = 1.00
    c_rev_deviation: float = 1.00
    c_u_deviation: float = 1.00
    # c_lag_deviation: float = 1.00
