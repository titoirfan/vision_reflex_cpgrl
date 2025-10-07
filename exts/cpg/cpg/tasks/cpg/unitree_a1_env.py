# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
from collections.abc import Sequence

import os
import numpy as np

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import ContactSensor, RayCaster

from .unitree_a1_env_cfg import UnitreeA1FlatEnvCfg, UnitreeA1RoughEnvCfg


class UnitreeA1Env(DirectRLEnv):
    cfg: UnitreeA1FlatEnvCfg | UnitreeA1RoughEnvCfg

    def __init__(self, cfg: UnitreeA1FlatEnvCfg | UnitreeA1RoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("trunk")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*_foot")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*_thigh")

        # Random heading tracking and standing still envs
        self.heading_targets = torch.zeros(self.num_envs, device=self.device)
        self.heading_tracking_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.standing_still_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        push_robot = self.cfg.events.push_robot
        if push_robot is not None and "push_duration" in push_robot.params:
            self.push_duration = push_robot.params["push_duration"]
            self.timestep_since_last_push = torch.full((self.num_envs, ), self.push_duration, dtype=torch.int, device=self.device)
            
            if "force_range" in push_robot.params:
                asset_cfg = push_robot.params["asset_cfg"]
                self.pushed_body_ids = asset_cfg.body_ids
                self.num_pushed_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else self._robot.num_bodies
            elif "velocity_range" in push_robot.params:
                self.push_vel_b = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)

        # Visualization
        self.set_debug_vis(self.cfg.debug_vis)

        # Timestep-wise logs for evaluation
        if self.cfg.eval_mode:
            self.log_time = torch.zeros(self.num_envs, 1, 1, dtype=torch.float, device=self.device)

            self.log_base_linear_vel = torch.zeros(self.num_envs, 3, 1, dtype=torch.float, device=self.device)
            self.log_base_angular_vel = torch.zeros(self.num_envs, 3, 1, dtype=torch.float, device=self.device)

            self.log_base_position = torch.zeros(self.num_envs, 3, 1, dtype=torch.float, device=self.device)
            self.log_terrain_origins = torch.zeros(self.num_envs, 3, 1, dtype=torch.float, device=self.device)

            self.log_joint_acc = torch.zeros(self.num_envs, 12, 1, dtype=torch.float, device=self.device)

            self.log_energy = torch.zeros(self.num_envs, 1, 1, dtype=torch.float, device=self.device)
            self.log_commands = torch.zeros(self.num_envs, 3, 1, dtype=torch.float, device=self.device)
            self.log_base_tracked_vel = torch.zeros(self.num_envs, 3, 1, dtype=torch.float, device=self.device)

            self.log_died = torch.zeros(self.num_envs, 1, 4, dtype=torch.bool, device=self.device)
            self.log_survival_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.log_survival_mask = torch.zeros(self.num_envs, 1, 1, dtype=torch.float, device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        if isinstance(self.cfg, UnitreeA1RoughEnvCfg):
            # we add a height scanner for perceptive locomotion
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = self.cfg.light
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

        # Process external forces applied to the robot if necessary
        self._process_external_forces()

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._update_commands()
        self._previous_actions = self._actions.clone()
        height_data = None
        if isinstance(self.cfg, UnitreeA1RoughEnvCfg):
            height_data = self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5

        sensory_obs = {
            "base_ang_vel": self._robot.data.root_ang_vel_b,
            "base_projected_gravity": self._robot.data.projected_gravity_b,
            "joint_pos": self._robot.data.joint_pos - self._robot.data.default_joint_pos,
            "joint_vel": self._robot.data.joint_vel,
            "height_scan": height_data,
        }
        if self.cfg.observation_noises.enable:
            self._apply_observation_noises(sensory_obs)

        if isinstance(self.cfg, UnitreeA1RoughEnvCfg):
            sensory_obs["height_scan"] = sensory_obs["height_scan"].clip(-1.0, 1.0)

        obs = torch.cat(
            [tensor for tensor in sensory_obs.values() if tensor is not None] + [self._commands, self._actions], dim=-1
        )
        observations = {"policy": obs}

        if self.cfg.eval_mode:
            self._update_eval_logs()

        return observations

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(
            torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1
        )
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )
        # undesired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)

        if self.cfg.eval_mode:
            died_persistent = torch.logical_or(died, self.log_died[:, :, -1].squeeze(1))
            self.log_died = torch.cat((self.log_died, died_persistent.reshape(-1, 1, 1)), dim=2)

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        # Log distance traveled
        distances = torch.norm(
            self._robot.data.root_pos_w[env_ids, :2] - self._terrain.env_origins[env_ids, :2], dim=1
        )

        # Apply curriculum
        if self.cfg.enable_curriculum:
            mean_terrain_level, max_terrain_level = self._apply_curriculum(env_ids)

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

            # If we are evaluating, we want all envs to have the same max episode length
            if self.cfg.eval_mode:
                self.episode_length_buf[:] = 0

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        # Reset robot state - redo the position reset to apply curriculum
        default_root_state = self._robot.data.default_root_state[env_ids].clone()

        reset_base_cfg = self.cfg.events.reset_base
        if reset_base_cfg is not None and "pose_range" in reset_base_cfg.params:
            range_list = [reset_base_cfg.params['pose_range'].get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        else:
            range_list = [(0.0, 0.0)] * 6
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        
        positions = default_root_state[:, 0:3] + self._terrain.env_origins[env_ids] + rand_samples[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = math_utils.quat_mul(default_root_state[:, 3:7], orientations_delta)
        self._robot.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids)

        if reset_base_cfg is None:
            self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        
        if self.cfg.events.reset_robot_joints is None:
            joint_pos = self._robot.data.default_joint_pos[env_ids]
            joint_vel = self._robot.data.default_joint_vel[env_ids]
            self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Sample new commands
        self.sample_new_commands(env_ids)

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        if self.cfg.enable_curriculum:
            extras["Metrics/Mean_terrain_level"] = mean_terrain_level
            extras["Metrics/Max_terrain_level"] = max_terrain_level
        extras["Metrics/Mean_distance"] = torch.mean(distances)
        extras["Metrics/Max_distance"] = torch.max(distances)

        self.extras["log"].update(extras)

    def sample_new_commands(self, env_ids: Sequence[int]):
        # Sample new commands
        rand = torch.zeros_like(self._commands[env_ids, 0])
        self._commands[env_ids, 0] = rand.uniform_(*self.cfg.commands.lin_vel_x_ranges)
        self._commands[env_ids, 1] = rand.uniform_(*self.cfg.commands.lin_vel_y_ranges)
        self._commands[env_ids, 2] = rand.uniform_(*self.cfg.commands.ang_vel_z_ranges)

        # Sample heading tracking and standing still environments
        self._sample_heading_envs(env_ids)
        self._sample_standing_envs(env_ids)

    def _sample_standing_envs(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)

        self.standing_still_envs[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.commands.standing_still_envs_prob

    def _sample_heading_envs(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)

        self.heading_tracking_envs[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.commands.heading_tracking_envs_prob
        self.heading_targets[env_ids] = math_utils.wrap_to_pi(r.uniform_(*self.cfg.commands.heading_target_ranges) + self._robot.data.heading_w[env_ids])

    def _update_commands(self):
        if self.cfg.commands.sample_heading_tracking_envs:
            env_ids = self.heading_tracking_envs.nonzero(as_tuple=False).flatten()
            heading_errors = math_utils.wrap_to_pi(self.heading_targets[env_ids] - self._robot.data.heading_w[env_ids])

            self._commands[env_ids, 2] = torch.clip(
                heading_errors * self.cfg.commands.heading_tracking_kp, *self.cfg.commands.ang_vel_z_ranges
            )

        if self.cfg.commands.sample_standing_still_envs:
            env_ids = self.standing_still_envs.nonzero(as_tuple=False).flatten()
            self._commands[env_ids, :] = 0.0

    def _apply_curriculum(self, env_ids: Sequence[int]) -> torch.Tensor:
        # compute the distance the robot walked
        distance = torch.norm(
            self._robot.data.root_pos_w[env_ids, :2] - self._terrain.env_origins[env_ids, :2], dim=1
        )

        # robots that walked far enough progress to harder terrains
        move_up = distance > self._terrain.cfg.terrain_generator.size[0] / 2

        # robots that walked less than half of their required distance go to simpler terrains
        move_down = distance < torch.norm(self._commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        move_down *= ~move_up

        # update terrain levels
        self._terrain.update_env_origins(env_ids, move_up, move_down)

        # return the mean terrain level
        return torch.mean(self._terrain.terrain_levels.float()), torch.max(self._terrain.terrain_levels.float())

    def _apply_observation_noises(self, obs: torch.Tensor):
        # apply all noises listed in config to given observations
        for key, noise in self.cfg.observation_noises.noises.items():
            # if matching key is not found or the observation is None
            if key not in obs or obs[key] is None:
                continue

            # apply noise to the observation
            obs[key] = noise.func(obs[key], noise)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            # Create markers if necessary for the first time
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)

            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # Check if robot is initialized
        if not self._robot.is_initialized:
            return

        # Get marker location
        base_pos_w = self._robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self._commands[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self._robot.data.root_lin_vel_b[:, :2])

        # Display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""

        # Arrow scale
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0

        # Arrow direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)

        # Convert everything back from base to world frame
        base_quat_w = self._robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat

    def _enforce_joint_limits(self, joint_targets: torch.Tensor) -> torch.Tensor:
        return torch.clip(
            joint_targets.clone(),
            self._robot.data.default_joint_pos_limits[..., 0],
            self._robot.data.default_joint_pos_limits[..., 1],
        )

    def _process_external_forces(self):
        if self.cfg.events.push_robot is None or "push_duration" not in self.cfg.events.push_robot.params:
            return

        self.timestep_since_last_push[:] += 1

        if "force_range" in self.cfg.events.push_robot.params:
            reset_env_ids = torch.nonzero((self.timestep_since_last_push > self.push_duration)).squeeze()
            if not torch.numel(reset_env_ids):
                return

            zeros = torch.zeros((len(reset_env_ids), self.num_pushed_bodies, 3), device=self.device)
            self._robot.set_external_force_and_torque(zeros, zeros, env_ids=reset_env_ids, body_ids=self.pushed_body_ids)
        elif "velocity_range" in self.cfg.events.push_robot.params:
            apply_env_ids = torch.nonzero((self.timestep_since_last_push <= self.push_duration)).squeeze()

            # transform to simulation frame
            lin_vel_w = math_utils.quat_rotate(self._robot.data.root_quat_w[apply_env_ids], self.push_vel_b[apply_env_ids])
            vel_w = torch.cat((lin_vel_w, self._robot.data.root_ang_vel_w[apply_env_ids]), dim=-1)

            # set the velocities into the physics simulation
            self._robot.write_root_velocity_to_sim(vel_w, env_ids=apply_env_ids)

    def _update_eval_logs(self):
        # Log time
        current_time = self.episode_length_buf * self.step_dt
        self.log_time = torch.cat((self.log_time, current_time.reshape(-1, 1, 1)), dim=2)

        self.log_base_linear_vel = torch.cat((self.log_base_linear_vel, self._robot.data.root_lin_vel_b.unsqueeze(2)), dim=2)
        self.log_base_angular_vel = torch.cat((self.log_base_angular_vel, self._robot.data.root_ang_vel_b.unsqueeze(2)), dim=2)

        self.log_base_position = torch.cat((self.log_base_position, self._robot.data.root_pos_w.unsqueeze(2)), dim=2)
        self.log_terrain_origins = torch.cat((self.log_terrain_origins, self._terrain.env_origins.unsqueeze(2)), dim=2)

        self.log_joint_acc = torch.cat((self.log_joint_acc, self._robot.data.joint_acc.unsqueeze(2)), dim=2)

        energy = torch.sum(torch.abs(self._robot.data.applied_torque * self._robot.data.joint_vel), dim=1)
        self.log_energy = torch.cat((self.log_energy, energy.reshape(-1, 1, 1)), dim=2)

        self.log_commands = torch.cat((self.log_commands, self._commands.unsqueeze(2)), dim=2)
        self.log_base_tracked_vel = torch.zeros_like(self.log_commands, dtype=torch.float, device=self.device)
        self.log_base_tracked_vel[:, :2, :] = self.log_base_linear_vel[:, :2, :]
        self.log_base_tracked_vel[:, 2, :] = self.log_base_angular_vel[:, 2, :]

        # These can be done after eval, but I'd rather put it here so we can calculate metrics any time
        self.log_survival_mask = torch.logical_not(self.log_died).float()

        # Compute survival time, account for envs that survived all the way
        death_indices = torch.argmax(self.log_died.int(), dim=2).squeeze(1)  # Survival time for envs that died, 0 otherwise
        no_death_envs = ~self.log_died.any(dim=2).squeeze(1)  # Mask for envs that did not die
        self.log_survival_time = torch.where(no_death_envs, torch.tensor(self.log_died.shape[2], dtype=torch.float, device=self.log_survival_time.device), death_indices.float())

    def save_eval_logs(self, dir):
        print(f"Survived: {torch.sum(torch.logical_not(self.log_died[:, :, -1].squeeze(1)))}/{self.num_envs} envs")

        if not os.path.exists(dir):
            os.makedirs(dir)
        
        np.save(os.path.join(dir, "time.npy"), self.log_time.cpu().numpy())

        np.save(os.path.join(dir, "base_linear_velocity.npy"), self.log_base_linear_vel.cpu().numpy())
        np.save(os.path.join(dir, "base_position.npy"), self.log_base_position.cpu().numpy())
        np.save(os.path.join(dir, "command.npy"), self.log_commands.cpu().numpy())
        np.save(os.path.join(dir, "base_tracked_velocity.npy"), self.log_base_tracked_vel.cpu().numpy())

        np.save(os.path.join(dir, "death_status.npy"), self.log_died.cpu().numpy())
        np.save(os.path.join(dir, "survival_mask.npy"), self.log_survival_mask.cpu().numpy())
        np.save(os.path.join(dir, "survival_time.npy"), self.log_survival_time.cpu().numpy())

        # Metrics
        metrics = (f"Success Rate: {self._compute_success_rate()}\n"
                   f"CoT: {self._compute_mean_cost_of_transport()}\n"
                   f"Joint Acc: {self._compute_mean_joint_accelerations()}\n"
                   f"Angular Vel: {self._compute_mean_angular_velocity()}\n"
                   f"Vel MAE: {self._compute_velocity_tracking_mae()}\n"
                   f"Mean Path Length: {torch.mean(self._compute_path_length())}\n"
                   f"Mean Survival Time: {torch.mean(self.log_survival_time)}\n")
        
        with open(os.path.join(dir, "metrics.txt"), "w") as f:
            f.write(metrics)

    def _compute_mean_cost_of_transport(self):
        mass = torch.sum(self._robot.data.default_mass[0, :], dim=0)
        gravity_acc = 9.80665

        energy_sums = torch.sum(self.log_energy * self.log_survival_mask, dim=2)

        command_mask = self.log_commands != 0.0
        tracked_velocities = self.log_base_tracked_vel * command_mask
        velocity_sums = torch.sum(torch.abs(tracked_velocities * self.log_survival_mask), dim=(1, 2))

        cost_of_transports = energy_sums / (mass * gravity_acc * velocity_sums)

        return torch.mean(cost_of_transports)

    def _compute_path_length(self):
        pos_differences = self.log_base_position[:, :2, 1:] - self.log_base_position[:, :2, :-1]
        adjacent_pos_length = torch.norm(pos_differences, dim=1)
        return torch.sum(adjacent_pos_length * self.log_survival_mask[:, 0, :-1], dim=1)

    def _compute_success_rate(self):
        survived_envs = torch.logical_not(self.log_died.any(dim=2).squeeze(1))

        return torch.sum(survived_envs) / self.num_envs

    def _compute_mean_joint_accelerations(self):
        # The episodic mean of the mean joint acceleration
        mean_joint_acc = torch.mean(torch.abs(self.log_joint_acc * self.log_survival_mask), dim=1)
        episodic_mean_joint_acc = torch.sum(mean_joint_acc, dim=1) / self.log_survival_time

        # Mean over all envs
        return torch.mean(episodic_mean_joint_acc)
    
    def _compute_mean_angular_velocity(self):
        # The episodic mean of the mean undesired base angular velocity
        mean_angular_vel = torch.mean(torch.abs(self.log_base_angular_vel[:, :2, :] * self.log_survival_mask), dim=1)
        episodic_mean_angular_vel = torch.sum(mean_angular_vel, dim=1) / self.log_survival_time

        # Mean over all envs
        return torch.mean(episodic_mean_angular_vel)

    def _compute_velocity_tracking_mae(self):
        # Compute the norm of the velocity tracking error, then take its episodic mean
        tracking_error = torch.norm((self.log_commands - self.log_base_tracked_vel) * self.log_survival_mask, dim=1)
        mean_tracking_error = torch.sum(tracking_error, dim=1) / self.log_survival_time

        # Mean over all envs
        return torch.mean(mean_tracking_error)
