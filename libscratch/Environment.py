import math
import os
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from libscratch.Elegant import ElegantWrapper
from libscratch.Utils import change_num_initial_particles


class ACCElegantEnvironment(gym.Env):
    def __init__(
        self,
        stage=None,
        n_bins=5,
        init_num_particles=None,
        results_path=" ",
        input_beamline_file="machine.lte",
        input_beam_file="track.ele",
        beamline_name="machine",
        output_beamline_file="updated_machine.lte",
        reset_specific_keys_bool=True,
        logger=None,
        file_handler=None,
        elegant_path="/Users/anwar/Downloads/sdds/darwin-x86/",
        sddsPath="/Users/anwar/Downloads/sdds/defns.rpn",
        override_dynamic_command=False,
        overridden_command=" ",
    ):
        super().__init__()

        self.override_dynamic_command = override_dynamic_command
        self.overridden_command = overridden_command
        self.input_file_path = input_beamline_file
        self.input_beam_file = input_beam_file
        self.output_file = output_beamline_file
        self.beamline_name = beamline_name
        self.elegantPath = elegant_path
        self.sddsPath = sddsPath
        self.results_path = results_path
        self.n_bins = n_bins
        self.initial_reward = 1.0
        self.init_num_particles = init_num_particles

        self.wrapper = ElegantWrapper(
            self.input_file_path,
            self.input_beam_file,
            self.beamline_name,
            self.output_file,
            elegant_path=self.elegantPath,
            sddsPath=self.sddsPath,
            results_path=self.results_path,
            overrid_dynmaic_commnad=self.override_dynamic_command,
            overrideen_command=self.overridden_command,
        )

        self.reset_specific_keys_bool = reset_specific_keys_bool
        self.wrapper.reset_specific_keys_bool = self.reset_specific_keys_bool
        self.max_num_of_vars = len(self.wrapper.chroneological_order_controllable_vars)
        self.max_num_of_states = self.wrapper.max_itteration

        if stage is not None and stage < self.max_num_of_vars:
            self.stage = stage
        else:
            self.stage = None

        self.stage_mask = None
        self.observation = None
        self.variables = self.wrapper.chroneological_variables

        self._set_action_space(self.variables)
        self.reset()

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_shape,),
            dtype=np.float32,
        )

        self.logger = logger
        self.file_handler = file_handler

        if self.stage is not None:
            self.stage_mask = self._get_stage_mask()

        self.iteration = 0

    def _set_action_space(self, variables):
        """Set the action spaces for the environment."""
        action_low = []
        action_high = []

        for var in variables:
            if "K1" in var:
                action_low.append(-20.0)
                action_high.append(20.0)
            elif any(x in var for x in ["VKICK", "HKICK", "FSE"]):
                action_low.append(-0.005)
                action_high.append(0.005)
            else:
                action_low.append(-1.0)
                action_high.append(1.0)

        action_low = np.array(action_low, dtype=np.float32)
        action_high = np.array(action_high, dtype=np.float32)

        self.action_space_fake = spaces.Box(
            low=action_low,
            high=action_high,
            dtype=np.float32,
        )
        self.action_space_fake.n = self.action_space_fake.low.size

        real_low = np.array([-20.0, -0.005, -0.005, -0.005], dtype=np.float32)
        real_high = np.array([20.0, 0.005, 0.005, 0.005], dtype=np.float32)

        self.action_space = spaces.Box(
            low=real_low,
            high=real_high,
            dtype=np.float32,
        )
        self.action_space.n = self.action_space.low.size

    def _convert_variables(self, values):
        """Convert values from [-1, 1] into the real action range."""
        values = np.asarray(values, dtype=np.float32)
        converted_variables = []

        for i, value in enumerate(values):
            range_min = self.action_space.low[i]
            range_max = self.action_space.high[i]
            converted_variable = value * (range_max - range_min) / 2 + (range_max + range_min) / 2
            converted_variables.append(converted_variable)

        return np.round(np.asarray(converted_variables, dtype=np.float32), 4)

    def _get_stage_mask(self):
        """Generate a mask based on the current curriculum-learning stage."""
        stage_mask = np.zeros(len(self.variables), dtype=np.float32)
        count = 0

        for i in range(self.stage):
            self.iteration = i
            count += self._check_number_of_variables_to_be_set_at_this_iteration()
            stage_mask[:count] = 1.0

        return stage_mask

    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        super().reset(seed=seed)

        if self.init_num_particles is None:
            self.init_num_particles = random.randint(500, 50000)

        _, self.initial_number_of_particles = change_num_initial_particles(
            self.input_beam_file + ".ele",
            self.init_num_particles,
        )

        self._set_initial_number_of_particles(self.initial_number_of_particles)
        self.done = False
        self.reward = 0.0

        self.wrapper = ElegantWrapper(
            self.input_file_path,
            self.input_beam_file,
            self.beamline_name,
            self.output_file,
            elegant_path=self.elegantPath,
            sddsPath=self.sddsPath,
            results_path=self.results_path,
            overrid_dynmaic_commnad=self.override_dynamic_command,
            overrideen_command=self.overridden_command,
        )
        self.wrapper.reset_specific_keys_bool = self.reset_specific_keys_bool

        self.iteration = 0
        self.previous_mask_len = 0
        self.mask_len = 0
        self.actions_ = np.zeros((len(self.variables),), dtype=np.float32)

        values = np.zeros((len(self.variables),), dtype=np.float32)
        _, success, _ = self.wrapper.run_elegant_simulation(values)
        if not success:
            raise RuntimeError("Elegant simulation failed during env.reset().")

        observations, reward, output_file, done = self.wrapper.get_results(
            self.initial_number_of_particles
        )
        if observations is None:
            raise RuntimeError(
                "Elegant produced no initial observation during env.reset()."
            )

        if reward != 0:
            self.initial_reward = float(reward)
            self.number_of_particle_prev = self.initial_reward
        else:
            self.initial_reward = 1.0
            self.number_of_particle_prev = 1.0

        self.observation_shape = 16 + self.n_bins**2 + 1 + 1 + 10 + 4
        self.observation = np.asarray(observations, dtype=np.float32)
        return self.observation, {}

    def _set_initial_number_of_particles(self, new):
        self.initial_number_of_particles = new

    def _check_number_of_variables_to_be_set_at_this_iteration(self):
        """Check how many variables should be controlled at the current iteration."""
        count = 0
        if self.iteration < len(self.wrapper.chroneological_order_controllable_vars):
            current_var = self.wrapper.chroneological_order_controllable_vars[self.iteration]
            for var in self.variables:
                base_name = (
                    var.replace("K1", "")
                    .replace("VKICK", "")
                    .replace("HKICK", "")
                    .replace("FSE", "")
                )
                if current_var == base_name:
                    count += 1
        return count

    def _get_mask(self, mask_len):
        """Generate a cumulative mask up to mask_len."""
        mask = np.zeros(len(self.variables), dtype=np.float32)
        mask[:mask_len] = 1.0
        return mask

    def _get_action_mask(self, count):
        mask = np.zeros(self.action_space.n, dtype=np.float32)
        if count == 3:
            mask[:count] = 1.0
        elif count == 1:
            mask[-count] = 1.0
        return mask

    def _get_new_action(self, count, action):
        action = np.asarray(action, dtype=np.float32).flatten()

        if count == 3:
            return action[:count]
        if count == 1:
            return np.array([action[-1]], dtype=np.float32)
        if count == 0:
            return np.array([], dtype=np.float32)

        raise ValueError(f"Unexpected number of variables for this iteration: {count}")

    def _correct_action(self, action):
        """
        Update only the variables for the current iteration while keeping previous ones.
        """
        action_mask = np.zeros(len(self.variables), dtype=np.float32)
        action_mask[self.previous_mask_len:self.mask_len] = action

        previous_actions_mask = np.zeros(len(self.variables), dtype=np.float32)
        previous_actions_mask[:self.previous_mask_len] = 1.0

        previous_actions = (previous_actions_mask * self.actions_) + action_mask
        return previous_actions.astype(np.float32)

    def _log_episode(self, info):
        """Log episode information if a logger exists."""
        if self.logger is None or self.file_handler is None:
            return

        ep_info = {
            "reward": info["reward"],
            "done": info["done"],
            "itteration": info["itteration"],
            "current_element": info["output_file"],
            "dict_vars": info["dict_vars"],
            "initial_number_of_particles": info["initial_number_of_particles"],
            "number_of_particles": info["number_of_particles"],
        }
        self.logger.writerow(ep_info)
        self.file_handler.flush()

    def step(self, action, convert=False):
        """
        Perform one environment step.
        """
        action = np.asarray(action, dtype=np.float32).flatten()

        if convert:
            action = self._convert_variables(action)

        num_of_vars_iteration = self._check_number_of_variables_to_be_set_at_this_iteration()
        self.mask_len = self.previous_mask_len + num_of_vars_iteration
        self.action_mask = self._get_action_mask(num_of_vars_iteration)
        new_actions = self._get_new_action(num_of_vars_iteration, action)
        self.mask = self._get_mask(self.mask_len)

        self.actions_ = self._correct_action(new_actions)

        if self.stage_mask is not None:
            self.actions_ = (self.actions_ * self.stage_mask).astype(np.float32)

        self.previous_mask_len = self.mask_len
        elegant_input, success, dict_vars = self.wrapper.run_elegant_simulation(self.actions_)
        if not success:
            raise RuntimeError(
                f"Elegant simulation failed during env.step() at iteration {self.iteration}."
            )

        if not os.path.exists("inputs"):
            os.makedirs("inputs")

        try:
            with open(f"inputs/elegant_input{self.iteration}.txt", "w") as file:
                file.write(elegant_input)
        except TypeError as exc:
            print(f"Warning: {exc}")

        observations_dataframe, reward, output_file, done = self.wrapper.get_results(
            self.initial_number_of_particles
        )

        self.number_of_particle_curr = reward
        reward = self.number_of_particle_curr / max(self.initial_reward, 1.0)

        self.done = done
        if self.number_of_particle_curr <= 3:
            self.done = True
            reward -= math.sqrt(
                abs((self.max_num_of_vars**2) - ((self.iteration + 1) ** 2))
            ) * (1 / self.max_num_of_vars)
        elif self.number_of_particle_curr > 3:
            reward *= self.number_of_particle_curr / max(self.number_of_particle_prev, 1.0)
            self.number_of_particle_prev = self.number_of_particle_curr
            if output_file == "final_WP":
                self.done = True

        self.reward = float(reward)

        if observations_dataframe is not None:
            self.observation = np.asarray(observations_dataframe, dtype=np.float32)
        elif self.observation is None:
            with open("error_log.txt", "a") as error_file:
                error_file.write(
                    f"Iteration: {self.iteration}, Output File: {output_file}\n"
                )
            self.observation = np.zeros(self.observation_shape, dtype=np.float32)

        if self.iteration < self.max_num_of_vars:
            self.iteration += 1
        else:
            print("itteration is over the max length of the beamline")
            self.done = True
            print("we set done to True")

        if self.stage == self.iteration:
            self.done = True

        info = {
            "dict_vars": dict_vars,
            "actions": self.actions_,
            "itteration": self.iteration,
            "reward": self.reward,
            "output_file": output_file,
            "done": self.done,
            "masked_action": self.actions_,
            "observations_dataframe": observations_dataframe,
            "initial_number_of_particles": self.initial_number_of_particles,
            "number_of_particles": self.wrapper.get_num_particles(),
        }

        if self.done:
            self._log_episode(info)

        return self.observation, float(self.reward), self.done, False, info

    def get_number_of_particles(self):
        return self.wrapper.num_particles

    def stage_learning(self):
        """Placeholder for curriculum-learning logic."""
        pass