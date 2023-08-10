import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from torch.autograd import Variable
import torch.nn.functional as F
from collections import namedtuple
import memory as mem
from feedforward import Feedforward

torch.set_num_threads(1)

class UnsupportedSpace(Exception):
    """Exception raised when the Sensor or Action space are not compatible

    Attributes:
        message -- explanation of the error
    """
    def __init__(self, message="Unsupported Space"):
        self.message = message
        super().__init__(self.message)

class QFunction(Feedforward):
    def __init__(self, observation_dim, action_dim, upper_bound, hidden_sizes,
                 learning_rate, output_size, activation_fun=torch.nn.LeakyReLU()):
        super().__init__(input_size=observation_dim + action_dim, hidden_sizes=hidden_sizes,
                         output_size=output_size, upper_bound=upper_bound, activation_fun=activation_fun)



    def Q_value(self, observations, actions, log=False):

        logits = self.forward(torch.hstack([observations,actions]), "critic")

        if log:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

class OUNoise():
    def __init__(self, shape, theta: float = 0.15, dt: float = 1e-2):
        self._shape = shape
        self._theta = theta
        self._dt = dt
        self.noise_prev = np.zeros(self._shape)
        self.reset()

    def __call__(self) -> np.ndarray:
        noise = (
            self.noise_prev
            + self._theta * ( - self.noise_prev) * self._dt
            + np.sqrt(self._dt) * np.random.normal(size=self._shape)
        )
        self.noise_prev = noise
        return noise

    def reset(self) -> None:
        self.noise_prev = np.zeros(self._shape)

class D4PGAgent(object):
    """
    Agent implementing Q-learning with NN function approximation.
    """
    def __init__(self, observation_space, action_space, **userconfig):

        if not isinstance(observation_space, spaces.box.Box):
            raise UnsupportedSpace('Observation space {} incompatible ' \
                                   'with {}. (Require: Box)'.format(observation_space, self))
        if not isinstance(action_space, spaces.box.Box):
            raise UnsupportedSpace('Action space {} incompatible with {}.' \
                                   ' (Require Box)'.format(action_space, self))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._config = {
            "tau": 0.005,  # for soft update of target parameters
            "eps": 0.1,    # strength of noise
            "eps_decay": 1,
            "eps_min": 0.01,
            "discount": 0.995,
            "rollout": 2,
            "buffer_size": int(1e6),
            "batch_size": 256,
            "lr_actor": 0.0001,
            "lr_critic": 0.0005,
            "weight_decay": 0,
            "gradient_clip": 0,
            "soft_updates_every": 1, # how often to copy weights over to target networks (Gradually)
            "hard_updates_every": 50, # how often to copy weights over to target networks (Instant)
            "hard_update": True, # Hard update OR Soft Update?
            "hidden_sizes_actor": [512, 256],
            "hidden_sizes_critic": [512, 256, 128],
            "use_target_net": True,
            "optimizer_eps": 0.000001,  # Optimizer epsilon: Term added to denominator for numerical stability
            "action_noise_theta": 0.15,  # Describes Ornstein-Uhlenbeck (OU) process noise
            "action_noise_sigma": 0.1, # Describes Ornstein-Uhlenbeck (OU) process noise
            "reward_norm": True,
            "batch_norm": True,
            "noise" : 0.2
        }
        self._observation_space = observation_space
        self._obs_dim = self._observation_space.shape[0]
        self._action_space = action_space
        #self._action_n = int(action_space.shape[0]/2)
        self._action_n = int(action_space.shape[0])
        self.action_noise = OUNoise((self._action_n))

        #self.buffer = mem.Memory(max_size=self._config["buffer_size"])
        self.buffer = mem.Memory(max_size=self._config["buffer_size"])
        self.replay_sample = namedtuple("ReplaySample", ["state", "action", "reward", "next_state", "done"])

        self.actor_loss = 0
        self.critic_loss = 0
        self.learn_step = 0
        self.episode = 0

        # D4PG variables
        # Lower and upper bounds of critic value output distribution, these will vary with environment
        self.vmin = -10.0  # Min value which atom's action-value function begins
        self.vmax = 10.0  # Max value which atom's action-value function begins
        self.num_atoms = 51  # Multiple atoms can represent the distribution of V
        self.delta = (self.vmax - self.vmin) / float((self.num_atoms - 1))
        self.atoms = torch.linspace(self.vmin, self.vmax, self.num_atoms)
        self.bin_centers = np.array([self.vmin + i * self.delta for i in range(self.num_atoms)]).reshape(-1, 1)

        self._config.update(userconfig)


        #Initialize ACTOR networks
        self.actor = Feedforward(input_size=self._obs_dim,
                                    hidden_sizes=self._config["hidden_sizes_actor"],
                                    output_size=self._action_n,
                                    upper_bound=None,
                                    activation_fun=torch.nn.LeakyReLU(),
                                    output_activation=torch.nn.Tanh(),
                                    batch_norm=self._config["batch_norm"]).to(self.device)

        self.actor_target = Feedforward(input_size=self._obs_dim,
                                    hidden_sizes=self._config["hidden_sizes_actor"],
                                    output_size=self._action_n,
                                    upper_bound=None,
                                    activation_fun=torch.nn.LeakyReLU(),
                                    output_activation=torch.nn.Tanh(),
                                    batch_norm=self._config["batch_norm"]).to(self.device)

        self.hard_update(self.actor, self.actor_target)

        #Initialize CRITIC networks
        self.critic = QFunction(observation_dim=self._obs_dim,
                            action_dim=self._action_n,
                            output_size=self.num_atoms,
                            upper_bound=None,
                            hidden_sizes=self._config["hidden_sizes_critic"],
                            learning_rate=self._config["lr_critic"]).to(self.device)

        # Target Q1 Network
        self.critic_target = QFunction(observation_dim=self._obs_dim,
                                   action_dim=self._action_n,
                                   output_size=self.num_atoms,
                                   upper_bound=None,
                                   hidden_sizes=self._config["hidden_sizes_critic"],
                                   learning_rate=0).to(self.device)

        self.hard_update(self.critic, self.critic_target)


        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                        lr=self._config["lr_actor"],
                                        eps=self._config["optimizer_eps"],
                                        weight_decay=self._config["weight_decay"])

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                        lr=self._config["lr_critic"],
                                        eps=self._config["optimizer_eps"],
                                        weight_decay=self._config["weight_decay"])
        self.train_iter = 0

        print("D4PG Agent initialized")
        print("eps", self._config["eps"])
        print("lr_actor", self._config["lr_actor"])
        print("lr_critic", self._config["lr_critic"])
        print("batch_size", self._config["batch_size"])
        print("batch_norm", self._config["batch_norm"])
        print("reward_norm", self._config["reward_norm"])
        print("hard_update", self._config["hard_update"])
        print("hard_updates_every", self._config["hard_updates_every"])
        print("noise", self._config["noise"])


    def act(self, observation, evaluation=False):

        observation = torch.from_numpy(observation).float().to(self.device)

        if evaluation is False:

            if np.random.random() > self._config["eps"]:

                self.actor.eval()
                with torch.no_grad():
                    action = self.actor.predict(observation).detach().cpu().data.numpy()
                self.actor.train()

                #action = self._action_space.low + (action + 1.0) / 2.0 * (self._action_space.high - self._action_space.low)
                #action += self.eps_ * self.action_noise()   # action in -1 to 1 (+ noise)
                action += self._gauss_noise(self._action_n)# if np.random.rand() < self._config["noise"] else 0
            else:
                #action = self._action_space.sample()[:4]
                action = self._action_space.sample()

        else:
            self.actor.eval()
            with torch.no_grad():
                action = self.actor.predict(observation).detach().cpu().data.numpy()

        return action.clip(-1, 1)

    @property
    def eps_(self):
        """
        This property ensures that the annealing process is run every time that
        E is called.
        Anneals the epsilon rate down to a specified minimum to ensure there is
        always some noisiness to the policy actions. Returns as a property.
        """

        self._config['eps'] = max(self._config["eps_min"], self._config['eps'] * self._config['eps_decay'])
        return self._config['eps']

    def store_transition(self, transition):
        self.buffer.add_transition(transition)

    def state(self):
        return (self.actor.state_dict(), self.actor_optimizer.state_dict(),
                self.critic.state_dict(), self.critic_optimizer.state_dict(), self._config)

    def restore_state(self, state):
        self.actor.load_state_dict(state[0])
        self.actor_optimizer.load_state_dict(state[1])
        self.critic.load_state_dict(state[2])
        self.critic_optimizer.load_state_dict(state[3])
        self.hard_update(self.actor, self.actor_target)
        self.hard_update(self.critic, self.critic_target)

    def reset(self):
        self.action_noise.reset()

    def train_iteration(self, num_iterations=32):
        # Convert a NumPy array to a Torch tensor with float32 data type
        to_torch = lambda x: torch.from_numpy(x.astype(np.float32))

        # Increment episode count and set epsilon
        self.episode += 1
        self._config["eps"] = self.eps_

        for i in range(num_iterations):
            # Sample data from the replay buffer
            #sampled_data, weights, tree_idx_lst = self.buffer.sample(batch_size=self._config['batch_size'])
            sampled_data = self.buffer.sample(batch_size=self._config['batch_size'])
            states = to_torch(np.stack(sampled_data[:, 0]))  # Current state
            actions = to_torch(np.stack(sampled_data[:, 1])[:, None]).squeeze(1)  # Chosen actions
            rewards = to_torch(np.stack(sampled_data[:, 2])[:, None])  # Rewards (batchsize,1)
            next_states = to_torch(np.stack(sampled_data[:, 3]))  # Next states
            dones = to_torch(np.stack(sampled_data[:, 4])[:, None])  # Done signals (batchsize,1)

            # Normalize rewards if reward normalization is enabled
            if self._config["reward_norm"]:
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            atoms = self.atoms.unsqueeze(0)

            # Calculate projected target probabilities using target networks
            actions_target = self.actor_target.predict(next_states)
            probs_target = self.critic_target.Q_value(next_states, actions_target)
            projected_target_probs = self.categorical_projection(rewards, probs_target, dones, states, actions)


            # Calculate log probability distribution using current critic network
            log_probs = self.critic.Q_value(states, actions, log=True)

            # Calculate critic loss (cross-entropy)
            critic_loss = -(projected_target_probs * log_probs).sum(-1).mean()
            #critic_loss = self.huber_loss(projected_target_probs, log_probs).sum(-1).mean()
            #print("critic_loss", critic_loss, critic_loss.shape)


            # Predict action for actor network loss calculation
            predicted_action = self.actor.forward(states, "actor")

            # Predict value distribution using current critic network and predicted action
            probs = self.critic.Q_value(states, predicted_action)

            # Calculate expected reward by summing probabilities and atom values
            expected_reward = (probs * atoms).sum(-1)

            # Calculate actor loss (policy gradient)
            actor_loss = -torch.mean(expected_reward)

            # Perform gradient ascent for actor
            self.actor.train()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self._config["gradient_clip"] != 0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self._config["gradient_clip"])
            self.actor_optimizer.step()

            # Perform gradient descent for critic
            self.critic.train()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self._config["gradient_clip"] != 0:
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self._config["gradient_clip"])
            self.critic_optimizer.step()

            # Update target networks (hard or soft)
            self.update_target_networks()

            # Store losses for visualization
            self.actor_loss = actor_loss.item()
            self.critic_loss = critic_loss.item()

            return self.critic_loss, self.actor_loss #, weights, tree_idx_lst

    def categorical_projection(self, rewards, probs_target, dones, states, actions):
        """
        Returns the projected value distribution for the input state/action pair
        """

        vmin = self.vmin
        vmax = self.vmax
        num_atoms = self.num_atoms
        delta_z = (vmax - vmin) / (num_atoms - 1)
        atoms = self.atoms
        gamma = self._config['discount']
        rollout = self._config["rollout"]  # N-Step bootstrapping for Temporal Difference Update Calculations

        # entropy_tau = 0.03
        # alpha = 0.9
        # lo = -1
        # entropy_coeff = 0.001
        # N = 32
        #
        # print("Probs Target shape: ", probs_target.shape)
        # q_t_n = probs_target
        # print("Q_t_n shape: ", q_t_n.shape)
        # logsum = torch.logsumexp(q_t_n / entropy_tau, dim=1).unsqueeze(-1)
        # print("Logsum shape: ", logsum.shape)
        # tau_log_pi_next = (q_t_n - entropy_tau * logsum)
        # print("Tau Log Pi Next shape: ", tau_log_pi_next.shape)
        # pi_target = F.softmax(tau_log_pi_next, dim=1)
        # print("Pi Target shape: ", pi_target.shape)
        # Q_target = (gamma**rollout * (pi_target * (probs_target-tau_log_pi_next)*(1 - dones)))
        # print("Q Target shape: ", Q_target.shape)
        # q_k_target = self.critic_target.Q_value(states, actions)
        # print("Q_k_target shape: ", q_k_target.shape)
        # tau_log_pik = q_k_target - entropy_tau * torch.logsumexp(q_k_target/entropy_tau, 0)
        # print("Tau Log Pik shape: ", tau_log_pik.shape)
        # munchausen_reward = (rewards + alpha*torch.clamp(tau_log_pik, min=lo, max=0))
        # print("Munchausen Reward shape: ", munchausen_reward.shape)
        # Q_targets = munchausen_reward + Q_target

        # Rewards were stored with 0->(N-1) summed, take Reward and add it to
        # the discounted expected reward at N (ROLLOUT) timesteps
        projected_atoms = rewards + gamma ** rollout * atoms.unsqueeze(0) + (1 - dones)
        projected_atoms.clamp_(vmin, vmax)

        b = (projected_atoms - vmin) / delta_z

        lower_bound = b.floor()
        upper_bound = b.ceil()

        # Don't think (lower_bound == upper_bound) matters here
        m_lower = (upper_bound + (lower_bound == upper_bound).float() - b) * probs_target
        m_upper = (b - lower_bound.float()) * probs_target

        projected_probs = torch.tensor(np.zeros(probs_target.size())).to(self.device)

        for idx in range(probs_target.size(0)):
            projected_probs[idx].index_add_(0, lower_bound[idx].long(), m_lower[idx].double())
            projected_probs[idx].index_add_(0, upper_bound[idx].long(), m_upper[idx].double())

        return projected_probs.float()

    def train(self, num_iterations=32):

        self.episode += 1
        self._config["eps"] = self.eps_
        #print("Episode: ", self.episode, " Epsilon: ", self._config["eps"])

        # Perform the training iterations
        for i in range(num_iterations):
            # Sample a batch from the replay buffer
            samples = self.sample_from_buffer(batch_size=self._config['batch_size'])

            # Update the critic network and store the critic loss
            critic_loss = self.update_critic(samples)
            # Update the actor network and store the actor loss
            actor_loss = self.update_actor(samples)

            # Update the target networks (hard or soft using Polyak averaging)
            self.update_target_networks()

        return critic_loss, actor_loss

    def update_critic(self, samples):

        states = self.to_tensor(np.stack([sample.state for sample in samples]))
        actions = self.to_tensor(np.stack([sample.action for sample in samples]))
        rewards = self.to_tensor(np.stack([sample.reward for sample in samples])[:, None])
        next_states = self.to_tensor(np.stack([sample.next_state for sample in samples]))
        dones = self.to_tensor(np.stack([sample.done for sample in samples])[:, None])

        if self._config["reward_norm"] is True:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Calculate log probability DISTRIBUTION using Zw w.r.t. stored actions
        q_dist_log = self.critic.Q_value(states, actions, log=True)
        q_dist = self.critic.Q_value(states, actions, log=True)

        target_z_dist = self.critic_target.Q_value(next_states, self.actor_target(next_states, "actor"), log=False)

        projected_target_probs = self.categorical_l2_projection(target_z_dist.cpu().data.numpy(),
                                                                rewards, dones)
        critic_loss = -(self.to_tensor(projected_target_probs, requires_grad=False) * q_dist).sum(
            dim=1).mean()

        self.critic.train()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        if self._config["gradient_clip"] != 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self._config["gradient_clip"])

        self.copy_gradients(self.actor, self.actor_target)
        self.critic_optimizer.step()

        return critic_loss.item()

    def update_actor(self, samples):

        states = self.to_tensor(np.stack([sample.state for sample in samples]))

        actor_loss = self.critic.Q_value(states, self.actor(states, "actor"), log=False)
        actor_loss = -actor_loss.matmul(self.to_tensor(self.bin_centers)).mean()

        self.actor.train()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self._config["gradient_clip"] != 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self._config["gradient_clip"])
        self.copy_gradients(self.critic, self.critic_target)
        self.actor_optimizer.step()

        return actor_loss.item()

    def update_target_networks(self):

        if self._config["hard_update"]:
            self.learn_step = (self.learn_step + 1) % self._config["hard_updates_every"]
        else:
            self.learn_step = (self.learn_step + 1) % self._config["soft_updates_every"]

        if self._config["hard_update"] and self.learn_step == 0:
            self.hard_update(self.critic, self.critic_target)
            self.hard_update(self.actor, self.actor_target)
        elif not self._config["hard_update"] and self.learn_step == 0:
            self.soft_update(self.critic, self.critic_target, self._config["tau"])
            self.soft_update(self.actor, self.actor_target, self._config["tau"])

    def categorical_l2_projection(self, target_z_dist, rewards, dones):

        # Reshape rewards and dones for easier processing
        rewards = rewards.detach().numpy().reshape(-1)
        dones = dones.detach().numpy().reshape(-1).astype(bool)

        # Initialize the projected distribution
        proj_distr = np.zeros((self._config["batch_size"], self.num_atoms), dtype=np.float32)

        # Calculate the Categorical L2 projection for each atom
        for atom in range(self.num_atoms):
            # Calculate the projected target values for the current atom
            tz_j = np.minimum(self.vmax, np.maximum(self.vmin, rewards + (self.vmin + atom * self.delta) * self._config[
                "discount"]))

            # Calculate the indices for lower and upper bounding of the projection
            b_j = (tz_j - self.vmin) / self.delta
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)

            # Handle cases where lower and upper bounds are equal
            eq_mask = (u == l)
            proj_distr[eq_mask, l[eq_mask]] += target_z_dist[eq_mask, atom]

            # Handle cases where lower and upper bounds are not equal
            ne_mask = (u != l)
            proj_distr[ne_mask, l[ne_mask]] += target_z_dist[ne_mask, atom] * (u - b_j)[ne_mask]
            proj_distr[ne_mask, u[ne_mask]] += target_z_dist[ne_mask, atom] * (b_j - l)[ne_mask]

        # Handle terminal states
        if dones.any():
            # Set the projected distribution to zero for terminal states
            proj_distr[dones] = 0.0

            # Calculate the target values for terminal states
            tz_j = np.minimum(self.vmax, np.maximum(self.vmin, rewards[dones]))

            # Calculate the indices for lower and upper bounding of the projection for terminal states
            b_j = (tz_j - self.vmin) / self.delta
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)

            # Handle cases where lower and upper bounds are equal for terminal states
            eq_mask = (u == l)
            eq_dones = dones.copy()
            eq_dones[dones] = eq_mask
            if eq_dones.any():
                #print("proj_distr[eq_dones, l]", proj_distr[eq_dones, l[eq_mask]].shape)
                proj_distr[eq_dones, l[eq_mask]] = 1.0

            # Handle cases where lower and upper bounds are not equal for terminal states
            ne_mask = (u != l)
            ne_dones = dones.copy()
            ne_dones[dones] = ne_mask
            if ne_dones.any():
                #print("ne_dones", ne_dones.shape)
                #print("b_j", b_j.shape, b_j)
                #print("l", l.shape, l)
                #print("ne_mask", ne_mask.shape, ne_mask)
                #print("l[ne_mask]", l[ne_mask].shape, l[ne_mask])
                #print("u", u.shape, u)
                #print("(u - b_j)[ne_mask]", (u - b_j)[ne_mask].shape, (u - b_j)[ne_mask])
                #print("(b_j - l)[ne_mask]", (b_j - l)[ne_mask].shape, (b_j - l)[ne_mask])
                #print("proj_distr", proj_distr.shape)
                #print("proj_distr[ne_dones, l]", proj_distr[ne_dones, l[ne_mask]].shape, proj_distr[ne_dones, l[ne_mask]])

                proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
                proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]

        return proj_distr

    def _gauss_noise(self, shape):
        """
        Returns the epsilon scaled noise distribution for adding to Actor
        calculated action policy.
        """

        n = np.random.normal(0, self._config["noise"], shape)
        return n


    def hard_update(self, local_model, target_model):
        """
        Fully copy parameters from active network to target network. To be used
        in conjunction with a parameter "C" that modulated how many timesteps
        between these hard updates.
        """
        target_model.load_state_dict(local_model.state_dict())


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def to_tensor(self, x, requires_grad=True, dtype=torch.FloatTensor):
        x = torch.from_numpy(x.astype(np.float32))
        x = Variable(x, requires_grad=requires_grad).type(dtype)
        return x

    def copy_gradients(self, model_local, model_global ):
        for param_local, param_global in zip(model_local.parameters(), model_global.parameters()):
            if param_global.grad is not None:
                return
            param_global._grad = param_local.grad

    def sample_from_buffer(self, batch_size):
        return [self.replay_sample(*sample) for sample in self.buffer.sample(batch_size=batch_size)]


    # Huber loss function
    def huber_loss(self, y_pred, y, delta=1.0):
        huber_mse = 0.5 * (y - y_pred) ** 2
        huber_mae = delta * (torch.abs(y - y_pred) - 0.5 * delta)
        return torch.where(torch.abs(y - y_pred) <= delta, huber_mse, huber_mae)