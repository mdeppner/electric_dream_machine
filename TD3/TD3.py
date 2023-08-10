import torch
import copy
import optparse
import pickle
import numpy as np
import gymnasium as gym
import memory as mem
import laserhockey.hockey_env as h_env
import torch.nn as nn
import torch.nn.functional as F


from gymnasium import spaces
from importlib import reload
from feedforward import Feedforward
from torch.distributions import MultivariateNormal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)


class UnsupportedSpace(Exception):
    """Exception raised when the Sensor or Action space are not compatible

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="Unsupported Space"):
        self.message = message
        super().__init__(self.message)


class Actor(Feedforward):

    def __init__(self, observation_dim, action_dim, max_action, activation_fun=torch.nn.ReLU(),
                                  output_activation=torch.nn.Tanh(), hidden_sizes=[400, 300],  learning_rate=0.0002):
        super().__init__(input_size=observation_dim, hidden_sizes=hidden_sizes, output_size=action_dim,
                         activation_fun=activation_fun, output_activation=output_activation)

        self.max_action = max_action
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, eps=0.000001)

    def forward(self, x):
        for layer, activation_fun in zip(self.layers, self.activations):
            x = activation_fun(layer(x))
        if self.output_activation is not None:
            return self.output_activation(self.readout(x)) * self.max_action
        else:
            return self.readout(x)
    def predict(self, x):
        with torch.no_grad():
            return self.forward(torch.from_numpy(x.astype(np.float32))).numpy() * self.max_action

class QFunction(Feedforward):
    def __init__(self, observation_dim, action_dim, hidden_sizes=[400, 300],
                 learning_rate=0.0002):
        super().__init__(input_size=observation_dim + action_dim, hidden_sizes=hidden_sizes,
                         output_size=1, activation_fun=torch.nn.ReLU(), output_activation=None)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=learning_rate,
                                          eps=0.000001)
        self.loss = torch.nn.MSELoss()

    def fit(self, observations, actions, targets):  # all arguments should be torch tensors
        self.train()  # put model in training mode
        # Forward pass

        pred = self.Q_value(observations, actions)
        # Compute Loss
        loss = self.loss(pred, targets).mean()

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def Q_value(self, observations, actions):
        return self.forward(torch.hstack([observations, actions]))


class TD3Agent(object):
    """
    Agent implementing Q-learning with NN function approximation.
    """

    def __init__(self, observation_space, action_space, env_name, **userconfig):

        if not isinstance(observation_space, spaces.box.Box):
            raise UnsupportedSpace('Observation space {} incompatible ' \
                                   'with {}. (Require: Box)'.format(observation_space, self))
        if not isinstance(action_space, spaces.box.Box):
            raise UnsupportedSpace('Action space {} incompatible with {}.' \
                                   ' (Require Box)'.format(action_space, self))

        self._env_name = env_name
        self._observation_space = observation_space
        self._obs_dim = self._observation_space.shape[0]
        self._action_space = action_space
        self.max_action = action_space.high[0]
        self._config = {
            "eps": 0.1,  # Epsilon: noise strength to add to policy exploration noise
            "discount": 0.99,
            "buffer_size": int(1e6),
            "batch_size": 256,
            "learning_rate_actor": 0.0001,
            "learning_rate_critic": 0.0001,
            "hidden_sizes_actor": [400, 300],
            "hidden_sizes_critic1": [400, 300],
            "hidden_sizes_critic2": [400, 300],
            "policy_noise_std": 0.2,
            "policy_noise_clip": 0.5,
            "update_target_every": 2,
            "use_target_net": True,
            "tau": 0.005,
            "full_td3": True,
            "clipped_dqn": False,
            "delayed_updates": False,
            "policy_smoothing": False,
            "train_shooting": False,
            "train_defense": False,
        }
        self._config.update(userconfig)
        self._eps = self._config['eps']
        self._tau = self._config['tau']
        self._full_td3 = self._config['full_td3']
        self._clip_dqn = self._config['clipped_dqn']
        self._delayed_updates = self._config['delayed_updates']
        self._policy_smoothing = self._config['policy_smoothing']
        self._train_shooting = self._config['train_shooting']
        self._train_defense = self._config['train_defense']
        self._action_n = int(action_space.shape[0]/2)
        self.action_dim = self._action_n

        self.buffer = mem.Memory(max_size=self._config["buffer_size"])

        # Q1 Network - Critic Network
        self.Q1 = QFunction(observation_dim=self._obs_dim,
                            action_dim=self._action_n,
                            hidden_sizes=self._config["hidden_sizes_critic1"],
                            learning_rate=self._config["learning_rate_critic"])

        # Target Q1 Network
        self.Q1_target = QFunction(observation_dim=self._obs_dim,
                                   action_dim=self._action_n,
                                   hidden_sizes=self._config["hidden_sizes_critic1"],
                                   learning_rate=0.0001)

        # Q2 Network aka Critic Network
        self.Q2 = QFunction(observation_dim=self._obs_dim,
                            action_dim=self._action_n,
                            hidden_sizes=self._config["hidden_sizes_critic2"],
                            learning_rate=self._config["learning_rate_critic"])

        # Target Q2 Network
        self.Q2_target = QFunction(observation_dim=self._obs_dim,
                                   action_dim=self._action_n,
                                   hidden_sizes=self._config["hidden_sizes_critic2"],
                                   learning_rate=0.0001)

        # Policy Network - Actor Network
        self.policy = Actor(observation_dim=self._obs_dim,
                            hidden_sizes=self._config["hidden_sizes_actor"],
                            action_dim=self._action_n,
                            max_action=self.max_action,
                            activation_fun=torch.nn.ReLU(),
                            output_activation=torch.nn.Tanh())

        # Policy Target Network - Actor Target Network
        self.policy_target = Actor(observation_dim=self._obs_dim,
                                   hidden_sizes=self._config["hidden_sizes_actor"],
                                   action_dim=self._action_n,
                                   max_action=self.max_action,
                                   activation_fun=torch.nn.ReLU(),
                                   output_activation=torch.nn.Tanh())

        self._copy_nets()

        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=self._config["learning_rate_actor"],
                                          eps=0.000001)
        self.train_iter = 0

    def _copy_nets(self):
        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q2_target.load_state_dict(self.Q2.state_dict())
        self.policy_target.load_state_dict(self.policy.state_dict())

    def act(self, observation, eps=None):
        if eps is None:
            eps = self._eps
        action_dim = int(self._action_n)

        action = self.policy.predict(observation)
        action = self._action_space.low[:action_dim] + (action + 1.0) / 2.0 * (self._action_space.high[:action_dim] - self._action_space.low[:action_dim])

        noise = np.random.normal(0.0, eps, size=action_dim)
        noisy_action = action + noise
        noisy_action = noisy_action.clip(min=self._action_space.low[:action_dim], max=self._action_space.high[:action_dim])

        return noisy_action

    def store_transition(self, transition):
        self.buffer.add_transition(transition)

    def state(self):
        return (self.Q1.state_dict(), self.Q1.optimizer.state_dict(),
                self.Q2.state_dict(), self.Q2.optimizer.state_dict(),
                self.policy.state_dict(), self.policy.optimizer.state_dict())

    def restore_state(self, state):
        self.Q1.load_state_dict(state[0])
        self.Q1.optimizer.load_state_dict(state[1])
        self.Q2.load_state_dict(state[2])
        self.Q2.optimizer.load_state_dict(state[3])
        self.policy.load_state_dict(state[4])
        self.policy.optimizer.load_state_dict(state[5])
        self._copy_nets()


    def target_policy_smoothing(self, policy_action):
        # functionality to add noise to the policy action according to TD3 paper
        noise = torch.zeros_like(policy_action).normal_(0, self._config["policy_noise_std"]).to(device)
        noise = torch.clamp(noise, min=-self._config["policy_noise_clip"], max=self._config["policy_noise_clip"])
        noisy_policy_action = torch.clamp(policy_action + noise, min=self._action_space.low[0], max=self._action_space.high[0])
        return noisy_policy_action

    def train(self, iter_fit=32):

        to_torch = lambda x: torch.from_numpy(x.astype(np.float32))
        losses_Q1 = []
        losses_Q2 = []
        actor_losses = []
        self.train_iter += 1
        for i in range(iter_fit):

            # sample from the replay buffer
            data = self.buffer.sample(batch=self._config['batch_size'])
            s = to_torch(np.stack(data[:, 0]))  # s_t
            a = to_torch(np.stack(data[:, 1]))  # a_t
            rew = to_torch(np.stack(data[:, 2])[:, None])  # rew  (batchsize,1)
            s_prime = to_torch(np.stack(data[:, 3]))  # s_t+1
            done = to_torch(np.stack(data[:, 4])[:, None])  # done signal  (batchsize,1)

            policy_action = self.policy_target.forward(s_prime)

            # Enable or disable Policy Smoothing
            if self._full_td3 or self._policy_smoothing:
                policy_action = self.target_policy_smoothing(policy_action)

            if self._full_td3 or self._clip_dqn:
                if self._config["use_target_net"]:
                    # from target actor network to each target critic network
                    # the target critic network gets as input prediction pi'(s') from policy/actor target network
                    q1_target = self.Q1_target.Q_value(s_prime, policy_action)
                    q2_target = self.Q2_target.Q_value(s_prime, policy_action)
                    q_prime = torch.min(q1_target, q2_target).detach()

                else:
                    q1 = self.Q1.Q_value(s_prime, self.policy.forward(s_prime))
                    q2 = self.Q2.Q_value(s_prime, self.policy.forward(s_prime))
                    q_prime = torch.min(q1, q2).detach()

                # target
                gamma = self._config['discount']
                td_target = rew + gamma * (1.0 - done) * q_prime

                # optimize the Q objective
                fit_loss_Q1 = self.Q1.fit(s, a, td_target)
                fit_loss_Q2 = self.Q2.fit(s, a, td_target)

                losses_Q1.append(fit_loss_Q1)
                losses_Q2.append(fit_loss_Q2)

            else:
                if self._config["use_target_net"]:
                    # from target actor network to each target critic network
                    # the target critic network gets as input prediction pi'(s') from policy/actor target network
                    q_prime = self.Q1_target.Q_value(s_prime, policy_action)
                else:
                    q_prime = self.Q1.Q_value(s_prime, policy_action)

                # target
                gamma = self._config['discount']
                td_target = rew + gamma * (1.0 - done) * q_prime

                # optimize the Q objective
                fit_loss_Q1 = self.Q1.fit(s, a, td_target)
                losses_Q1.append(fit_loss_Q1)

            pred_current_action = self.policy.forward(s)
            pred_current_q1 = self.Q1.Q_value(s, pred_current_action)
            actor_loss = -torch.mean(pred_current_q1)

            actor_losses.append(actor_loss.item())

            if (self._full_td3 or self._delayed_updates) and self._config["use_target_net"]:
                if i % self._config["update_target_every"] == 0:

                    self.policy.optimizer.zero_grad()
                    actor_loss.backward()
                    self.policy.optimizer.step()

                    self.soft_update_targets()
            else:
                self._copy_nets()
                # optimize actor objective
                self.policy.optimizer.zero_grad()
                actor_loss.backward()
                self.optimizer.step()

        return losses_Q1, losses_Q2, actor_losses

    # Function to perform Polyak averaging to update the parameters of the provided network
    def soft_update_net(self, source_net_params, target_net_params):

        for source_param, target_param in zip(source_net_params, target_net_params):
            target_param.data.copy_(
                self._tau * source_param.data + (1 - self._tau) * target_param.data)

    # Polyak averaging over all three target networks
    def soft_update_targets(self):

        self.soft_update_net(self.policy.parameters(), self.policy_target.parameters())
        self.soft_update_net(self.Q1.parameters(), self.Q1_target.parameters())
        self.soft_update_net(self.Q2.parameters(), self.Q2_target.parameters())


def main():
    optParser = optparse.OptionParser()
    optParser.add_option('-e', '--env', action='store', type='string',
                         dest='env_name', default="Pendulum-v1",
                         help='Environment (default %default)')
    optParser.add_option('-n', '--eps', action='store', type='float',
                         dest='eps', default=0.1,
                         help='Policy noise (default %default)')
    optParser.add_option('-t', '--train', action='store', type='int',
                         dest='train', default=32,
                         help='number of training batches per episode (default %default)')
    optParser.add_option('-l', '--lr', action='store', type='float',
                         dest='lr', default=0.0001,
                         help='learning rate for actor/policy (default %default)')
    optParser.add_option('-m', '--maxepisodes', action='store', type='float',
                         dest='max_episodes', default=2000,
                         help='number of episodes (default %default)')
    optParser.add_option('-u', '--update', action='store', type='float',
                         dest='update_every', default=2,
                         help='number of episodes between target network updates (default %default)')
    optParser.add_option('-s', '--seed', action='store', type='int',
                         dest='seed', default=None,
                         help='random seed (default %default)')
    optParser.add_option('-a', '--tau', action='store', type='float',
                         dest='tau', default=0.005),
    optParser.add_option('-f', '--fulltd3', action='store',
                         dest='full_td3', default='True'),
    optParser.add_option('-c', '--clipped', action='store',
                         dest='clipped_dqn', default='False'),
    optParser.add_option('-d', '--delayed', action='store',
                         dest='delayed_updates', default='False'),
    optParser.add_option('-p', '--policysmoothing', action='store',
                         dest='policy_smoothing', default='False')
    optParser.add_option('-w', '--trainshooting', action='store_true',
                         dest='train_shooting', default=False),
    optParser.add_option('-x', '--traindefense', action='store_true',
                         dest='train_defense', default=False)

    opts, args = optParser.parse_args()
    print("opts", opts)
    print("args", args)
    ############## Hyperparameters ##############
    env_name = opts.env_name
    train_shooting = opts.train_shooting
    train_defense = opts.train_defense
    # creating environment
    if env_name == "LunarLander-v2":
        env = gym.make(env_name, continuous=True)
    elif env_name == "Hockey-v0":
        if train_shooting:
            print("Training shooting")
            env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
        elif train_defense:
            print("Training defense")
            env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
        else:
            print("Training normal game mode")
            env = h_env.HockeyEnv()
    else:
        env = gym.make(env_name)
    render = False
    log_interval = 10  # print avg reward in the interval
    max_episodes = int(opts.max_episodes)  # max training episodes
    max_timesteps = 2000  # max timesteps in one episode

    train_iter = opts.train  # update networks for given batched after every episode
    eps = opts.eps  # noise of TD3 policy
    lr = opts.lr  # learning rate of TD3 policy
    random_seed = opts.seed
    tau = opts.tau # value for Polyak averaging


    #############################################

    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    full_td3 = opts.full_td3.lower() == 'true'
    clipped_dqn = opts.clipped_dqn.lower() == 'true'
    delayed_updates = opts.delayed_updates.lower() == 'true'
    policy_smoothing = opts.policy_smoothing.lower() == 'true'


    action_space = env.action_space
    observation_space = env.observation_space

    td3 = TD3Agent(observation_space, action_space, env_name, eps=eps,
                   learning_rate_actor=lr, learning_rate_critic=lr,
                   update_target_every=opts.update_every, full_td3=full_td3,
                   clipped_dqn=clipped_dqn, delayed_updates=delayed_updates,
                   policy_smoothing=policy_smoothing, tau=tau,
                   train_shooting=train_shooting, train_defense=train_defense)

    #checkpoint = f"./results/hockey/td3_Hockey-v0_{6000}-eps{0.1}-t{32}-l{0.001}-s{None}-tau{0.005}.pth"
    #td3.restore_state(torch.load(checkpoint))


    # logging variables
    rewards = []
    lengths = []
    losses = []
    timestep = 0

    def save_statistics():
        if full_td3:
            create_pickle_dump('hockey')
        elif clipped_dqn and delayed_updates and policy_smoothing:
            create_pickle_dump('hockey')
        elif clipped_dqn and delayed_updates:
            create_pickle_dump('no_smoothing')
        elif clipped_dqn and policy_smoothing:
            create_pickle_dump('no_delay')
        elif delayed_updates and policy_smoothing:
            create_pickle_dump('no_clip')
        elif clipped_dqn:
            create_pickle_dump('only_clip')
        elif delayed_updates:
            create_pickle_dump('only_delay')
        elif policy_smoothing:
            create_pickle_dump('only_smooth')
        else:
            create_pickle_dump('ddpg')

    def create_pickle_dump(folder_name):
        with open(f"./results/{folder_name}/td3_{env_name}-eps{eps}-t{train_iter}-l{lr}-s{random_seed}-tau{tau}-stat.pkl", 'wb') as f:
            pickle.dump({"rewards": rewards, "lengths": lengths, "eps": eps, "train": train_iter,
                         "lr": lr, "update_every": opts.update_every, "losses": losses}, f)

    def save_checkpoint(td3_state):
        if opts.full_td3:
            create_checkpoint(td3_state, 'hockey')
        elif opts.clipped_dqn and opts.delayed_updates and opts.policy_smoothing:
            create_checkpoint(td3_state, 'hockey')
        elif opts.clipped_dqn and opts.delayed_updates:
            create_checkpoint(td3_state, 'no_smoothing')
        elif opts.clipped_dqn and opts.policy_smoothing:
            create_checkpoint(td3_state, 'no_delay')
        elif opts.delayed_updates and opts.policy_smoothing:
            create_checkpoint(td3_state, 'no_clip')
        elif opts.clipped_dqn:
            create_checkpoint(td3_state, 'only_clip')
        elif opts.delayed_updates:
            create_checkpoint(td3_state, 'only_delay')
        elif opts.policy_smoothing:
            create_checkpoint(td3_state, 'only_smooth')
        else:
            create_checkpoint(td3_state, 'ddpg')

    def save_policy_network(policy_network):
        torch.save(policy_network, f"./models/{env_name}_{i_episode}-eps{eps}-t{train_iter}-l{lr}-s{random_seed}-tau{tau}.pth")

    def create_checkpoint(td3_state, folder_name):
        torch.save(td3_state,
                   f'./results/{folder_name}/td3_{env_name}_{i_episode}-eps{eps}-t{train_iter}-l{lr}-s{random_seed}-tau{tau}.pth')


    # training loop
    for i_episode in range(1, max_episodes + 1):
        ob, _info = env.reset()
        if not train_defense or train_shooting:
            player2 = h_env.BasicOpponent(weak=False)
        total_reward = 0

        for t in range(max_timesteps):
            timestep += 1
            done = False

            # For the first 40 episodes we want to explore and choose random actions
            if i_episode <= 40:
                num_actions = td3.action_dim
                a = env.action_space.sample()[:num_actions]
                if train_defense or train_shooting:
                    a2 = [0,0.,0,0]
                else:
                    a2 = player2.act(env.obs_agent_two())

            # After 40 episodes of exploration the agent acts according to policy
            else:
                a = td3.act(ob)[:num_actions]
                if train_defense or train_shooting:
                    a2 = [0,0.,0,0]
                else:
                    a2 = player2.act(env.obs_agent_two())
            (ob_new, reward, done, trunc, _info) = env.step(np.hstack([a, a2]))
            total_reward += reward
            td3.store_transition((ob, a, reward, ob_new, done))
            ob = ob_new
            if done or trunc: break

        if i_episode <= 40:
            losses.extend([env.reward_range[0]])
        else:
            losses.extend(td3.train(train_iter))

        rewards.append(total_reward)
        lengths.append(t)

        # save every 500 episodes
        if i_episode % 500 == 0:
            print("########## Saving a checkpoint... ##########")
            save_checkpoint(td3.state())
            save_statistics()
            save_policy_network(td3.policy)

        # logging
        if i_episode % log_interval == 0:
            avg_reward = np.mean(rewards[-log_interval:])
            avg_length = int(np.mean(lengths[-log_interval:]))

            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, avg_reward))
    save_statistics()


if __name__ == '__main__':
    main()
