from argparse import ArgumentParser
from D4PG import *
import laserhockey.hockey_env as h_env
import random
import time
import pickle
from memory import *
import imageio




def main():
    argParser = ArgumentParser()

    argParser.add_argument("-eps", help="epsilon", default=0.99, type=float)
    argParser.add_argument("-eps_decay", help="epsilon decay", default=0.98, type=float)
    argParser.add_argument("-eps_min", help="minimum epsilon", default=0.05, type=float)
    argParser.add_argument("-eps_optimizer", help="epsilon optimizer", default=0.000001, type=int)
    argParser.add_argument("-reward_batch_norm", help="reward batch normalization", default=True, type=bool)
    argParser.add_argument("-gamma", help="discount factor", default=0.95, type=float)
    argParser.add_argument("-lr_actor", help="learning rate for actor/policy", default=0.0001, type=float)
    argParser.add_argument("-lr_critic", help="learning rate for critic", default=0.0002, type=float)
    argParser.add_argument("-train_iter", help="number of training batches per episode", default=32, type=int)
    argParser.add_argument("-tau", help="soft update factor for target networks", default=0.005, type=float)
    argParser.add_argument("-batch_size", help="batch size", default=128, type=int)
    argParser.add_argument("-hard_updates_every", help="update target networks every n steps", default=30, type=int)
    argParser.add_argument("-hidden_sizes_actor", help="hidden size of actor", default=[256, 128], type=list)
    argParser.add_argument("-hidden_sizes_critic", help="hidden size of critic", default=[256, 256, 128], type=list)
    argParser.add_argument("-max_episodes", help="number of episodes", default=5025, type=int)
    argParser.add_argument("-max_steps", help="number of steps per episode", default=500, type=int)
    argParser.add_argument("-seed", help="random seed", default=20, type=int)
    argParser.add_argument("-mode", help="Mode for training: attack | defense | normal", default='normal')
    argParser.add_argument("-noise", help="Noise for Actions", default=0.2, type=float)
    argParser.add_argument("-render", help="Render the environment", default=False, type=bool)
    argParser.add_argument("-batch_norm", help="Batch normalization", default=False, type=bool)
    argParser.add_argument("-reward_norm", help="Reward normalization", default=False, type=bool)

    opts = argParser.parse_args()

    ############## Hyperparameters ##############


    #env_name = "LunarLander-v2"
    #env = gym.make(env_name, continuous = True)
    env_name = "Hockey"


    if opts.mode == 'normal':
        mode = h_env.HockeyEnv_BasicOpponent.NORMAL
    elif opts.mode == 'attack':
        mode = h_env.HockeyEnv_BasicOpponent.TRAIN_SHOOTING
    elif opts.mode == 'defense':
        mode = h_env.HockeyEnv_BasicOpponent.TRAIN_DEFENSE
    else:
        raise ValueError('Unknown training mode. See --help')


    opponent = h_env.BasicOpponent(weak=True)
    env = h_env.HockeyEnv(mode=mode)
    #env = gym.make(env_name,continuous = True)

    log_interval = 10           # print avg reward in the interval
    max_episodes = opts.max_episodes         # max training episodes
    max_timesteps = opts.max_steps      # max timesteps in one episode

    train_iter = opts.train_iter      # update networks for given batched after every episode
    eps = opts.eps
    eps_decay = opts.eps_decay
    eps_min = opts.eps_min
    optimizer_eps = opts.eps_optimizer
    lr_actor  = opts.lr_actor
    lr_critic = opts.lr_critic  # learning rate of DDPG policy
    random_seed = opts.seed
    gamma = opts.gamma               # discount factor
    tau = opts.tau                 # soft update factor for target networks
    batch_size = opts.batch_size        # batch size for update
    hard_updates_every = opts.hard_updates_every
    noise = opts.noise
    batch_norm = opts.batch_norm
    reward_norm = opts.reward_norm
    d4pg = True
    render = opts.render
    #############################################


    if random_seed is not None:
        torch.manual_seed(random_seed)
    else:
        random_seed = np.random.randint(0, 1000)

    print("State space:", env.observation_space)
    print("Stace space type:", type(env.observation_space))
    print("Action space:", env.action_space)
    print("Random seed:", random_seed)


    d4pg_agent = D4PGAgent(env.observation_space,
                           env.action_space,
                           env_name,
                           eps = eps,
                           eps_decay = eps_decay,
                           eps_min = eps_min,
                           optimizer_eps = optimizer_eps,
                           noise = noise,
                           lr_actor = lr_actor,
                           lr_critic = lr_critic,
                           hard_updates_every = hard_updates_every,
                           batch_size = batch_size,
                           batch_norm = batch_norm,
                           reward_norm = reward_norm,
                           discount = gamma,
                           tau = tau)


    def evaluate_agent(agent, eval_episodes = 100, load_path=None):
        # Set the agent to evaluation mode
        rewards = []
        lengths = []
        rew_stats = []

        num_wins = {}
        num_losses = {}
        opponent = h_env.BasicOpponent(weak=True)

        #output_path = 'render_D4PG_strongBEST.gif'
        #frames = []

        if load_path is not None:
            agent = D4PGAgent(env.observation_space, env.action_space)
            saved_model = torch.load(load_path)
            agent.restore_state(saved_model)

        for episode in range(eval_episodes):
            # Reset the environment and the agent
            observation, _info = env.reset()
            observation_opponent = env.obs_agent_two()

            for timestep in range(max_timesteps):

                action = agent.act(observation, evaluation=True)
                action_opponent = opponent.act(observation_opponent)

                # Take the chosen actions in the environment and observe the new state and reward
                action_step = np.hstack([action, action_opponent])
                observation_new, reward, done, truncated, info = env.step(action_step)
                #frame = env.render(mode='rgb_array')
                #env.render()
                #frames.append(frame)

                observation = observation_new
                observation_opponent = env.obs_agent_two()


                if done or truncated:
                    # Record whether the agent won or lost the game
                    num_wins[episode] = 1 if env.winner == 1 else 0
                    num_losses[episode] = 1 if env.winner == -1 else 0
                    break
                # Log the average reward and episode length every log_interval episodes
            if episode % log_interval == 0 and episode >= 10:
                wins = 0
                losses = 0
                for epi in range(episode - 9, episode):
                    wins += num_wins[epi]
                    losses += num_losses[epi]
                print("Won stats: {} \t Lost stats: {} \t Draws: {}".format(wins, losses,
                                                                            log_interval - wins - losses))
            # Save the frames as a GIF
            #imageio.mimsave(output_path, frames, duration=0.1)


        return num_wins, num_losses


    time_intervals = [500, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]
    np_results = [[0,0,0]]
    for t in time_intervals:

        # Load pretrained agent
        load_path = "paste path to trained model here"
        saved_model = torch.load(load_path)
        d4pg_agent.restore_state(saved_model)
        print("restored agent with t = {}".format(t))

        # Evaluate the trained agent (you may add code here to perform specific evaluation tasks)
        eval_episodes = 500
        num_wins, num_losses = evaluate_agent(d4pg_agent, eval_episodes=eval_episodes)
        np_results.append([sum(num_wins.values()), sum(num_losses.values()), eval_episodes - sum(num_wins.values()) - sum(num_losses.values())])
        print("Won stats: {} \t Lost stats: {} \t Draws: {}".format(sum(num_wins.values()), sum(num_losses.values()), eval_episodes - sum(num_wins.values()) - sum(num_losses.values())))

    np.save("win_loss_stats.npy", np.array(np_results))

if __name__ == '__main__':
    main()