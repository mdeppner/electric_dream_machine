from argparse import ArgumentParser
from D4PG import *
import laserhockey.hockey_env as h_env
import random
import time
import pickle
from memory import *




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
    argParser.add_argument("-hidden_sizes_actor", help="hidden size of actor", default=[256, 256], type=list)
    argParser.add_argument("-hidden_sizes_critic", help="hidden size of critic", default=[256, 256, 128], type=list)
    argParser.add_argument("-max_episodes", help="number of episodes", default=5025, type=int)
    argParser.add_argument("-max_steps", help="number of steps per episode", default=500, type=int)
    argParser.add_argument("-seed", help="random seed", default=None, type=int)
    argParser.add_argument("-mode", help="Mode for training: attack | defense | normal", default='normal')
    argParser.add_argument("-noise", help="Noise for Actions", default=0.2, type=float)
    argParser.add_argument("-render", help="Render the environment", default=False, type=bool)
    argParser.add_argument("-batch_norm", help="Batch normalization", default=False, type=bool)
    argParser.add_argument("-reward_norm", help="Reward normalization", default=False, type=bool)

    opts = argParser.parse_args()

    ############## Hyperparameters ##############


    env_name = "LunarLander-v2"
    #env = gym.make(env_name, continuous = True)
    #env_name = "Hockey"


    if opts.mode == 'normal':
        mode = h_env.HockeyEnv_BasicOpponent.NORMAL
    elif opts.mode == 'attack':
        mode = h_env.HockeyEnv_BasicOpponent.TRAIN_SHOOTING
    elif opts.mode == 'defense':
        mode = h_env.HockeyEnv_BasicOpponent.TRAIN_DEFENSE
    else:
        raise ValueError('Unknown training mode. See --help')


    #opponent = h_env.BasicOpponent(weak=True)
    #env = h_env.HockeyEnv(mode=mode)
    env = gym.make(env_name,continuous = True)

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

    #load_path = f"./results_final4/D4PG_Hockey_30025-eps0.99-t32-la0.0001-lc0.0002-s914-model.pth"
    #saved_model = torch.load(load_path)
    # print("Configs of model: ", saved_model[4])
    #d4pg_agent.restore_state(saved_model)
    #print("restored agent")

    # logging variables
    rewards = []
    lengths = []
    losses = []

    def create_statistics(episode):
        if d4pg:
            save_statistics("D4PG")
            save_model("D4PG", episode)

    def save_statistics(path):
        with open(f"./lunar_lander/{path}/d4pg_{env_name}-eps{eps}-t{train_iter}-la{lr_actor}-lc{lr_critic}-s{random_seed}-stat.pkl", 'wb') as f:
            pickle.dump({"rewards" : rewards, "lengths": lengths, "eps": eps, "train": train_iter,
                         "lr_actor": lr_actor, "lr_critic": lr_critic, "hard_updates_every": opts.hard_updates_every, "losses": losses}, f)

    def save_model(path, episode):
        torch.save(d4pg_agent.state(),
                   f'./lunar_lander/{path}_{env_name}_{episode}-eps{eps}-t{train_iter}-la{lr_actor}-lc{lr_critic}-s{random_seed}-model.pth')

    def train_agent_hockey():
        # Lists to store episode statistics
        rewards = []
        lengths = []
        rew_stats = []
        loss_stats_critic = []
        loss_stats_actor = []
        num_wins = {}
        num_losses = {}

        # Main training loop for episodes
        for episode in range(1, max_episodes + 1):
            # Reset the environment and the agent at the beginning of each episode
            observation, info = env.reset()


            # Get the initial observation for the opponent agent
            obs_agent2 = env.obs_agent_two()

            # Initialize episode-specific variables
            total_reward = 0
            touched = 0
            first_touch = 0
            second_touch = 0

            # Episode loop: Interact with the environment for a maximum of max_timesteps steps
            for timestep in range(max_timesteps):
                done = False

                # Select an action for the agent based on the current episode and mode
                if episode <= 50:
                    action = env.action_space.sample()[:4]  # Random exploration for the first 50 episodes
                else:
                    action = d4pg_agent.act(observation)

                # Select an action for the opponent agent based on the mode
                if mode == "defense":
                    action_opponent = opponent.act(obs_agent2)
                elif mode == "attack":
                    action_opponent = [0, 0, 0, 0]  # The opponent doesn't take any action in attack mode
                else:
                    action_opponent = opponent.act(obs_agent2)

                # Take the chosen actions in the environment and observe the new state and reward
                action_step = np.hstack([action, action_opponent])
                observation_new, reward, done, truncated, info = env.step(action_step)

                # Update the agent's "touched" state based on the environment information
                touched = max(touched, info['reward_touch_puck'])
                second_touch = max(second_touch, info['reward_touch_puck'])

                #if episode > 2000 and episode % 10 == 0:
                    #env.render()
                    #print("reward: ", reward)
                    #print("info_close_puck :", info['reward_closeness_to_puck'])
                    #print("touched", touched)



                # Calculate the current reward with additional terms based on environment information
                not_touched = 1 - touched

                custom_rew = custom_reward((observation, action, reward, observation_new, done))[2]
                current_reward = 20 * custom_rew - not_touched * 0.1 + touched * 0.1 + second_touch * 0.1

                # Reset second touch variable each step
                second_touch = 0

                # Update the total reward obtained in this episode
                total_reward += current_reward

                # Store the experience (transition) in the D4PG agent's replay buffer
                d4pg_agent.store_transition((observation, action, current_reward, observation_new, done))

                # Update the current observation for the next timestep
                observation = observation_new

                obs_agent2 = env.obs_agent_two()

                # End the episode if the agent reaches a terminal state (done) or a time limit (truncated)
                if done or truncated:
                    # Record whether the agent won or lost the game
                    num_wins[episode] = 1 if env.winner == 1 else 0
                    num_losses[episode] = 1 if env.winner == -1 else 0
                    break

            # Record statistics for this episode
            rew_stats.append(total_reward)
            lengths.append(timestep)
            rewards.append(total_reward)

            # Train the D4PG agent after the initial exploration phase (episode > 50)
            if episode > 50:
                #critic_loss, actor_loss, weights, tree_idx_lst = d4pg_agent.train_iteration(num_iterations=train_iter)
                critic_loss, actor_loss = d4pg_agent.train_iteration(num_iterations=train_iter)

                #d4pg_agent.buffer.update_priorities(tree_idx_lst, torch.abs(critic_loss * weights))
                loss_stats_critic.append(critic_loss)
                loss_stats_actor.append(actor_loss)

            # Save the agent's model and create statistics every 500 episodes
            if episode % 500 == 0:
                print("########## Saving a checkpoint... ##########")
                save_model("D4PG", episode)
                create_statistics(episode)

            # Log the average reward and episode length every log_interval episodes
            if episode % log_interval == 0:
                avg_reward = np.mean(rewards[-log_interval:])
                avg_length = int(np.mean(lengths[-log_interval:]))
                if episode > 25:

                    avg_loss_critic = np.mean(loss_stats_critic[-log_interval:])
                    avg_loss_actor = np.mean(loss_stats_actor[-log_interval:])
                    print('Episode {} \t avg length: {} \t reward: {} \t loss_critic: {} \t loss_actor: {}'.format(episode, avg_length, avg_reward, avg_loss_critic, avg_loss_actor))
                    wins = 0
                    losses = 0
                    for epi in range(episode - 9, episode):
                        wins += num_wins[epi]
                        losses += num_losses[epi]
                    print("Won stats: {} \t Lost stats: {} \t Draws: {}".format(wins, losses, log_interval - wins - losses))


        # Create statistics at the end of training
        create_statistics(episode)

        return d4pg_agent


    def train_agent_lunar():
        for episode in range(1, max_episodes + 1):
            # Reset the environment and the agent at the beginning of each episode
            observation, _info = env.reset()
            d4pg_agent.reset()

            total_reward = 0

            # Episode loop: Interact with the environment for a maximum of max_timesteps steps
            for timestep in range(max_timesteps):
                # Increment timestep count
                timestep += 1

                done = False
                if episode <= 50:
                    action = env.action_space.sample()
                else:
                    action = d4pg_agent.act(observation)

                # Take the chosen action in the environment and observe the new state and reward
                new_observation, reward, done, truncated, _info = env.step(action)
                # print("Reward:", reward)

                # Update the total reward obtained in this episode
                total_reward += reward

                # Store the experience (transition) in the DDPG agent's replay buffer
                d4pg_agent.store_transition((observation, action, reward, new_observation, done))

                # Update the current observation for the next timestep
                observation = new_observation

                # End the episode if the agent reaches a terminal state (done) or a time limit (truncated)
                if done or truncated:
                    break

            if episode <= 50:
                pass
            else:
                losses.extend(d4pg_agent.train(train_iter))

            rewards.append(total_reward)
            lengths.append(timestep)

            # save every 500 episodes
            if episode % 500 == 0:
                print("########## Saving a checkpoint... ##########")
                save_model("D4PG")
                create_statistics()

            # logging
            if episode % log_interval == 0:
                avg_reward = np.mean(rewards[-log_interval:])
                avg_length = int(np.mean(lengths[-log_interval:]))

                print('Episode {} \t avg length: {} \t reward: {}'.format(episode, avg_length, avg_reward))
        create_statistics()

        return d4pg_agent

    def evaluate_agent(agent, eval_episodes = 100, load_path=None):
        # Set the agent to evaluation mode
        agent.eval()

        num_wins = {}
        num_losses = {}
        opponent = h_env.BasicOpponent(weak=True)

        if load_path is not None:
            agent = D4PGAgent(env.observation_space, env.action_space)
            saved_model = torch.load(load_path)
            agent.restore_state(saved_model)

        for episode in range(eval_episodes):
            # Reset the environment and the agent
            observation = env.reset()
            observation_opponent = env.obs_agent_two()

            #if (env.puck.position[0] < 5 and agent._config['mode'] == 'defense') or (
            #        env.puck.position[0] > 5 and agent._config['mode'] == 'shooting'
            #):
            #    continue

            for timestep in range(max_timesteps):

                action = agent.act(observation, evaluation=True)
                action_opponent = opponent.act(observation_opponent)

                # Take the chosen actions in the environment and observe the new state and reward
                action_step = np.hstack([action, action_opponent])
                observation_new, reward, done, truncated, info = env.step(action_step)

                observation = observation_new
                observation_opponent = env.obs_agent_two()


                if done or truncated:
                    # Record whether the agent won or lost the game
                    num_wins[episode] = 1 if env.winner == 1 else 0
                    num_losses[episode] = 1 if env.winner == -1 else 0
                    break

        return num_wins, num_losses

    # Train the D4PG agent
    #d4pg_agent_trained = train_agent_hockey()
    d4pg_agent_trained = train_agent_lunar()

    # Evaluate the trained agent (you may add code here to perform specific evaluation tasks)
    eval_episodes = 1000
    num_wins, num_losses = evaluate_agent(d4pg_agent_trained, eval_episodes=eval_episodes)
    print("Won stats: {} \t Lost stats: {} \t Draws: {}".format(sum(num_wins.values()), sum(num_losses.values()), eval_episodes - sum(num_wins.values()) - sum(num_losses.values())))

if __name__ == '__main__':
    main()