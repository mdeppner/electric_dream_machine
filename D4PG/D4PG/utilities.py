
def create_statistics():
    if d4pg:
        save_statistics("D4PG")

def save_statistics(path):
    with open \
            (f"./results/{path}/d4pg_{env_name}-eps{eps}-t{train_iter}-la{lr_actor}-lc{lr_critic}-s{random_seed}-stat.pkl", 'wb') as f:
        pickle.dump({"rewards" : rewards, "lengths": lengths, "eps": eps, "train": train_iter,
                     "lr_actor": lr_actor, "lr_critic": lr_critic, "update_every": opts.update_every, "losses": losses}, f)

def save_model(path):
    torch.save(ddpg_agent.state(),
               f'./results/{path}_{env_name}_{episode}-eps{eps}-t{train_iter}-la{lr_actor}-lc{lr_critic}-s{random_seed}.pth')

