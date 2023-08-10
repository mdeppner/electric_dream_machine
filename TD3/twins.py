import numpy as np
import torch


from laserhockey.hockey_env import BasicOpponent
from client.remoteControllerInterface import RemoteControllerInterface
from client.backend.client import Client
from TD3 import Actor

class TwinsOpponent(BasicOpponent, RemoteControllerInterface):

    def __init__(self, weak, keep_mode=True):
        BasicOpponent.__init__(self, weak=weak, keep_mode=keep_mode)
        RemoteControllerInterface.__init__(self, identifier='TD3')

        self.policy = torch.load("./models/Hockey-v0_9000-eps0.1-t32-l0.001-sNone-tau0.005.pth")
        self.policy.eval()
    def remote_act(self,
            obs : np.ndarray,
           ) -> np.ndarray:

        action = self.policy.predict(obs)
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        return action





if __name__ == '__main__':
    controller = TwinsOpponent(weak=False)

    # Play n (None for an infinite amount) games and quit
    client = Client(username='Electric Dream Machine',
                    password='bioP9heTei',
                    controller=controller,
                    output_path='tournament_logs/twins', # rollout buffer with finished games will be saved in here
                    interactive=False,
                    op='start_queuing',
                    # server_addr='localhost',
                    num_games=None)

    # Start interactive mode. Start playing by typing start_queuing. Stop playing by pressing escape and typing stop_queueing
    # client = Client(username='user0',
    #                 password='1234',
    #                 controller=controller,
    #                 output_path='logs/basic_opponents',
    #                )
