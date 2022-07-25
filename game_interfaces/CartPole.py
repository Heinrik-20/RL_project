import gym
import pygame
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from gym.utils.play import play
from game_interfaces.GameInterface import GameInterface

class CartPole(GameInterface):

    def __init__(self, to_play=False):
        self.__ai = not to_play
        if (to_play):
            self.__to_play()
        else:
            self.env = gym.make("CartPole-v1").unwrapped
            self.env.reset()
        return

    def get_actions(self):
        raise NotImplementedError

    def play_step(self, action):
        assert(self.__ai, f"Function can only be called for AI game playing agent")
        return self.env.step(action.item)


    def get_current_state(self):
        assert(self.__ai, f"Function can only be called for AI game playing agent")

        resize = resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

        def get_cart_location(screen_width):
            world_width = self.env.x_threshold * 2
            scale = screen_width / world_width
            return int(self.env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

        def get_screen():
            # Returned screen requested by gym is 400x600x3, but is sometimes larger
            # such as 800x1200x3. Transpose it into torch order (CHW).
            screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
            # Cart is in the lower half, so strip off the top and bottom of the screen
            _, screen_height, screen_width = screen.shape
            screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
            view_width = int(screen_width * 0.6)
            cart_location = get_cart_location(screen_width)
            if cart_location < view_width // 2:
                slice_range = slice(view_width)
            elif cart_location > (screen_width - view_width // 2):
                slice_range = slice(-view_width, None)
            else:
                slice_range = slice(cart_location - view_width // 2,
                                    cart_location + view_width // 2)
            # Strip off the edges, so that we have a square image centered on a cart
            screen = screen[:, :, slice_range]
            # Convert to float, rescale, convert to torch tensor
            # (this doesn't require a copy)
            screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
            screen = torch.from_numpy(screen)
            # Resize, and add a batch dimension (BCHW)
            return resize(screen).unsqueeze(0)

        return get_screen()
    
    def __to_play(self):
        mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}
        play(gym.make("CartPole-v1"), keys_to_action=mapping)
        return 

if __name__ == "__main__":
    to_play = True if input("Play Manually (y/n): ") == "y" else False
    cartPole = CartPole(to_play=to_play)
