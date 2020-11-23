from agents.a2c_lstm import *
from agents.c51_ddqn import *
from agents.drqn import *


architecture = "c51"
config_file = "scenarios/defend_the_center.cfg"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

game = DoomGame()
game.load_config(config_file)
game.set_sound_enabled(True)
game.set_screen_resolution(ScreenResolution.RES_640X480)
game.set_window_visible(False)
game.init()


if architecture == "c51":
    num_atoms = 51
    img_rows , img_cols = 64, 64
    img_channels = 4 
    state_size = (img_rows, img_cols, img_channels)

    agent = C51Agent(state_size, game.get_available_buttons_size(), num_atoms)

    agent.model = Networks.value_distribution_network(state_size, num_atoms, game.get_available_buttons_size(), agent.learning_rate)
    agent.target_model = Networks.value_distribution_network(state_size, num_atoms, game.get_available_buttons_size(), agent.learning_rate)

    trainC51(game, agent)

elif architecture == "drqn":
    img_rows , img_cols = 64, 64
    img_channels = 3 
    trace_length = 4

    state_size = (trace_length, img_rows, img_cols, img_channels)
    agent = DoubleDQNAgent(state_size, game.get_available_buttons_size(), trace_length)

    agent.model = Networks.drqn(state_size, game.get_available_buttons_size(), agent.learning_rate)
    agent.target_model = Networks.drqn(state_size, game.get_available_buttons_size(), agent.learning_rate)

    trainDRQN(game, agent)

elif architecture == "a2c":
    img_rows , img_cols = 64, 64
    img_channels = 3 
    trace_length = 4

    state_size = (trace_length, img_rows, img_cols, img_channels)
    
    agent = A2CAgent(state_size, game.get_available_buttons_size(), trace_length)
    agent.model = Networks.a2c_lstm(state_size, game.get_available_buttons_size(), agent.value_size, agent.learning_rate)

    trainA2C(game, agent)