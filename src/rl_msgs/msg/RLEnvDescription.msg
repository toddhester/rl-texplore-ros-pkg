# Message that contains details about an RL enviroment/task
float32 num_actions
float32 num_states

#Optional information to describe the range of a continous state
#Some RL algorithms may need this to discretize the state space
float32[] min_state_range
float32[] max_state_range

#Info needed for r-max some other methods
float32 max_reward
float32 reward_range

bool stochastic
bool episodic
string title
