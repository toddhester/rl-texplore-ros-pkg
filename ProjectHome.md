This project is a framework for running reinforcement learning experiments through ROS. Agents and Environments communicate actions, states, and rewards through a set of ROS messages. The code includes numerous environments (gridworlds, mountain car, cart pole, etc) as well as agents. It also includes a framework for model based agents where various model learning and exploration modules can be inserted along with a variety of available planners (value iteration, policy iteration, prioritized sweeping, uct, parallel uct). It also includes the [TEXPLORE](http://www.cs.utexas.edu/~todd/research.html) algorithm, which uses random forest models, along with an architecture to run model-based RL algorithms in real-time. This repository has been developed by [Todd Hester](http://www.cs.utexas.edu/~todd/) at the University of Texas at Austin.

## Packages Provided ##

This repository includes 5 ROS packages to provide reinforcement learning agents and environments, as well as methods for communicating between them:

  * [rl\_common](http://www.ros.org/wiki/rl_common): Some files that are common to both agents and environments.
  * [rl\_msgs](http://www.ros.org/wiki/rl_msgs): Definitions of ROS messages for agents and envs to communicate (similar to [RL-Glue](http://glue.rl-community.org/wiki/Main_Page)).
  * [rl\_agent](http://www.ros.org/wiki/rl_agent): A library of some RL agents including Q-Learning and TEXPLORE.
  * [rl\_env](http://www.ros.org/wiki/rl_env): A library of some RL environments such as Taxi and Fuel World.
  * [rl\_experiment](http://www.ros.org/wiki/rl_experiment): Code to run some RL experiments without ROS message passing.

## Documentation ##

Full documentation is available on the [ROS wiki](http://www.ros.org/wiki/reinforcement_learning).

The TEXPLORE algorithm and the real-time architecture included in this package are described more fully at the author's [website](http://www.cs.utexas.edu/~todd/research.html).