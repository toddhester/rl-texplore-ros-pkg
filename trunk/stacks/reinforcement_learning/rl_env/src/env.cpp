#include <ros/ros.h>
#include "std_msgs/String.h"

#include <rl_msgs/RLStateReward.h>
#include <rl_msgs/RLEnvDescription.h>
#include <rl_msgs/RLAction.h>
#include <rl_msgs/RLExperimentInfo.h>
#include <rl_msgs/RLEnvSeedExperience.h>

#include <ros/callback_queue.h>
#include <tf/transform_broadcaster.h>

#include <rl_common/core.hh>
#include <rl_common/Random.h>

#include "Env/RobotCarVel.hh"
#include "Env/fourrooms.hh"
#include "Env/tworooms.hh"
#include "Env/taxi.hh"
#include "Env/FuelRooms.hh"
#include "Env/stocks.hh"
#include "Env/energyrooms.hh"
#include "Env/MountainCar.hh"
#include "Env/CartPole.hh"

#include <getopt.h>
#include <stdlib.h>

#define NODE "RLEnvironment"

static ros::Publisher out_env_desc;
static ros::Publisher out_env_sr;
static ros::Publisher out_seed;

Environment* e;
Random rng;
bool PRINTS = false;//true;
int seed = 1;
char* envType;

// some default parameters
bool stochastic = true;
int nstocks = 3;
int nsectors = 3;
int delay = 0;
bool lag = false;


void displayHelp(){
  cout << "\n Call env --env type [options]\n";
  cout << "Env types: taxi tworooms fuellow fuelhigh mcar cartpole car2to7 stocks\n";
  cout << "\n Options:\n";
  cout << "--seed value (integer seed for random number generator)\n";
  cout << "--deterministic (deterministic version of domain)\n";
  cout << "--stochastic (stochastic version of domain)\n";
  cout << "--delay value (# steps of action delay (for mcar and tworooms)\n";
  cout << "--lag (turn on brake lag for car driving domain)\n";
  cout << "--nsectors value (# sectors for stocks domain)\n";
  cout << "--nstocks value (# stocks for stocks domain)\n";
  cout << "--prints (turn on debug printing of actions/rewards)\n";
  exit(-1);
}


/** process action from the agent */
void processAction(const rl_msgs::RLAction::ConstPtr &actionIn){

  rl_msgs::RLStateReward sr;

  // process action from the agent, affecting the environment
  sr.reward = e->apply(actionIn->action);
  sr.state = e->sensation();
  sr.terminal = e->terminal();

  // publish the state-reward message
  if (PRINTS) cout << "Got action " << actionIn->action << " at state: " << sr.state[0] << ", " << sr.state[1] << ", reward: " << sr.reward << endl;

  out_env_sr.publish(sr);

}

/** Process end-of-episode reward info. 
    Mostly to start new episode. */
void processEpisodeInfo(const rl_msgs::RLExperimentInfo::ConstPtr &infoIn){
  // start new episode if terminal
  if (PRINTS) cout << "Episode " << infoIn->episode_number << " terminated with reward: " << infoIn->episode_reward << ", start new episode " << endl;

  e->reset();

  rl_msgs::RLStateReward sr;
  sr.reward = 0;
  sr.state = e->sensation();
  sr.terminal = false;
  out_env_sr.publish(sr);
}


/** init the environment, publish a description. */
void initEnvironment(){

  // init the environment
  e = NULL;
  rl_msgs::RLEnvDescription desc;
  

  if (strcmp(envType, "cartpole") == 0){
    desc.title = "Environment: Cart Pole\n";
    e = new CartPole(rng, stochastic);
  }

  else if (strcmp(envType, "mcar") == 0){
    desc.title = "Environment: Mountain Car\n";
    e = new MountainCar(rng, stochastic, false, delay);
  }

  // taxi
  else if (strcmp(envType, "taxi") == 0){
    desc.title = "Environment: Taxi\n";
    e = new Taxi(rng, stochastic);
  }

  // two rooms
  else if (strcmp(envType, "tworoom") == 0){
    desc.title = "Environment: TwoRooms\n";
    e = new TwoRooms(rng, stochastic, true, delay, false);
  }

  // car vel, 2 to 7
  else if (strcmp(envType, "car2to7") == 0){
    desc.title = "Environment: Car Velocity 2 to 7 m/s\n";
    e = new RobotCarVel(rng, false, true, false, false);
  }
  // car vel, 7 to 2
  else if (strcmp(envType, "car7to2") == 0){
    desc.title = "Environment: Car Velocity 7 to 2 m/s\n";
    e = new RobotCarVel(rng, false, false, false, false);
  }
  // car vel, random vels
  else if (strcmp(envType, "carrandom") == 0){
    desc.title = "Environment: Car Velocity Random Velocities\n";
    e = new RobotCarVel(rng, true, false, false, false);
  }
  // car vel, random vels, with lag
  else if (strcmp(envType, "carrandomlag") == 0){
    desc.title = "Environment: Car Velocity Random Velocities with lag\n";
    e = new RobotCarVel(rng, true, false, false, true);
  }
  // car vel, 7 to 2 with lag
  else if (strcmp(envType, "car7to2lag") == 0){
    desc.title = "Environment: Car Velocity 7 to 2 m/s with lag\n";
    e = new RobotCarVel(rng, false, false, false, true);
  }
  // car vel, 10 to 6 with lag
  else if (strcmp(envType, "car10to6lag") == 0){
    desc.title = "Environment: Car Velocity 10 to 6 m/s with lag\n";
    e = new RobotCarVel(rng, false, false, true, true);
  }
  // car vel, 10 to 6 no lag
  else if (strcmp(envType, "car10to6") == 0){
    desc.title = "Environment: Car Velocity 10 to 6 m/s\n";
    e = new RobotCarVel(rng, false, false, true, false);
  }

  // four rooms
  else if (strcmp(envType, "fourroom") == 0){
    desc.title = "Environment: FourRooms\n";
    e = new FourRooms(rng, stochastic, true, false);
  }

  // four rooms with energy level
  else if (strcmp(envType, "energy") == 0){
    desc.title = "Environment: EnergyRooms\n";
    e = new EnergyRooms(rng, stochastic, true, false);
  }

  // gridworld with fuel (fuel stations on top and bottom with random costs)
  else if (strcmp(envType, "fuellow") == 0){
    desc.title = "Environment: FuelRooms, Little variation in fuel cost\n";
    e = new FuelRooms(rng, false, stochastic);
  }

  // gridworld with fuel (fuel stations on top and bottom with random costs)
  else if (strcmp(envType, "fuelhigh") == 0){
    desc.title = "Environment: FuelRooms, Large variation in fuel cost\n";
    e = new FuelRooms(rng, true, stochastic);
  }

  // stocks
  else if (strcmp(envType, "stocks") == 0){
    desc.title = "Enironment: Stocks\n";
    e = new Stocks(rng, stochastic, nsectors, nstocks);
  }

  else {
    std::cerr << "Invalid env type" << endl;
    displayHelp();
    exit(-1);
  }

  // fill in some more description info
  desc.num_actions = e->getNumActions();
  desc.episodic = e->isEpisodic();

  std::vector<float> maxFeats;
  std::vector<float> minFeats;

  e->getMinMaxFeatures(&minFeats, &maxFeats);
  desc.num_states = minFeats.size();
  desc.min_state_range = minFeats;
  desc.max_state_range = maxFeats;

  desc.stochastic = stochastic;
  float minReward;
  float maxReward;
  e->getMinMaxReward(&minReward, &maxReward);
  desc.max_reward = maxReward;
  desc.reward_range = maxReward - minReward;

  cout << desc.title << endl;

  // publish environment description
  out_env_desc.publish(desc);

  sleep(1);

  // send experiences
  std::vector<experience> seeds = e->getSeedings();
  for (unsigned i = 0; i < seeds.size(); i++){
    rl_msgs::RLEnvSeedExperience seed;
    seed.from_state = seeds[i].s;
    seed.to_state   = seeds[i].next;
    seed.action     = seeds[i].act;
    seed.reward     = seeds[i].reward;
    seed.terminal   = seeds[i].terminal;
    out_seed.publish(seed);
  }

  // now send first state message
  rl_msgs::RLStateReward sr;
  sr.terminal = false;
  sr.reward = 0;
  sr.state = e->sensation();
  out_env_sr.publish(sr);
  
}


/** Main function to start the env node. */
int main(int argc, char *argv[])
{
  ros::init(argc, argv, NODE);
  ros::NodeHandle node;

  if (argc < 2){
    cout << "Wrong number of arguments: <env> <stochastic> <seed>\n";
    exit(-1);
  }

  // env and seed are required
  if (argc < 3){
    displayHelp();
    exit(-1);
  }

  // parse options to change these parameters
  envType = argv[1];
  seed = std::atoi(argv[2]);

  // parse env type first
  bool gotEnv = false;
  for (int i = 1; i < argc-1; i++){
    if (strcmp(argv[i], "--env") == 0){
      gotEnv = true;
      envType = argv[i+1];
    }
  }
  if (!gotEnv) {
    cout << "--env type  option is required" << endl;
    displayHelp();
  }

  // now parse other options
  char ch;
  const char* optflags = "ds:";
  int option_index = 0;
  static struct option long_options[] = {
    {"env", 1, 0, 'e'},
    {"deterministic", 0, 0, 'd'},
    {"stochastic", 0, 0, 's'},
    {"delay", 1, 0, 'a'},
    {"nsectors", 1, 0, 'c'},
    {"nstocks", 1, 0, 't'},
    {"lag", 0, 0, 'l'},
    {"nolag", 0, 0, 'o'},
    {"seed", 1, 0, 'x'},
    {"prints", 0, 0, 'p'}
  };

  while(-1 != (ch = getopt_long_only(argc, argv, optflags, long_options, &option_index))) {
    switch(ch) {

    case 'x':
      seed = std::atoi(optarg);
      cout << "seed: " << seed << endl;
      break;

    case 'd':
      stochastic = false;
      cout << "stochastic: " << stochastic << endl;
      break;

    case 's':
      stochastic = true;
      cout << "stochastic: " << stochastic << endl;
      break;

    case 'a':
      delay = std::atoi(optarg);
      cout << "delay steps: " << delay << endl;
      break;

    case 'c':
      nsectors = std::atoi(optarg);
      cout << "nsectors: " << nsectors << endl;
      break;
    
    case 't':
      nstocks = std::atoi(optarg);
      cout << "nstocks: " << nstocks << endl;
      break;

    case 'l':
      lag = true;
      cout << "lag: " << lag << endl;
      break;

    case 'o':
      lag = false;
      cout << "lag: " << lag << endl;
      break;

    case 'e':
      // already processed this one
      cout << "env: " << envType << endl;
      break;

    case 'p':
      PRINTS = true;
      break;

    case 'h':
    case '?':
    case 0:
    default:
      displayHelp();
      break;
    }
  }


  int qDepth = 1;

  // Set up Publishers
  ros::init(argc, argv, "my_tf_broadcaster");
  tf::Transform transform;

  // Set up Publishers
  out_env_desc = node.advertise<rl_msgs::RLEnvDescription>("rl_env/rl_env_description",qDepth,true);
  out_env_sr = node.advertise<rl_msgs::RLStateReward>("rl_env/rl_state_reward",qDepth,false);
  out_seed = node.advertise<rl_msgs::RLEnvSeedExperience>("rl_env/rl_seed",20,false);

  // Set up subscribers
  ros::TransportHints noDelay = ros::TransportHints().tcpNoDelay(true);
  ros::Subscriber rl_action =  node.subscribe("rl_agent/rl_action", qDepth, processAction, noDelay);
  ros::Subscriber rl_exp_info =  node.subscribe("rl_agent/rl_experiment_info", qDepth, processEpisodeInfo, noDelay);

  // publish env description, first state
  // Setup RL World
  rng = Random(1+seed);
  initEnvironment();


  ROS_INFO(NODE ": starting main loop");
  
  ros::spin();                          // handle incoming data
  //while (ros::ok()){
  //  ros::getGlobalCallbackQueue()->callAvailable(ros::WallDuration(0.1));
  //}

  return 0;
}




