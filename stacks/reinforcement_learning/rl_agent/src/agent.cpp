#include <ros/ros.h>

#include <rl_msgs/RLStateReward.h>
#include <rl_msgs/RLEnvDescription.h>
#include <rl_msgs/RLAction.h>
#include <rl_msgs/RLExperimentInfo.h>
#include <rl_msgs/RLEnvSeedExperience.h>

#include <ros/callback_queue.h>

#include <rl_common/core.hh>
#include <rl_common/Random.h>
#include <rl_common/ExperienceFile.hh>

#include <rl_agent/DiscretizationAgent.hh>
#include <rl_agent/QLearner.hh>
#include <rl_agent/ModelBasedAgent.hh>
#include <rl_agent/SavedPolicy.hh>
#include <rl_agent/Dyna.hh>
#include "std_msgs/String.h"

#include <getopt.h>
#include <stdlib.h>

#define NODE "RLAgent"

static ros::Publisher out_rl_action;
static ros::Publisher out_exp_info;

bool firstAction = true;
int seed = 1;

Agent* agent = NULL;
bool PRINTS = true;

rl_msgs::RLExperimentInfo info;
char* agentType;

// default parameters
float discountfactor = 0.99;
float epsilon = 0.1;
float alpha = 0.3;
float initialvalue = 0.0;
float actrate = 10.0;
float lambda = 0.1;
int M = 5;
int model = C45TREE;
int explore = GREEDY;
int modelcombo = AVERAGE;
int planner = PAR_ETUCT_ACTUAL;
int nmodels = 5;
bool reltrans = true;
int nstates = 0;
int k = 1000;
char *filename = NULL;
// possibly over-written by command line arguments


void displayHelp(){
  cout << "\n Call agent --agent type [options]\n";
  cout << "Agent types: qlearner, modelbased, rmax, texplore, dyna, savedpolicy\n";
  cout << "\n Options:\n";
  cout << "--seed value (integer seed for random number generator)\n";
  cout << "--gamma value (discount factor between 0 and 1)\n";
  cout << "--epsilon value (epsilon for epsilon-greedy exploration)\n";
  cout << "--alpha value (learning rate alpha)\n";
  cout << "--initialvalue value (initial q values)\n";
  cout << "--actrate value (action selection rate (Hz))\n";
  cout << "--lamba value (lamba for eligibility traces)\n";
  cout << "--m value (parameter for R-Max)\n";
  cout << "--k value (For Dyna: # of model based updates to do between each real world update)\n";
  cout << "--filename file (file to load saved policy from for savedpolicy agent)\n";
  cout << "--model type (tabular,tree,m5tree)\n";
  cout << "--planner type (vi,pi,sweeping,uct,parallel-uct,delayed-uct,delayed-parallel-uct)\n";
  cout << "--explore type (unknowns,greedy,epsilongreedy)\n";
  cout << "--combo type (average,best,separate)\n";
  cout << "--nmodels value (# of models)\n";
  cout << "--nstates value (optionally discreteize domain into value # of states on each feature)\n";
  cout << "--reltrans (learn relative transitions)\n";
  cout << "--abstrans (learn absolute transitions)\n";
  cout << "--prints (turn on debug printing of actions/rewards)\n";

  exit(-1);

}

/** Process the state/reward message from the environment */
void processState(const rl_msgs::RLStateReward::ConstPtr &stateIn){

  if (agent == NULL){
    cout << "no agent yet" << endl;
    return;
  }

  rl_msgs::RLAction a;
  
  // first action
  if (firstAction){
    a.action = agent->first_action(stateIn->state);
    info.episode_reward = 0;
  } else {
    info.episode_reward += stateIn->reward;
    // if terminal, no action, but calculate reward sum
    if (stateIn->terminal){
      agent->last_action(stateIn->reward);
      cout << "Episode " << info.episode_number << " reward: " << info.episode_reward << endl;
      // publish episode reward message
      out_exp_info.publish(info);
      info.episode_number++;
      info.episode_reward = 0;
      firstAction = true;
      return;
    } else {
      a.action = agent->next_action(stateIn->reward, stateIn->state);
    }
  }
  firstAction = false;

  // publish agent's action
  out_rl_action.publish(a);
}


/** Process seeds for initializing model */
void processSeed(const rl_msgs::RLEnvSeedExperience::ConstPtr &seedIn){

  if (agent == NULL){
    cout << "no agent yet" << endl;
    return;
  }
  
  std::vector<experience> seeds;
  experience seed1;
  seed1.s = seedIn->from_state;
  seed1.next = seedIn->to_state;
  seed1.act = seedIn->action;
  seed1.reward = seedIn->reward;
  seed1.terminal = seedIn->terminal;
  seeds.push_back(seed1);

  agent->seedExp(seeds);

}

/** Process the env description message from the environment */
void processEnvDescription(const rl_msgs::RLEnvDescription::ConstPtr &envIn){

  // initialize the agent based on some info from the environment descriptor
  Random rng(seed+1);
  agent = NULL;


  if (strcmp(agentType, "qlearner") == 0){
    cout << "Agent: QLearner" << endl;
    agent = new QLearner(envIn->num_actions,
                         discountfactor, // gamma
                         initialvalue, // initial value
                         alpha, // alpha
                         epsilon, // epsilon
                         rng);
  } 

  else if (strcmp(agentType, "rmax") == 0 || strcmp(agentType, "texplore") == 0 || strcmp(agentType, "modelbased") == 0){
    cout << "Agent: Model Based" << endl;
    agent = new ModelBasedAgent(envIn->num_actions,
                                discountfactor,
                                envIn->max_reward, envIn->reward_range,
                                model, explore, modelcombo,
                                nmodels,
                                planner,
                                0.1, // epsilon
                                lambda,
                                (1.0/actrate), //0.1, //0.1, //0.01, // max time
                                M,
                                envIn->min_state_range, envIn->max_state_range,
                                nstates,
                                0, 0, false, reltrans, 0.2, 
                                envIn->stochastic, envIn->episodic,
                                rng);
    
  }

  else if (strcmp(agentType, "dyna") == 0){
    cout << "Agent: Dyna" << endl;
    agent = new Dyna(envIn->num_actions, discountfactor,
                     initialvalue, alpha, k, epsilon,
                     rng);
  }

  else if (strcmp(agentType, "savedpolicy") == 0){
    cout << "Agent: Saved Policy" << endl;
    agent = new SavedPolicy(envIn->num_actions, filename);
  }

  else {
    cout << "Invalid Agent!" << endl;
    displayHelp();
    exit(-1);
  }

  Agent* a2 = agent;
  // not for model based when doing continuous model
  if (nstates > 0 && (model != M5ALLMULTI || strcmp(agentType, "qlearner") == 0)){
    int totalStates = powf(nstates,envIn->min_state_range.size());
    if (PRINTS) cout << "Discretize with " << nstates << ", total: " << totalStates << endl;
    agent = new DiscretizationAgent(nstates, a2, 
                                    envIn->min_state_range, 
                                    envIn->max_state_range, PRINTS);
  }
  

  firstAction = true;
  info.episode_number = 0;
  info.episode_reward = 0;
}


/** Main method to start up agent node. */
int main(int argc, char *argv[])
{
  ros::init(argc, argv, NODE);
  ros::NodeHandle node;

  // agent is required required
  if (argc < 3){
    cout << "--agent type  option is required" << endl;
    displayHelp();
    exit(-1);
  }

  // parse options to change these parameters
  agentType = argv[1];
  seed = std::atoi(argv[2]);

  // parse agent type first
  bool gotAgent = false;
  for (int i = 1; i < argc-1; i++){
    if (strcmp(argv[i], "--agent") == 0){
      gotAgent = true;
      agentType = argv[i+1];
    }
  }
  if (!gotAgent) {
    cout << "--agent type  option is required" << endl;
    displayHelp();
  }

  // set some default options for rmax or texplore
  if (strcmp(agentType, "rmax") == 0){
    model = RMAX;
    explore = EXPLORE_UNKNOWN;
    modelcombo = BEST;
    planner = VALUE_ITERATION;
    nmodels = 1;
    reltrans = false;
    M = 5;
  } else if (strcmp(agentType, "texplore") == 0){
    model = C45TREE;
    explore = GREEDY;
    modelcombo = AVERAGE;
    planner = PAR_ETUCT_ACTUAL;
    nmodels = 5;
    reltrans = true;
    M = 0;
  } 

  char ch;
  const char* optflags = "geairlmoxpcn:";
  int option_index = 0;
  static struct option long_options[] = {
    {"gamma", 1, 0, 'g'},
    {"discountfactor", 1, 0, 'g'},
    {"epsilon", 1, 0, 'e'},
    {"alpha", 1, 0, 'a'},
    {"initialvalue", 1, 0, 'i'},
    {"actrate", 1, 0, 'r'},
    {"lambda", 1, 0, 'l'},
    {"m", 1, 0, 'm'},
    {"model", 1, 0, 'o'},
    {"explore", 1, 0, 'x'},
    {"planner", 1, 0, 'p'},
    {"combo", 1, 0, 'c'},
    {"nmodels", 1, 0, 'n'},
    {"reltrans", 0, 0, 't'},
    {"abstrans", 0, 0, 'b'},
    {"seed", 1, 0, 's'},
    {"agent", 1, 0, 'q'},
    {"prints", 0, 0, 'd'},
    {"nstates", 1, 0, 'w'},
    {"k", 1, 0, 'k'},
    {"filename", 1, 0, 'f'}

  };

  while(-1 != (ch = getopt_long_only(argc, argv, optflags, long_options, &option_index))) {
    switch(ch) {

    case 'g':
      discountfactor = std::atof(optarg);
      cout << "discountfactor: " << discountfactor << endl;
      break;
      
    case 'e':
      epsilon = std::atof(optarg);
      cout << "epsilon: " << epsilon << endl;
      break;
      
    case 'k':
      k = std::atoi(optarg);
      cout << "k: " << k << endl;
      break;

    case 'f':
      filename = optarg;
      cout << "policy filename: " <<  filename << endl;
      break;

    case 'a':
      alpha = std::atof(optarg);
      cout << "alpha: " << alpha << endl;
      break;

    case 'i':
      initialvalue = std::atof(optarg);
      cout << "initialvalue: " << initialvalue << endl;
      break;

    case 'r':
      actrate = std::atof(optarg);
      cout << "actrate: " << actrate << endl;
      break;

    case 'l':
      lambda = std::atof(optarg);
      cout << "lambda: " << lambda << endl;
      break;

    case 'm':
      M = std::atoi(optarg);
      cout << "M: " << M << endl;
      break;

    case 'o':
      {
        if (strcmp(optarg, "tabular") == 0) model = RMAX;
        else if (strcmp(optarg, "tree") == 0) model = C45TREE;
        else if (strcmp(optarg, "texplore") == 0) model = C45TREE;
        else if (strcmp(optarg, "c45tree") == 0) model = C45TREE;
        else if (strcmp(optarg, "m5tree") == 0) model = M5ALLMULTI;
        cout << "model: " << modelNames[model] << endl;
        break;
      }
      
    case 'x':
      {
        if (strcmp(optarg, "unknown") == 0) explore = EXPLORE_UNKNOWN;
        else if (strcmp(optarg, "greedy") == 0) explore = GREEDY;
        else if (strcmp(optarg, "epsilongreedy") == 0) explore = EPSILONGREEDY;
        else if (strcmp(optarg, "unvisitedstates") == 0) explore = UNVISITED_BONUS;
        else if (strcmp(optarg, "unvisitedactions") == 0) explore = UNVISITED_ACT_BONUS;
        cout << "explore: " << exploreNames[explore] << endl;
        break;
      }

    case 'p':
      {
        if (strcmp(optarg, "vi") == 0) planner = VALUE_ITERATION;
        else if (strcmp(optarg, "valueiteration") == 0) planner = VALUE_ITERATION;
        else if (strcmp(optarg, "policyiteration") == 0) planner = POLICY_ITERATION;
        else if (strcmp(optarg, "pi") == 0) planner = POLICY_ITERATION;
        else if (strcmp(optarg, "sweeping") == 0) planner = PRI_SWEEPING;
        else if (strcmp(optarg, "prioritizedsweeping") == 0) planner = PRI_SWEEPING;
        else if (strcmp(optarg, "uct") == 0) planner = ET_UCT_ACTUAL;
        else if (strcmp(optarg, "paralleluct") == 0) planner = PAR_ETUCT_ACTUAL;
        else if (strcmp(optarg, "realtimeuct") == 0) planner = PAR_ETUCT_ACTUAL;
        else if (strcmp(optarg, "realtime-uct") == 0) planner = PAR_ETUCT_ACTUAL;
        else if (strcmp(optarg, "parallel-uct") == 0) planner = PAR_ETUCT_ACTUAL;
        else if (strcmp(optarg, "delayeduct") == 0) planner = POMDP_ETUCT;
        else if (strcmp(optarg, "delayed-uct") == 0) planner = POMDP_ETUCT;
        else if (strcmp(optarg, "delayedparalleluct") == 0) planner = POMDP_PAR_ETUCT;
        else if (strcmp(optarg, "delayed-parallel-uct") == 0) planner = POMDP_PAR_ETUCT;
        cout << "planner: " << plannerNames[planner] << endl;
        break;
      }

    case 'c':
      {
        if (strcmp(optarg, "average") == 0) modelcombo = AVERAGE;
        else if (strcmp(optarg, "weighted") == 0) modelcombo = WEIGHTAVG;
        else if (strcmp(optarg, "best") == 0) modelcombo = BEST;
        else if (strcmp(optarg, "separate") == 0) modelcombo = SEPARATE;
        cout << "modelcombo: " << comboNames[modelcombo] << endl;
        break;
      }

    case 'n':
      nmodels = std::atoi(optarg);
      cout << "nmodels: " << nmodels << endl;
      break;

    case 't':
      reltrans = true;
      cout << "reltrans: " << reltrans << endl;
      break;

    case 'b':
      reltrans = false;
      cout << "reltrans: " << reltrans << endl;
      break;

    case 's':
      seed = std::atoi(optarg);
      cout << "seed: " << seed << endl;
      break;

    case 'q':
      // already processed this one
      cout << "agent: " << agentType << endl;
      break;
    
    case 'd':
      PRINTS = true;
      break;

    case 'w':
      nstates = std::atoi(optarg);
      cout << "nstates for discretization: " << nstates << endl;
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
  out_rl_action = node.advertise<rl_msgs::RLAction>("rl_agent/rl_action",qDepth, false);
  out_exp_info = node.advertise<rl_msgs::RLExperimentInfo>("rl_agent/rl_experiment_info",qDepth, false);

  // Set up subscribers
  ros::TransportHints noDelay = ros::TransportHints().tcpNoDelay(true);

  ros::Subscriber rl_description =  node.subscribe("rl_env/rl_env_description", qDepth, processEnvDescription, noDelay);
  ros::Subscriber rl_state =  node.subscribe("rl_env/rl_state_reward", qDepth, processState, noDelay);
  ros::Subscriber rl_seed =  node.subscribe("rl_env/rl_seed", 20, processSeed, noDelay);

  ROS_INFO(NODE ": starting main loop");

  ros::spin();                          // handle incoming data
  //while (ros::ok()){
  //  ros::getGlobalCallbackQueue()->callAvailable(ros::WallDuration(0.1));
  //}



  return 0;
}



