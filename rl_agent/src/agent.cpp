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
#include <rl_agent/Sarsa.hh>

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
int modelcombo = BEST;
int planner = PAR_ETUCT_ACTUAL;
int nmodels = 1;
bool reltrans = true;
int nstates = 0;
int k = 1000;
char *filename = NULL;
int history = 0;
float v = 0;
float n = 0;
// possibly over-written by command line arguments


void displayHelp(){
  cout << "\n Call agent --agent type [options]\n";
  cout << "Agent types: qlearner sarsa modelbased rmax texplore dyna savedpolicy\n";
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
  cout << "--history value (# steps of history to use for planning with delay)\n";
  cout << "--filename file (file to load saved policy from for savedpolicy agent)\n";
  cout << "--model type (tabular,tree,m5tree)\n";
  cout << "--planner type (vi,pi,sweeping,uct,parallel-uct,delayed-uct,delayed-parallel-uct)\n";
  cout << "--explore type (unknown,greedy,epsilongreedy,variancenovelty)\n";
  cout << "--combo type (average,best,separate)\n";
  cout << "--nmodels value (# of models)\n";
  cout << "--nstates value (optionally discretize domain into value # of states on each feature)\n";
  cout << "--reltrans (learn relative transitions)\n";
  cout << "--abstrans (learn absolute transitions)\n";
  cout << "--v value (For TEXPLORE: b/v coefficient for rewarding state-actions where models disagree)\n";
  cout << "--n value (For TEXPLORE: n coefficient for rewarding state-actions which are novel)\n";
  cout << "--prints (turn on debug printing of actions/rewards)\n";

  cout << "\n For more info, see: http://www.ros.org/wiki/rl_agent\n";
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
    info.number_actions = 1;
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
      info.number_actions++;
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
                                history, v, n, false, reltrans, 0.2,
                                envIn->stochastic, envIn->episodic,
                                rng);

  }

  else if (strcmp(agentType, "dyna") == 0){
    cout << "Agent: Dyna" << endl;
    agent = new Dyna(envIn->num_actions, discountfactor,
                     initialvalue, alpha, k, epsilon,
                     rng);
  }

  else if (strcmp(agentType, "sarsa") == 0){
    cout << "Agent: Sarsa" << endl;
    agent = new Sarsa(envIn->num_actions, discountfactor,
                      initialvalue, alpha, epsilon, lambda,
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
    explore = DIFF_AND_NOVEL_BONUS;
    v = 0;
    n = 0;
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
    {"nmodels", 1, 0, '#'},
    {"reltrans", 0, 0, 't'},
    {"abstrans", 0, 0, '0'},
    {"seed", 1, 0, 's'},
    {"agent", 1, 0, 'q'},
    {"prints", 0, 0, 'd'},
    {"nstates", 1, 0, 'w'},
    {"k", 1, 0, 'k'},
    {"filename", 1, 0, 'f'},
    {"history", 1, 0, 'y'},
    {"b", 1, 0, 'b'},
    {"v", 1, 0, 'v'},
    {"n", 1, 0, 'n'}
  };

  bool epsilonChanged = false;
  bool actrateChanged = false;
  bool mChanged = false;
  bool bvnChanged = false;
  bool lambdaChanged = false;

  while(-1 != (ch = getopt_long_only(argc, argv, optflags, long_options, &option_index))) {
    switch(ch) {

    case 'g':
      discountfactor = std::atof(optarg);
      cout << "discountfactor: " << discountfactor << endl;
      break;

    case 'e':
      epsilonChanged = true;
      epsilon = std::atof(optarg);
      cout << "epsilon: " << epsilon << endl;
      break;

    case 'y':
      {
        if (strcmp(agentType, "texplore") == 0 || strcmp(agentType, "modelbased") == 0){
          history = std::atoi(optarg);
          cout << "history: " << history << endl;
        } else {
          cout << "--history is not a valid option for agent: " << agentType << endl;
          exit(-1);
        }
        break;
      }

    case 'k':
      {
        if (strcmp(agentType, "dyna") == 0){
          k = std::atoi(optarg);
          cout << "k: " << k << endl;
        } else {
          cout << "--k is only a valid option for the Dyna agent" << endl;
          exit(-1);
        }
        break;
      }

    case 'f':
      filename = optarg;
      cout << "policy filename: " <<  filename << endl;
      break;

    case 'a':
      {
        if (strcmp(agentType, "qlearner") == 0 || strcmp(agentType, "dyna") == 0 || strcmp(agentType, "sarsa") == 0){
          alpha = std::atof(optarg);
          cout << "alpha: " << alpha << endl;
        } else {
          cout << "--alpha option is only valid for Q-Learning, Dyna, and Sarsa" << endl;
          exit(-1);
        }
        break;
      }

    case 'i':
      {
        if (strcmp(agentType, "qlearner") == 0 || strcmp(agentType, "dyna") == 0 || strcmp(agentType, "sarsa") == 0){
          initialvalue = std::atof(optarg);
          cout << "initialvalue: " << initialvalue << endl;
        } else {
          cout << "--initialvalue option is only valid for Q-Learning, Dyna, and Sarsa" << endl;
          exit(-1);
        }
        break;
      }

    case 'r':
      {
        actrateChanged = true;
        if (strcmp(agentType, "texplore") == 0 || strcmp(agentType, "modelbased") == 0 || strcmp(agentType, "rmax") == 0){
          actrate = std::atof(optarg);
          cout << "actrate: " << actrate << endl;
        } else {
          cout << "Model-free methods do not require an action rate" << endl;
          exit(-1);
        }
        break;
      }

    case 'l':
      {
        lambdaChanged = true;
        if (strcmp(agentType, "texplore") == 0 || strcmp(agentType, "modelbased") == 0 || strcmp(agentType, "rmax") == 0 || strcmp(agentType, "sarsa") == 0){
          lambda = std::atof(optarg);
          cout << "lambda: " << lambda << endl;
        } else {
          cout << "--lambda option is invalid for this agent: " << agentType << endl;
          exit(-1);
        }
        break;
      }

    case 'm':
      {
        mChanged = true;
        if (strcmp(agentType, "texplore") == 0 || strcmp(agentType, "modelbased") == 0 || strcmp(agentType, "rmax") == 0){
          M = std::atoi(optarg);
          cout << "M: " << M << endl;
        } else {
          cout << "--M option only useful for model-based agents, not " << agentType << endl;
          exit(-1);
        }
        break;
      }

    case 'o':
      {
        if (strcmp(agentType, "texplore") == 0 || strcmp(agentType, "modelbased") == 0 || strcmp(agentType, "rmax") == 0){
          if (strcmp(optarg, "tabular") == 0) model = RMAX;
          else if (strcmp(optarg, "tree") == 0) model = C45TREE;
          else if (strcmp(optarg, "texplore") == 0) model = C45TREE;
          else if (strcmp(optarg, "c45tree") == 0) model = C45TREE;
          else if (strcmp(optarg, "m5tree") == 0) model = M5ALLMULTI;
          if (strcmp(agentType, "rmax") == 0 && model != RMAX){
            cout << "R-Max should use tabular model" << endl;
            exit(-1);
          }
        } else {
          cout << "Model-free methods do not need a model, --model option does nothing for this agent type" << endl;
          exit(-1);
        }
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
        else if (strcmp(optarg, "variancenovelty") == 0) explore = DIFF_AND_NOVEL_BONUS;
        if (strcmp(agentType, "rmax") == 0 && explore != EXPLORE_UNKNOWN){
          cout << "R-Max should use \"--explore unknown\" exploration" << endl;
          exit(-1);
        }
        else if (strcmp(agentType, "texplore") != 0 && strcmp(agentType, "modelbased") != 0 && strcmp(agentType, "rmax") != 0 && (explore != GREEDY && explore != EPSILONGREEDY)) {
          cout << "Model free methods must use either greedy or epsilon-greedy exploration!" << endl;
          explore = EPSILONGREEDY;
          exit(-1);
        }
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
        if (strcmp(agentType, "texplore") != 0 && strcmp(agentType, "modelbased") != 0 && strcmp(agentType, "rmax") != 0){
          cout << "Model-free methods do not require planners, --planner option does nothing with this agent" << endl;
          exit(-1);
        }
        if (strcmp(agentType, "rmax") == 0 && planner != VALUE_ITERATION){
          cout << "Typical implementation of R-Max would use value iteration, but another planner type is ok" << endl;
        }
        cout << "planner: " << plannerNames[planner] << endl;
        break;
      }

    case 'c':
      {
        if (strcmp(agentType, "texplore") == 0 || strcmp(agentType, "modelbased") == 0){
          if (strcmp(optarg, "average") == 0) modelcombo = AVERAGE;
          else if (strcmp(optarg, "weighted") == 0) modelcombo = WEIGHTAVG;
          else if (strcmp(optarg, "best") == 0) modelcombo = BEST;
          else if (strcmp(optarg, "separate") == 0) modelcombo = SEPARATE;
          cout << "modelcombo: " << comboNames[modelcombo] << endl;
        } else {
          cout << "--combo is an invalid option for agent: " << agentType << endl;
          exit(-1);
        }
        break;
      }

    case '#':
      {
        if (strcmp(agentType, "texplore") == 0 || strcmp(agentType, "modelbased") == 0){
          nmodels = std::atoi(optarg);
          cout << "nmodels: " << nmodels << endl;
        } else {
          cout << "--nmodels is an invalid option for agent: " << agentType << endl;
          exit(-1);
        }
        if (nmodels < 1){
          cout << "nmodels must be > 0" << endl;
          exit(-1);
        }
        break;
      }

    case 't':
      {
        if (strcmp(agentType, "texplore") == 0 || strcmp(agentType, "modelbased") == 0){
          reltrans = true;
          cout << "reltrans: " << reltrans << endl;
        } else {
          cout << "--reltrans is an invalid option for agent: " << agentType << endl;
          exit(-1);
        }
        break;
      }

    case '0':
      {
        if (strcmp(agentType, "texplore") == 0 || strcmp(agentType, "modelbased") == 0){
          reltrans = false;
          cout << "reltrans: " << reltrans << endl;
        } else {
          cout << "--abstrans is an invalid option for agent: " << agentType << endl;
          exit(-1);
        }
        break;
      }

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

    case 'v':
    case 'b':
      {
        bvnChanged = true;
        if (strcmp(agentType, "texplore") == 0){
          v = std::atof(optarg);
          cout << "v coefficient (variance bonus): " << v << endl;
        }
        else {
          cout << "--v and --b are invalid options for agent: " << agentType << endl;
          exit(-1);
        }
        break;
      }

    case 'n':
      {
        bvnChanged = true;
        if (strcmp(agentType, "texplore") == 0){
          n = std::atof(optarg);
          cout << "n coefficient (novelty bonus): " << n << endl;
        }
        else {
          cout << "--n is an invalid option for agent: " << agentType << endl;
          exit(-1);
        }
        break;
      }

    case 'h':
    case '?':
    case 0:
    default:
      displayHelp();
      break;
    }
  }

  // default back to greedy if no coefficients
  if (explore == DIFF_AND_NOVEL_BONUS && v == 0 && n == 0)
    explore = GREEDY;

  // check for conflicting options
  // changed epsilon but not doing epsilon greedy exploration
  if (epsilonChanged && explore != EPSILONGREEDY){
    cout << "No reason to change epsilon when not using epsilon-greedy exploration" << endl;
    exit(-1);
  }

  // set history value but not doing uct w/history planner
  if (history > 0 && (planner == VALUE_ITERATION || planner == POLICY_ITERATION || planner == PRI_SWEEPING)){
    cout << "No reason to set history higher than 0 if not using a UCT planner" << endl;
    exit(-1);
  }

  // set action rate but not doing real-time planner
  if (actrateChanged && (planner == VALUE_ITERATION || planner == POLICY_ITERATION || planner == PRI_SWEEPING)){
    cout << "No reason to set actrate if not using a UCT planner" << endl;
    exit(-1);
  }

  // set lambda but not doing uct (lambda)
  if (lambdaChanged && (strcmp(agentType, "texplore") == 0 || strcmp(agentType, "modelbased") == 0 || strcmp(agentType, "rmax") == 0) && (planner == VALUE_ITERATION || planner == POLICY_ITERATION || planner == PRI_SWEEPING)){
    cout << "No reason to set actrate if not using a UCT planner" << endl;
    exit(-1);
  }

  // set n/v/b but not doing that diff_novel exploration
  if (bvnChanged && explore != DIFF_AND_NOVEL_BONUS){
    cout << "No reason to set n or v if not doing variance & novelty exploration" << endl;
    exit(-1);
  }

  // set combo other than best but only doing 1 model
  if (modelcombo != BEST && nmodels == 1){
    cout << "No reason to have model combo other than best with nmodels = 1" << endl;
    exit(-1);
  }

  // set M but not doing explore unknown
  if (mChanged && explore != EXPLORE_UNKNOWN){
    cout << "No reason to set M if not doing R-max style Explore Unknown exploration" << endl;
    exit(-1);
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

  if(agent != NULL) {
    if(filename != NULL) {
      agent->savePolicy(filename);
    }
  }

  return 0;
}



