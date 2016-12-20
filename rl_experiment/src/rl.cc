/** \file Main file that starts agents and environments
    \author Todd Hester
*/

#include <rl_common/Random.h>
#include <rl_common/core.hh>

#include <stdio.h>
#include <string.h>
#include <sys/time.h>

//////////////////
// Environments //
//////////////////
#include <rl_env/RobotCarVel.hh>
#include <rl_env/fourrooms.hh>
#include <rl_env/tworooms.hh>
#include <rl_env/taxi.hh>
#include <rl_env/FuelRooms.hh>
#include <rl_env/stocks.hh>
#include <rl_env/energyrooms.hh>
#include <rl_env/MountainCar.hh>
#include <rl_env/CartPole.hh>
#include <rl_env/LightWorld.hh>


////////////
// Agents //
////////////
#include <rl_agent/QLearner.hh>
#include <rl_agent/ModelBasedAgent.hh>
#include <rl_agent/DiscretizationAgent.hh>
#include <rl_agent/SavedPolicy.hh>
#include <rl_agent/Dyna.hh>
#include <rl_agent/Sarsa.hh>



#include <vector>
#include <sstream>
#include <iostream>

#include <getopt.h>
#include <stdlib.h>

unsigned NUMEPISODES = 1000; //10; //200; //500; //200;
const unsigned NUMTRIALS = 1; //30; //30; //5; //30; //30; //50
unsigned MAXSTEPS = 1000; // per episode
bool PRINTS = false;


void displayHelp(){
  cout << "\n Call experiment --agent type --env type [options]\n";
  cout << "Agent types: qlearner sarsa modelbased rmax texplore dyna savedpolicy\n";
  cout << "Env types: taxi tworooms fourrooms energy fuelworld mcar cartpole car2to7 car7to2 carrandom stocks lightworld\n";

  cout << "\n Agent Options:\n";
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

  cout << "\n Env Options:\n";
  cout << "--deterministic (deterministic version of domain)\n";
  cout << "--stochastic (stochastic version of domain)\n";
  cout << "--delay value (# steps of action delay (for mcar and tworooms)\n";
  cout << "--lag (turn on brake lag for car driving domain)\n";
  cout << "--highvar (have variation fuel costs in Fuel World)\n";
  cout << "--nsectors value (# sectors for stocks domain)\n";
  cout << "--nstocks value (# stocks for stocks domain)\n";

  cout << "\n--prints (turn on debug printing of actions/rewards)\n";
  cout << "--nepisodes value (# of episodes to run (1000 default)\n";
  cout << "--seed value (integer seed for random number generator)\n";

  cout << "\n For more info, see: http://www.ros.org/wiki/rl_experiment\n";

  exit(-1);

}


int main(int argc, char **argv) {

  // default params for env and agent
  char* agentType = NULL;
  char* envType = NULL;
  float discountfactor = 0.99;
  float epsilon = 0.1;
  float alpha = 0.3;
  float initialvalue = 0.0;
  float actrate = 10.0;
  float lambda = 0.1;
  int M = 5;
  int modelType = C45TREE;
  int exploreType = GREEDY;
  int predType = BEST;
  int plannerType = PAR_ETUCT_ACTUAL;
  int nmodels = 1;
  bool reltrans = true;
  bool deptrans = false;
  float v = 0;
  float n = 0;
  float featPct = 0.2;
  int nstates = 0;
  int k = 1000;
  char *filename = NULL;
  bool stochastic = true;
  int nstocks = 3;
  int nsectors = 3;
  int delay = 0;
  bool lag = false;
  bool highvar = false;
  int history = 0;
  int seed = 1;
  // change some of these parameters based on command line args

  // parse agent type
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
    modelType = RMAX;
    exploreType = EXPLORE_UNKNOWN;
    predType = BEST;
    plannerType = VALUE_ITERATION;
    nmodels = 1;
    reltrans = false;
    M = 5;
    history = 0;
  } else if (strcmp(agentType, "texplore") == 0){
    modelType = C45TREE;
    exploreType = DIFF_AND_NOVEL_BONUS;
    v = 0;
    n = 0;
    predType = AVERAGE;
    plannerType = PAR_ETUCT_ACTUAL;
    nmodels = 5;
    reltrans = true;
    M = 0;
    history = 0;
  }

  // parse env type
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

  // parse other arguments
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
    {"n", 1, 0, 'n'},

    {"env", 1, 0, 1},
    {"deterministic", 0, 0, 2},
    {"stochastic", 0, 0, 3},
    {"delay", 1, 0, 4},
    {"nsectors", 1, 0, 5},
    {"nstocks", 1, 0, 6},
    {"lag", 0, 0, 7},
    {"nolag", 0, 0, 8},
    {"highvar", 0, 0, 11},
    {"nepisodes", 1, 0, 12}

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
          if (strcmp(optarg, "tabular") == 0) modelType = RMAX;
          else if (strcmp(optarg, "tree") == 0) modelType = C45TREE;
          else if (strcmp(optarg, "texplore") == 0) modelType = C45TREE;
          else if (strcmp(optarg, "c45tree") == 0) modelType = C45TREE;
          else if (strcmp(optarg, "m5tree") == 0) modelType = M5ALLMULTI;
          if (strcmp(agentType, "rmax") == 0 && modelType != RMAX){
            cout << "R-Max should use tabular model" << endl;
            exit(-1);
          }
        } else {
          cout << "Model-free methods do not need a model, --model option does nothing for this agent type" << endl;
          exit(-1);
        }
        cout << "model: " << modelNames[modelType] << endl;
        break;
      }

    case 'x':
      {
        if (strcmp(optarg, "unknown") == 0) exploreType = EXPLORE_UNKNOWN;
        else if (strcmp(optarg, "greedy") == 0) exploreType = GREEDY;
        else if (strcmp(optarg, "epsilongreedy") == 0) exploreType = EPSILONGREEDY;
        else if (strcmp(optarg, "unvisitedstates") == 0) exploreType = UNVISITED_BONUS;
        else if (strcmp(optarg, "unvisitedactions") == 0) exploreType = UNVISITED_ACT_BONUS;
        else if (strcmp(optarg, "variancenovelty") == 0) exploreType = DIFF_AND_NOVEL_BONUS;
        if (strcmp(agentType, "rmax") == 0 && exploreType != EXPLORE_UNKNOWN){
          cout << "R-Max should use \"--explore unknown\" exploration" << endl;
          exit(-1);
        }
        else if (strcmp(agentType, "texplore") != 0 && strcmp(agentType, "modelbased") != 0 && strcmp(agentType, "rmax") != 0 && (exploreType != GREEDY && exploreType != EPSILONGREEDY)) {
          cout << "Model free methods must use either greedy or epsilon-greedy exploration!" << endl;
          exploreType = EPSILONGREEDY;
          exit(-1);
        }
        cout << "explore: " << exploreNames[exploreType] << endl;
        break;
      }

    case 'p':
      {
        if (strcmp(optarg, "vi") == 0) plannerType = VALUE_ITERATION;
        else if (strcmp(optarg, "valueiteration") == 0) plannerType = VALUE_ITERATION;
        else if (strcmp(optarg, "policyiteration") == 0) plannerType = POLICY_ITERATION;
        else if (strcmp(optarg, "pi") == 0) plannerType = POLICY_ITERATION;
        else if (strcmp(optarg, "sweeping") == 0) plannerType = PRI_SWEEPING;
        else if (strcmp(optarg, "prioritizedsweeping") == 0) plannerType = PRI_SWEEPING;
        else if (strcmp(optarg, "uct") == 0) plannerType = ET_UCT_ACTUAL;
        else if (strcmp(optarg, "paralleluct") == 0) plannerType = PAR_ETUCT_ACTUAL;
        else if (strcmp(optarg, "realtimeuct") == 0) plannerType = PAR_ETUCT_ACTUAL;
        else if (strcmp(optarg, "realtime-uct") == 0) plannerType = PAR_ETUCT_ACTUAL;
        else if (strcmp(optarg, "parallel-uct") == 0) plannerType = PAR_ETUCT_ACTUAL;
        else if (strcmp(optarg, "delayeduct") == 0) plannerType = POMDP_ETUCT;
        else if (strcmp(optarg, "delayed-uct") == 0) plannerType = POMDP_ETUCT;
        else if (strcmp(optarg, "delayedparalleluct") == 0) plannerType = POMDP_PAR_ETUCT;
        else if (strcmp(optarg, "delayed-parallel-uct") == 0) plannerType = POMDP_PAR_ETUCT;
        if (strcmp(agentType, "texplore") != 0 && strcmp(agentType, "modelbased") != 0 && strcmp(agentType, "rmax") != 0){
          cout << "Model-free methods do not require planners, --planner option does nothing with this agent" << endl;
          exit(-1);
        }
        if (strcmp(agentType, "rmax") == 0 && plannerType != VALUE_ITERATION){
          cout << "Typical implementation of R-Max would use value iteration, but another planner type is ok" << endl;
        }
        cout << "planner: " << plannerNames[plannerType] << endl;
        break;
      }

    case 'c':
      {
        if (strcmp(agentType, "texplore") == 0 || strcmp(agentType, "modelbased") == 0){
          if (strcmp(optarg, "average") == 0) predType = AVERAGE;
          else if (strcmp(optarg, "weighted") == 0) predType = WEIGHTAVG;
          else if (strcmp(optarg, "best") == 0) predType = BEST;
          else if (strcmp(optarg, "separate") == 0) predType = SEPARATE;
          cout << "predType: " << comboNames[predType] << endl;
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

    case 2:
      stochastic = false;
      cout << "stochastic: " << stochastic << endl;
      break;

    case 11:
      {
        if (strcmp(envType, "fuelworld") == 0){
          highvar = true;
          cout << "fuel world fuel cost variation: " << highvar << endl;
        } else {
          cout << "--highvar is only a valid option for the fuelworld domain." << endl;
          exit(-1);
        }
        break;
      }

    case 3:
      stochastic = true;
      cout << "stochastic: " << stochastic << endl;
      break;

    case 4:
      {
        if (strcmp(envType, "mcar") == 0 || strcmp(envType, "tworooms") == 0){
          delay = std::atoi(optarg);
          cout << "delay steps: " << delay << endl;
        } else {
          cout << "--delay option is only valid for the mcar and tworooms domains" << endl;
          exit(-1);
        }
        break;
      }

    case 5:
      {
        if (strcmp(envType, "stocks") == 0){
          nsectors = std::atoi(optarg);
          cout << "nsectors: " << nsectors << endl;
        } else {
          cout << "--nsectors option is only valid for the stocks domain" << endl;
          exit(-1);
        }
        break;
      }

    case 6:
      {
        if (strcmp(envType, "stocks") == 0){
          nstocks = std::atoi(optarg);
          cout << "nstocks: " << nstocks << endl;
        } else {
          cout << "--nstocks option is only valid for the stocks domain" << endl;
          exit(-1);
        }
        break;
      }

    case 7:
      {
        if (strcmp(envType, "car2to7") == 0 || strcmp(envType, "car7to2") == 0 || strcmp(envType, "carrandom") == 0){
          lag = true;
          cout << "lag: " << lag << endl;
        } else {
          cout << "--lag option is only valid for car velocity tasks" << endl;
          exit(-1);
        }
        break;
      }

    case 8:
      {
        if (strcmp(envType, "car2to7") == 0 || strcmp(envType, "car7to2") == 0 || strcmp(envType, "carrandom") == 0){
          lag = false;
          cout << "lag: " << lag << endl;
        } else {
          cout << "--nolag option is only valid for car velocity tasks" << endl;
          exit(-1);
        }
        break;
      }

    case 1:
      // already processed this one
      cout << "env: " << envType << endl;
      break;

    case 12:
      NUMEPISODES = std::atoi(optarg);
      cout << "Num Episodes: " << NUMEPISODES << endl;
      break;

    case 'h':
    case '?':
    case 0:
    default:
      displayHelp();
      break;
    }
  }

  // default back to greedy if no coefficients
  if (exploreType == DIFF_AND_NOVEL_BONUS && v == 0 && n == 0)
    exploreType = GREEDY;

  // check for conflicting options
  // changed epsilon but not doing epsilon greedy exploration
  if (epsilonChanged && exploreType != EPSILONGREEDY){
    cout << "No reason to change epsilon when not using epsilon-greedy exploration" << endl;
    exit(-1);
  }

  // set history value but not doing uct w/history planner
  if (history > 0 && (plannerType == VALUE_ITERATION || plannerType == POLICY_ITERATION || plannerType == PRI_SWEEPING)){
    cout << "No reason to set history higher than 0 if not using a UCT planner" << endl;
    exit(-1);
  }

  // set action rate but not doing real-time planner
  if (actrateChanged && (plannerType == VALUE_ITERATION || plannerType == POLICY_ITERATION || plannerType == PRI_SWEEPING)){
    cout << "No reason to set actrate if not using a UCT planner" << endl;
    exit(-1);
  }

  // set lambda but not doing uct (lambda)
  if (lambdaChanged && (strcmp(agentType, "texplore") == 0 || strcmp(agentType, "modelbased") == 0 || strcmp(agentType, "rmax") == 0) && (plannerType == VALUE_ITERATION || plannerType == POLICY_ITERATION || plannerType == PRI_SWEEPING)){
    cout << "No reason to set actrate if not using a UCT planner" << endl;
    exit(-1);
  }

  // set n/v/b but not doing that diff_novel exploration
  if (bvnChanged && exploreType != DIFF_AND_NOVEL_BONUS){
    cout << "No reason to set n or v if not doing variance & novelty exploration" << endl;
    exit(-1);
  }

  // set combo other than best but only doing 1 model
  if (predType != BEST && nmodels == 1){
    cout << "No reason to have model combo other than best with nmodels = 1" << endl;
    exit(-1);
  }

  // set M but not doing explore unknown
  if (mChanged && exploreType != EXPLORE_UNKNOWN){
    cout << "No reason to set M if not doing R-max style Explore Unknown exploration" << endl;
    exit(-1);
  }

  if (PRINTS){
    if (stochastic)
      cout << "Stohastic\n";
    else
      cout << "Deterministic\n";
  }

  Random rng(1 + seed);

  std::vector<int> statesPerDim;

  // Construct environment here.
  Environment* e;

  if (strcmp(envType, "cartpole") == 0){
    if (PRINTS) cout << "Environment: Cart Pole\n";
    e = new CartPole(rng, stochastic);
  }

  else if (strcmp(envType, "mcar") == 0){
    if (PRINTS) cout << "Environment: Mountain Car\n";
    e = new MountainCar(rng, stochastic, false, delay);
  }

  // taxi
  else if (strcmp(envType, "taxi") == 0){
    if (PRINTS) cout << "Environment: Taxi\n";
    e = new Taxi(rng, stochastic);
  }

  // Light World
  else if (strcmp(envType, "lightworld") == 0){
    if (PRINTS) cout << "Environment: Light World\n";
    e = new LightWorld(rng, stochastic, 4);
  }

  // two rooms
  else if (strcmp(envType, "tworooms") == 0){
    if (PRINTS) cout << "Environment: TwoRooms\n";
    e = new TwoRooms(rng, stochastic, true, delay, false);
  }

  // car vel, 2 to 7
  else if (strcmp(envType, "car2to7") == 0){
    if (PRINTS) cout << "Environment: Car Velocity 2 to 7 m/s\n";
    e = new RobotCarVel(rng, false, true, false, lag);
    statesPerDim.resize(4,0);
    statesPerDim[0] = 12;
    statesPerDim[1] = 120;
    statesPerDim[2] = 4;
    statesPerDim[3] = 10;
    MAXSTEPS = 100;
  }
  // car vel, 7 to 2
  else if (strcmp(envType, "car7to2") == 0){
    if (PRINTS) cout << "Environment: Car Velocity 7 to 2 m/s\n";
    e = new RobotCarVel(rng, false, false, false, lag);
    statesPerDim.resize(4,0);
    statesPerDim[0] = 12;
    statesPerDim[1] = 120;
    statesPerDim[2] = 4;
    statesPerDim[3] = 10;
    MAXSTEPS = 100;
  }
  // car vel, random vels
  else if (strcmp(envType, "carrandom") == 0){
    if (PRINTS) cout << "Environment: Car Velocity Random Velocities\n";
    e = new RobotCarVel(rng, true, false, false, lag);
    statesPerDim.resize(4,0);
    statesPerDim[0] = 12;
    statesPerDim[1] = 48;
    statesPerDim[2] = 4;
    statesPerDim[3] = 10;
    MAXSTEPS = 100;
  }

  // four rooms
  else if (strcmp(envType, "fourrooms") == 0){
    if (PRINTS) cout << "Environment: FourRooms\n";
    e = new FourRooms(rng, stochastic, true, false);
  }

  // four rooms with energy level
  else if (strcmp(envType, "energy") == 0){
    if (PRINTS) cout << "Environment: EnergyRooms\n";
    e = new EnergyRooms(rng, stochastic, true, false);
  }

  // gridworld with fuel (fuel stations on top and bottom with random costs)
  else if (strcmp(envType, "fuelworld") == 0){
    if (PRINTS) cout << "Environment: FuelWorld\n";
    e = new FuelRooms(rng, highvar, stochastic);
  }

  // stocks
  else if (strcmp(envType, "stocks") == 0){
    if (PRINTS) cout << "Enironment: Stocks with " << nsectors
                     << " sectors and " << nstocks << " stocks\n";
    e = new Stocks(rng, stochastic, nsectors, nstocks);
  }

  else {
    std::cerr << "Invalid env type" << endl;
    exit(-1);
  }

  const int numactions = e->getNumActions(); // Most agents will need this?

  std::vector<float> minValues;
  std::vector<float> maxValues;
  e->getMinMaxFeatures(&minValues, &maxValues);
  bool episodic = e->isEpisodic();

  cout << "Environment is ";
  if (!episodic) cout << "NOT ";
  cout << "episodic." << endl;

  // lets just check this for now
  for (unsigned i = 0; i < minValues.size(); i++){
    if (PRINTS) cout << "Feat " << i << " min: " << minValues[i]
                     << " max: " << maxValues[i] << endl;
  }

  // get max/min reward for the domain
  float rMax = 0.0;
  float rMin = -1.0;

  e->getMinMaxReward(&rMin, &rMax);
  float rRange = rMax - rMin;
  if (PRINTS) cout << "Min Reward: " << rMin
                   << ", Max Reward: " << rMax << endl;

  // set rmax as a bonus for certain exploration types
  if (rMax <= 0.0 && (exploreType == TWO_MODE_PLUS_R ||
                      exploreType == CONTINUOUS_BONUS_R ||
                      exploreType == CONTINUOUS_BONUS ||
                      exploreType == THRESHOLD_BONUS_R)){
    rMax = 1.0;
  }


  float rsum = 0;

  if (statesPerDim.size() == 0){
    cout << "set statesPerDim to " << nstates << " for all dim" << endl;
    statesPerDim.resize(minValues.size(), nstates);
  }

  for (unsigned j = 0; j < NUMTRIALS; ++j) {

    // Construct agent here.
    Agent* agent;

    if (strcmp(agentType, "qlearner") == 0){
      if (PRINTS) cout << "Agent: QLearner" << endl;
      agent = new QLearner(numactions,
                           discountfactor,
                           initialvalue, //0.0, // initialvalue
                           alpha, // alpha
                           epsilon, // epsilon
                           rng);
    }

    else if (strcmp(agentType, "dyna") == 0){
      if (PRINTS) cout << "Agent: Dyna" << endl;
      agent = new Dyna(numactions,
                       discountfactor,
                       initialvalue, //0.0, // initialvalue
                       alpha, // alpha
                       k, // k
                       epsilon, // epsilon
                       rng);
    }

    else if (strcmp(agentType, "sarsa") == 0){
      if (PRINTS) cout << "Agent: SARSA" << endl;
      agent = new Sarsa(numactions,
                        discountfactor,
                        initialvalue, //0.0, // initialvalue
                        alpha, // alpha
                        epsilon, // epsilon
                        lambda,
                        rng);
    }

    else if (strcmp(agentType, "modelbased") == 0 || strcmp(agentType, "rmax") || strcmp(agentType, "texplore")){
      if (PRINTS) cout << "Agent: Model Based" << endl;
      agent = new ModelBasedAgent(numactions,
                                  discountfactor,
                                  rMax, rRange,
                                  modelType,
                                  exploreType,
                                  predType,
                                  nmodels,
                                  plannerType,
                                  epsilon, // epsilon
                                  lambda,
                                  (1.0/actrate), //0.1, //0.1, //0.01, // max time
                                  M,
                                  minValues, maxValues,
                                  statesPerDim,//0,
                                  history, v, n,
                                  deptrans, reltrans, featPct, stochastic, episodic,
                                  rng);
    }

    else if (strcmp(agentType, "savedpolicy") == 0){
      if (PRINTS) cout << "Agent: Saved Policy" << endl;
      agent = new SavedPolicy(numactions,filename);
    }

    else {
      std::cerr << "ERROR: Invalid agent type" << endl;
      exit(-1);
    }

    // start discrete agent if we're discretizing (if nstates > 0 and not agent type 'c')
    int totalStates = 1;
    Agent* a2 = agent;
    // not for model based when doing continuous model
    if (nstates > 0 && (modelType != M5ALLMULTI || strcmp(agentType, "qlearner") == 0)){
      int totalStates = powf(nstates,minValues.size());
      if (PRINTS) cout << "Discretize with " << nstates << ", total: " << totalStates << endl;
      agent = new DiscretizationAgent(nstates, a2,
                                      minValues, maxValues, PRINTS);
    }
    else {
      totalStates = 1;
      for (unsigned i = 0; i < minValues.size(); i++){
        int range = 1+maxValues[i] - minValues[i];
        totalStates *= range;
      }
      if (PRINTS) cout << "No discretization, total: " << totalStates << endl;
    }

    // before we start, seed the agent with some experiences
    agent->seedExp(e->getSeedings());

    // STEP BY STEP DOMAIN
    if (!episodic){

      // performance tracking
      float sum = 0;
      int steps = 0;
      float trialSum = 0.0;

      int a = 0;
      float r = 0;

      //////////////////////////////////
      // non-episodic
      //////////////////////////////////
      for (unsigned i = 0; i < NUMEPISODES; ++i){

        std::vector<float> es = e->sensation();

        // first step
        if (i == 0){

          // first action
          a = agent->first_action(es);
          r = e->apply(a);

        } else {
          // next action
          a = agent->next_action(r, es);
          r = e->apply(a);
        }

        // update performance
        sum += r;
        ++steps;

        std::cerr << r << endl;

      }
      ///////////////////////////////////

      rsum += sum;
      trialSum += sum;
      if (PRINTS) cout << "Rsum(trial " << j << "): " << trialSum << " Avg: "
                       << (rsum / (float)(j+1))<< endl;

    }

    // EPISODIC DOMAINS
    else {

      //////////////////////////////////
      // episodic
      //////////////////////////////////
      for (unsigned i = 0; i < NUMEPISODES; ++i) {

        // performance tracking
        float sum = 0;
        int steps = 0;

        // first action
        std::vector<float> es = e->sensation();
        int a = agent->first_action(es);
        float r = e->apply(a);

        // update performance
        sum += r;
        ++steps;

        while (!e->terminal() && steps < MAXSTEPS) {

          // perform an action
          es = e->sensation();
          a = agent->next_action(r, es);
          r = e->apply(a);

          // update performance info
          sum += r;
          ++steps;

        }

        // terminal/last state
        if (e->terminal()){
          agent->last_action(r);
        }else{
          agent->next_action(r, e->sensation());
        }

        e->reset();
        std::cerr << sum << endl;

        rsum += sum;

      }

    }

    if (NUMTRIALS > 1) delete agent;

  }

  if (PRINTS) cout << "Avg Rsum: " << (rsum / (float)NUMTRIALS) << endl;

} // end main

