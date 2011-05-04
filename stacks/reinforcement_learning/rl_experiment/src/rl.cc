/** \file Main file that starts agents and environments
    \author Todd Hester
*/

#include "../Common/Random.h"
#include "../Common/core.hh"

#include <stdio.h>
#include <string.h>
#include <sys/time.h>

//////////////////
// Environments //
//////////////////
//#include "../Env/penaltykick.hh"
#include "../Env/lights.hh"
#include "../Env/taxi.hh"
#include "../Env/tworooms.hh"
#include "../Env/wmaze.hh"
//#include "../Env/oneroom.hh"
#include "../Env/fourrooms.hh"
//#include "../Env/TrapRoom.hh"
//#include "../Env/big4rooms.hh"
#include "../Env/playroom.hh"
//#include "../Env/MultiRoom.hh"
#include "../Env/energyrooms.hh"
//#include "../Env/LargeDomain.hh"
#include "../Env/Castle.hh"
#include "../Env/redherring.hh"
#include "../Env/SmallCastle.hh"
#include "../Env/stocks.hh"
#include "../Env/CastleR.hh"
//#include "../Env/RisingOptimum.hh"
#include "../Env/FuelRooms.hh"
#include "../Env/Explore.hh"
#include "../Env/nfl.hh"
#include "../Env/largegrid.hh"
#include "../Env/VaryingGrid.hh"
//#include "../Env/Minesweeper.hh"
#include "../Env/Chain.hh"
#include "../Env/austinmap.hh"
#include "../Env/MountainCar.hh"
#include "../Env/PuddleWorld.hh"
#include "../Env/CartPole.hh"
#include "../Env/Acrobot.hh"
#include "../Env/TestGrid.hh"
#include "../Env/BusSuspension.hh"
#include "../Env/MassSpringDamper.hh"
#include "../Env/trickroom.hh"
#include "../Env/RadioJamming.hh"
#include "../Env/teleport.hh"
#include "../Env/RobotCarVel.hh"
#include "../Env/RobotCarStop.hh"
#include "../Env/Chase.hh"

////////////
// Agents //
////////////
#include "../Agent/QLearner.hh"
#include "../Agent/dQLearner.hh"
#include "../Agent/QwithFA.hh"
#include "../Agent/FlatRMax.hh"
#include "../Agent/RLDT.hh"
#include "../Agent/ModelBasedAgent.hh"
#include "../Agent/ManualAgent.hh"
#include "../Agent/DiscretizationAgent.hh"
#include "../Agent/SavedPolicy.hh"
#include "../Agent/Sarsa.hh"
#include "../Agent/Dyna.hh"

//#include "../FittedRmaxQ/NickAgent.hh"

// hand coded ones
//#include "../Agent/HandStock.hh"
//#include "../Agent/HandCodedFuel.hh"
//#include "../Agent/HandCodedExplore.hh"
//#include "../Agent/HandCoded.hh"


#include <vector>
#include <sstream>
#include <iostream>

#define P_AGENT 1
#define P_MODEL 2
#define P_PLAN  3
#define P_EXPLORE   4
#define P_ENV  5
#define P_STOCH 6
#define P_SEED 7
#define P_PREDTYPE 8
#define P_NMODELS 9
#define P_NSTATES 10
#define P_ACTRATE 11
#define P_BCOEFF  12
#define P_HISTORY 13
#define P_FEATPCT 14

const unsigned MAXEPISODES = 10000; //50000; //500000; //500; //200; //1000; //2000 // episodes per trial
unsigned NUMEPISODES = 100; //10; //200; //500; //200;
const unsigned NUMTRIALS = 1; //30; //30; //5; //30; //30; //50
unsigned MAXSTEPS = 1000; // per episode
const bool PRINTS = true;


bool depTrans = false;
bool relTrans = true;

#define VISUALIZE
//#define LOGDATA
//#define TIME
//#define TIMELOG
//#define EPTIME

//float rperstep[NUMTRIALS][NUMEPISODES * MAXSTEPS];
//float rperstepsums[NUMEPISODES * MAXSTEPS];
#ifdef LOGDATA
float rdata[NUMTRIALS][MAXEPISODES];
float sdata[NUMTRIALS][MAXEPISODES];
float rsums[MAXEPISODES];
float ssums[MAXEPISODES];
#endif

#ifdef TIMELOG
float timeRewards[2000]; // avg reward every 10 seconds
float actRewards[2000];  // avg reward every 100 actions
#endif

#ifdef EPTIME
float episodeTimes[5000];
#endif

double getSeconds();
void logExp(ofstream *v, int epi, int step, std::vector<float> s, int a, float r);

int main(int argc, char **argv) {

  if (argc != 15){// || *argv[P_AGENT] != 'm')) {
    std::cerr << "Usage: rl <agent> <modelType> <plannerType> <exploreType> <env> <stochastic> <seed> <predType> <nModels> <disc> <actrate> <b> <history> <featPct>\n";
    std::cerr << "Agent: q-QLearner, a-QwithFA, r-RMax, m-ModelBased c-ContinuousModelBased p-SavedPolicy i-Interactive d-dQLearner\n";
    std::cerr << "Model: t-Tree, r-RMax, s-Stump, x-M5Multi, k-M5Single, m-LSTmulti e->LSTsingle g-GPregrss\n";
    std::cerr << "Planner: v-ValueIter p-PolicyIter s-PriSweeping a-UCT k-ParallelUCT m-MBS\n";
    std::cerr << "Explore: 0-unknown, 3-contbonus, 4-threshbonus, 5-contbonus+R 6-threshbonus+R 7-noexplore 8-epsilon-greedy\n";
    std::cerr << "Env: Name of the environment (taxi, fourroom, fuellow, fuelhigh, mcar, puddle, cartpole, bus, msd, trick, radio, etc)\n";
    std::cerr << "Stochastic: 0-deterministic, 1-stochastic \n";
    std::cerr << "Predtype: a-avg, w-weightavg, b-best, s-separate\n";
    std::cerr << "# Models\n";
    std::cerr << "Discretization # states\n";
    std::cerr << "Action Hz rate\n";
    std::cerr << "b coefficient (for explore type 3-6)\n";
    std::cerr << "History size\n";
    std::cerr << "Fraction of possible splits to remove from each node in tree\n";
    exit(0);
  }

  //  float MAX_TIME = atof(argv[P_AGENT]);
  //bool timeInput = (MAX_TIME > 0);


  /*
    if (timeInput){
    *argv[P_AGENT] = 'm';

    // unless planner tells us its parallel (l)
    if (*argv[P_PLAN] == 'l'){
    *argv[P_AGENT] = 'p';
    *argv[P_PLAN] = 'v';
    }
    } else {
    MAX_TIME = 1.0;
    }
  */

  // some agents dont use modeltype, plannertype, exploretype
  if (*argv[P_AGENT] != 'm' && *argv[P_AGENT] != 'c'){
    if (*argv[P_MODEL] != '0' || *argv[P_PLAN] != '0' || *argv[P_EXPLORE] != '0'){
      std::cerr << "Usage: rl <agent> <modelType> <plannerType> <exploreType> <env> <stochastic> <seed> \n";
      std::cerr << "Only Model Based Agent uses a model, planner, and exploreType other than 0\n";
      exit(0);
    }
  }





  bool stochastic = false;
  if (std::atoi(argv[P_STOCH]) == 1){
    if (PRINTS) cout << "Stohastic\n";
    stochastic = true;
  } else if (std::atoi(argv[P_STOCH]) == 0) {
    if (PRINTS) cout << "Deterministic\n";
    stochastic = false;
  } else {
    std::cerr << "Invalid value for stochastic" << endl;
    exit(-1);
  }

  // define/print model,planner,explore type for modelbasedagent
  int modelType = 0;
  int exploreType = 0;
  int plannerType = 0;
  int nModels = 1;
  int predType = 1;
  int history = std::atoi(argv[P_HISTORY]);
  float b = std::atof(argv[P_BCOEFF]);
  float featPct = std::atof(argv[P_FEATPCT]);

  if (*argv[P_AGENT] == 'm' || *argv[P_AGENT] == 'c'){

    // get predtype and nmodels
    nModels = std::atoi(argv[P_NMODELS]);

    if (*argv[P_PREDTYPE] == 'a')
      predType = AVERAGE;
    else if (*argv[P_PREDTYPE] == 'w')
      predType = WEIGHTAVG;
    else if (*argv[P_PREDTYPE] == 'b')
      predType = BEST;
    else if (*argv[P_PREDTYPE] == 's')
      predType = SEPARATE;
    else
      predType = std::atoi(argv[P_PREDTYPE]);

    if (predType == AVERAGE){
      if (PRINTS) cout << "Prediction: Average " << nModels << " models." << endl;
    }
    else if (predType == WEIGHTAVG){
      if (PRINTS) cout << "Prediction: Weighted average of " << nModels
                       << " models." << endl;
    }
    else if (predType == BEST){
      if (PRINTS) cout << "Prediction: Take best of " << nModels << " models."
                       << endl;
    }
    else if (predType == SEPARATE){
      if (PRINTS) cout << "Prediction: Average " << nModels 
                       << " models for uncertainty and build separate full model for planning." << endl;
    }

    // tree
    if (*argv[P_MODEL] == 't')
      modelType = C45TREE;
    // rmax
    else if (*argv[P_MODEL] == 'r')
      modelType = RMAX;
    // stump
    else if (*argv[P_MODEL] == 's')
      modelType = STUMP;
    // continuous tree
    else if (*argv[P_MODEL] == 'z')
      modelType = ALLM5TYPES;
    else if (*argv[P_MODEL] == 'c')
      modelType = M5MULTI;
    else if (*argv[P_MODEL] == 'x')
      modelType = M5ALLMULTI;
    else if (*argv[P_MODEL] == 'k')
      modelType = M5ALLSINGLE;
    else if (*argv[P_MODEL] == 'l')
      modelType = M5SINGLE;
    else if (*argv[P_MODEL] == 'e')
      modelType = LSTSINGLE;
    else if (*argv[P_MODEL] == 'm')
      modelType = LSTMULTI;
    else if (*argv[P_MODEL] == 'g')
      modelType = GPREGRESS;
    else if (*argv[P_MODEL] == 'p')
      modelType = GPTREE;
    else
      modelType = std::atoi(argv[P_MODEL]);

    if (modelType == RMAX){
      if (PRINTS) cout << "Model: RMax" << endl;
    } else if (modelType == SLF){
      if (PRINTS) cout << "Model: SLF" << endl;
    } else if (modelType == C45TREE){
      if (PRINTS) cout << "Model: C4.5 Tree" << endl;
    } else if (modelType == SVM){
      if (PRINTS) cout << "Model: SVM" << endl;
    } else if (modelType == STUMP){
      if (PRINTS) cout << "Model: Stump" << endl;
    } else if (modelType == ALLM5TYPES){
      if (PRINTS) cout << "Model: Combine all M5 variants" << endl;
    } else if (modelType == M5MULTI){
      if (PRINTS) cout << "Model: M5 Multivariate Tree" << endl;
    } else if (modelType == M5ALLMULTI){
      if (PRINTS) cout << "Model: M5 Multivariate Tree (all feats)" << endl;
    } else if (modelType == M5SINGLE){
      if (PRINTS) cout << "Model: M5 Simple Tree" << endl;
    } else if (modelType == M5ALLSINGLE){
      if (PRINTS) cout << "Model: M5 Simple Tree (all feats)" << endl;
    } else if (modelType == LSTSINGLE){
      if (PRINTS) cout << "Model: Simple LM Splits Tree" << endl;
    } else if (modelType == LSTMULTI){
      if (PRINTS) cout << "Model: Multi LM Splits Tree" << endl;
    } else if (modelType == GPREGRESS){
      if (PRINTS) cout << "Model: GP Regression" << endl;
    } else if (modelType == GPTREE){
      if (PRINTS) cout << "Model: GP Tree" << endl;
    } else{
      std::cerr << "ERROR: invalid model type" << endl;
      exit(-1);
    }

    if (PRINTS){
      if (relTrans) cout << "Model using relative transitions" << endl;
      else cout << "Model using absolute transitions" << endl;
      if (depTrans) cout << "Dependent transitions" << endl;
      else cout << "Independent transitions" << endl;
      cout << "Feat_pct: " << featPct << endl;
    }

    if (*argv[P_PLAN] == 'v'){
      plannerType = VALUE_ITERATION;
      if (PRINTS) cout << "Planner: Value Iteration\n";
    } else if (*argv[P_PLAN] == 'p'){
      plannerType = POLICY_ITERATION;
      if (PRINTS) cout << "Planner: Policy Iteration\n";
    } else if (*argv[P_PLAN] == 's'){
      plannerType = PRI_SWEEPING;
      if (PRINTS) cout << "Planner: Prioritized Sweeping\n";
    } else if (*argv[P_PLAN] == 'g'){
      plannerType = MOD_PRI_SWEEPING;
      if (PRINTS) cout << "Planner: Modified Prioritized Sweeping\n";
    } else if (*argv[P_PLAN] == 'u'){
      plannerType = UCT;
      if (PRINTS) cout << "Planner: UCT\n";
    } else if (*argv[P_PLAN] == 'w'){
      plannerType = UCT_WITH_L;
      if (PRINTS) cout << "Planner: UCT with lambdas\n";
    } else if (*argv[P_PLAN] == 'y'){
      plannerType = UCT_WITH_ENV;
      if (PRINTS) cout << "Planner: UCT with complete env model\n";
    } else if (*argv[P_PLAN] == 'e'){
      plannerType = ET_UCT;
      if (PRINTS) cout << "Planner: UCT(lambda)\n";
    } else if (*argv[P_PLAN] == 'a'){
      plannerType = ET_UCT_ACTUAL;
      if (PRINTS) cout << "Planner: UCT(lambda) from real-values states\n";
    } else if (*argv[P_PLAN] == 'r'){
      plannerType = ET_UCT_CORNERS;
      if (PRINTS) cout << "Planner: UCT(lambda) sample corners of discrete state\n";
    } else if (*argv[P_PLAN] == 'k'){
      plannerType = PAR_ETUCT_ACTUAL;
      if (PRINTS) cout << "Planner: Parallel UCT(lambda) from real-values states\n";
    } else if (*argv[P_PLAN] == 'o'){
      plannerType = POMDP_ETUCT;
      if (PRINTS) cout << "Planner: PO UCT(lambda) with " << history << " state history\n";
    } else if (*argv[P_PLAN] == 'i'){
      plannerType = POMDP_PAR_ETUCT;
      if (PRINTS) cout << "Planner: Parallel PO UCT(lambda) with " << history << " state history\n";
    } else if (*argv[P_PLAN] == 't'){
      plannerType = PAR_ETUCT_CORNERS;
      if (PRINTS) cout << "Planner: Parallel UCT(lambda) sample corners of discrete state\n";
    } else if (*argv[P_PLAN] == 'f'){
      plannerType = ET_UCT_L1;
      if (PRINTS) cout << "Planner: UCT(lambda) with lambda=1\n";
    } else if (*argv[P_PLAN] == 'm'){
      plannerType = MBS_VI;
      if (PRINTS) cout << "Planner: MBS k=1 with VI\n";
    } else if (*argv[P_PLAN] == 'h'){
      plannerType = SWEEPING_UCT_HYBRID;
      if (PRINTS) cout << "Planner: Hybrid of Prioritized Sweeping and UCT(lambda)\n";
    } else if (*argv[P_PLAN] == 'c'){
      plannerType = CMAC_PLANNER;
      if (PRINTS) cout << "Planner: CMAC (Tile Coding)\n";
    } else if (*argv[P_PLAN] == 'n'){
      plannerType = NN_PLANNER;
      if (PRINTS) cout << "Planner: Neural Network\n";
    } else if (*argv[P_PLAN] == 'l'){
      plannerType = PARALLEL_ET_UCT;
      if (PRINTS) cout << "Planner: Parallel version of UCT(lambda)\n";
    } else {
      std::cerr << "ERROR: invalid planner type\n";
      exit(-1);
    }

    exploreType = std::atoi(argv[P_EXPLORE]);
    if (exploreType == 0){
      if (PRINTS) cout << "Explore: Unknowns\n";
    } else if (exploreType == 1){
      if (PRINTS) cout << "Explore: Two Modes\n";
    } else if (exploreType == 2){
      if (PRINTS) cout << "Explore: Two Modes + reward\n";
    } else if (exploreType == 3){
      if (PRINTS) cout << "Explore: Continuously varying bonus b: " << b << endl;
    } else if (exploreType == 4){
      if (PRINTS) cout << "Explore: Thresholded bonus b: " << b << endl;
    } else if (exploreType == 5){
      if (PRINTS) cout << "Explore: Continuously varying bonus b: " << b << " + reward\n";
    } else if (exploreType == 6){
      if (PRINTS) cout << "Explore: Thresholded bonus  b: " << b << " + reward\n";
    } else if (exploreType == 7){
      if (PRINTS) cout << "Explore: No exploration bonuses\n";
    } else if (exploreType == EPSILONGREEDY){
      if (PRINTS) cout << "Explore: epsilon-greedy\n";
    } else if (exploreType == VISITS_CONF){
      if (PRINTS) cout << "Explore: visits and confidence\n";
    } else if (exploreType == UNVISITED_BONUS){
      if (PRINTS) cout << "Explore: Small bonus for unvisited states\n";
    } else if (exploreType == UNVISITED_ACT_BONUS){
      if (PRINTS) cout << "Explore: Small bonus for unvisited state-actions\n";
    } else{
      std::cerr << "ERROR: Invalid exploration type\n";
      exit(-1);
    }
  }

  float actrate = std::atoi(argv[P_ACTRATE]);

  std::ostringstream expstring2;
  expstring2 << exploreType;

  // create the appropriate condor file for this
  if (std::atoi(argv[P_SEED]) == -1){
    cout << "Seed -1, make condor file" << endl << flush;

    string filename("Condor/condor.");

    std::ostringstream numstring2;
    numstring2 << nModels;

    filename += *argv[1];
    filename += ".";
    filename += *argv[2];
    filename += ".";
    filename += *argv[3];
    filename += ".";
    filename += expstring2.str();
    filename += ".";
    filename += argv[5];
    filename += ".";
    filename += *argv[6];
    filename += ".";
    filename += *argv[8];
    filename += ".";
    filename += numstring2.str();
    filename += ".";
    filename += argv[10];
    filename += ".";
    filename += argv[11];
    filename += ".";
    filename += argv[12];
    filename += ".";
    filename += argv[13];
    filename += ".";
    filename += argv[14];
    filename += ".desc";

    cout << "\nCreating condor file: " << filename << endl << flush;
    cout << endl << "condor_submit " << filename << endl << flush;

    ofstream fout(filename.c_str());
    //ofstream fout("Condor/condor.desc");
    fout << "+Group = \"GRAD\"\n";
    fout << "+Project = \"AI_ROBOTICS\"\n";
    fout << "+ProjectDescription = \"Reinforcement Learning\"\n";
    fout << "universe = vanilla\n";
    fout << "executable = Build/rl\n";
    fout << "Requirements = Memory >= 4000 && Lucid\n";
    fout << "Log = " << *argv[1] << "." << *argv[2] << "." << *argv[3] << "."
         << expstring2.str() << "." << argv[5] << "." << *argv[6] << "." << *argv[8]
         << "." << nModels << "." << argv[10]  << "." << argv[11] << "." << argv[12] 
         << "." << argv[13] << "." << argv[14] << ".condor.log\n";
    fout << "Notification = Never\n";
    fout << "Arguments = " << *argv[1] << " " << *argv[2] << " " << *argv[3] << " "
         << expstring2.str() << " " << argv[5] << " " << *argv[6]<< " $(Process) " << " " << *argv[8]
         << " " << nModels << " " << argv[10] << " " << argv[11] << " " << argv[12] 
         << " " << argv[13] << " " << argv[14] << "\n";
    fout << "Output = " << *argv[1] << "." << *argv[2] << "." << *argv[3] << "."
         << expstring2.str() << "." << argv[5] << "." << *argv[6] << "." << *argv[8]
         << "." << nModels << "." << argv[10] << "." << argv[11] << "." << argv[12] 
         << "." << argv[13] << "." << argv[14]
         << ".condor.out.$(Process)\n";
    //fout << "Output = " << *argv[3] << "." << *argv[1] << "." << argv[5] << "."
    // << *argv[6] << "." << *argv[2] << "." << expstring2.str() << ".results.$(Process)\n";
    fout << "Error = " << *argv[1] << "." << *argv[2] << "." << *argv[3] << "."
         << expstring2.str() << "." << argv[5] << "." << *argv[6] << "." << *argv[8]
         << "." << nModels << "." << argv[10] << "." << argv[11] << "." << argv[12] 
         << "." << argv[13] << "." << argv[14]
         << ".condor.err.$(Process)\n";
    fout << "Queue 30\n";

    fout.close();
    exit(-1);
  }

#ifdef VISUALIZE
  string filename("visualize.exp.");
  filename += argv[1];
  filename += ".";
  filename += argv[2];
  filename += ".";
  filename += argv[3];
  filename += ".";
  filename += argv[4];
  filename += ".";
  filename += argv[5];
  filename += ".";
  filename += argv[6];
  filename += ".";
  filename += argv[8];
  filename += ".";
  filename += argv[9];
  filename += ".";
  filename += argv[10];
  filename += ".";
  filename += argv[11];
  filename += ".";
  filename += argv[12];
  filename += ".";
  filename += argv[13];
  filename += ".";
  filename += argv[14];
  filename += ".seed.";
  filename += argv[7];
  ofstream vout(filename.c_str());
  vout << argv[1] << "," << argv[2] << "," << argv[3]
       << "," << argv[4] << "," << argv[5] << "," << argv[6]
       << "," << argv[7] << "," << argv[8] << "," << argv[9]
       << "," << argv[10] << "," << argv[11] << "," << argv[12] 
       << "," << argv[13] << "," << argv[14] << endl;
#endif


  Random rng(1 + std::atoi(argv[P_SEED]));

  std::vector<int> statesPerDim;

  // Construct environment here.
  Environment* e;

  if (strcmp(argv[P_ENV], "acrobot") == 0){
    if (PRINTS) cout << "Environment: Acrobot\n";
    e = new Acrobot(rng, stochastic);
  }

  else if (strcmp(argv[P_ENV], "chase") == 0){
    if (PRINTS) cout << "Environment: Chase\n";
    e = new Chase(rng, stochastic);
  }

  else if (strcmp(argv[P_ENV], "puddle") == 0){
    if (PRINTS) cout << "Environment: Puddle World\n";
    e = new PuddleWorld(rng, stochastic);
  }

  else if (strcmp(argv[P_ENV], "cartpole") == 0){
    if (PRINTS) cout << "Environment: Cart Pole\n";
    e = new CartPole(rng, stochastic);
  }

  else if (strcmp(argv[P_ENV], "bus") == 0){
    if (PRINTS) cout << "Environment: Bus Suspension\n";
    e = new BusSuspension(rng, stochastic);
  }

  else if (strcmp(argv[P_ENV], "msd") == 0){
    if (PRINTS) cout << "Environment: Mass Spring Damper\n";
    e = new MassSpringDamper(rng, stochastic);
  }
  
  else if (strcmp(argv[P_ENV], "mcar") == 0){
    if (PRINTS) cout << "Environment: Mountain Car\n";
    e = new MountainCar(rng, stochastic, false, 0);
  }

  else if (strcmp(argv[P_ENV], "linmcar") == 0){
    if (PRINTS) cout << "Environment: Linear Mountain Car\n";
    e = new MountainCar(rng, stochastic, true, 0);
  }

  else if (strcmp(argv[P_ENV], "mcar-d1") == 0){
    if (PRINTS) cout << "Environment: Mountain Car - 1 step delay\n";
    e = new MountainCar(rng, stochastic, false, 1);
  }

  else if (strcmp(argv[P_ENV], "mcar-d2") == 0){
    if (PRINTS) cout << "Environment: Mountain Car - 2 step delay\n";
    e = new MountainCar(rng, stochastic, false, 2);
  }
  

  else if (strcmp(argv[P_ENV], "testgrid") == 0){
    if (PRINTS) cout << "Environment: Test Grid\n";
    e = new TestGrid(rng, stochastic, false);
  }

  else if (strcmp(argv[P_ENV], "shortpath") == 0){
    if (PRINTS) cout << "Environment: Short PathTest Grid\n";
    e = new TestGrid(rng, stochastic, true);
  }

  // taxi
  else if (strcmp(argv[P_ENV], "taxi") == 0){
    if (PRINTS) cout << "Environment: Taxi\n";
    e = new Taxi(rng, stochastic);
  }

  /*
  // one room
  else if (*argv[P_ENV] == '1'){
  if (PRINTS) cout << "Environment: OneRoom\n";
  e = new OneRoom(rng, stochastic, true);
  }
  */

  // wmaze
  else if (strcmp(argv[P_ENV], "wmaze") == 0){
    if (PRINTS) cout << "Environment: W-Maze\n";
    e = new WMaze(rng, stochastic, 0);
  }
  else if (strcmp(argv[P_ENV], "wmaze1") == 0){
    if (PRINTS) cout << "Environment: W-Maze with 1 step delay\n";
    e = new WMaze(rng, stochastic, 1);
  }
  else if (strcmp(argv[P_ENV], "wmaze2") == 0){
    if (PRINTS) cout << "Environment: W-Maze with 2 step delay\n";
    e = new WMaze(rng, stochastic, 2);
  }
  else if (strcmp(argv[P_ENV], "wmaze5") == 0){
    if (PRINTS) cout << "Environment: W-Maze with 5 step delay\n";
    e = new WMaze(rng, stochastic, 5);
  }

  // two rooms
  else if (strcmp(argv[P_ENV], "tworoom") == 0){
    if (PRINTS) cout << "Environment: TwoRooms\n";
    e = new TwoRooms(rng, stochastic, true, 0, false);
  }

  // two rooms - delayed actions
  else if (strcmp(argv[P_ENV], "delayed") == 0){
    if (PRINTS) cout << "Environment: TwoRooms w/ 1-step action delay\n";
    e = new TwoRooms(rng, stochastic, true, 1, false);
  }

  // two rooms - 2 step delayed actions
  else if (strcmp(argv[P_ENV], "delayed2") == 0){
    if (PRINTS) cout << "Environment: TwoRooms w/ 2-step action delay\n";
    e = new TwoRooms(rng, stochastic, true, 2, false);
  }

  // two rooms - 5 step delayed actions
  else if (strcmp(argv[P_ENV], "delayed5") == 0){
    if (PRINTS) cout << "Environment: TwoRooms w/ 5-step action delay\n";
    e = new TwoRooms(rng, stochastic, true, 5, false);
  }

  // two rooms - multiple possible goals
  else if (strcmp(argv[P_ENV], "multigoal") == 0){
    if (PRINTS) cout << "Environment: TwoRooms w/ Multiple Goals\n";
    e = new TwoRooms(rng, stochastic, true, 0, true);
  }

  // car vel, 2 to 7
  else if (strcmp(argv[P_ENV], "car2to7") == 0){
    if (PRINTS) cout << "Environment: Car Velocity 2 to 7 m/s\n";
    e = new RobotCarVel(rng, false, true, false, false);
    statesPerDim.resize(4,0);
    statesPerDim[0] = 12;
    statesPerDim[1] = 120;
    statesPerDim[2] = 4;
    statesPerDim[3] = 10;
    MAXSTEPS = 100;//200;
    NUMEPISODES = 201;
  }
  // car vel, 7 to 2
  else if (strcmp(argv[P_ENV], "car7to2") == 0){
    if (PRINTS) cout << "Environment: Car Velocity 7 to 2 m/s\n";
    e = new RobotCarVel(rng, false, false, false, false);
    statesPerDim.resize(4,0);
    statesPerDim[0] = 12;
    statesPerDim[1] = 120;
    statesPerDim[2] = 4;
    statesPerDim[3] = 10;
    MAXSTEPS = 100;//200;
    NUMEPISODES = 201;
  }
  // car vel, random vels
  else if (strcmp(argv[P_ENV], "carrandom") == 0){
    if (PRINTS) cout << "Environment: Car Velocity Random Velocities\n";
    e = new RobotCarVel(rng, true, false, false, false);
    statesPerDim.resize(4,0);
    statesPerDim[0] = 12;
    statesPerDim[1] = 48;
    statesPerDim[2] = 4;
    statesPerDim[3] = 10;
    MAXSTEPS = 100;//200;
    NUMEPISODES = 401;
  }
  // car vel, random vels, with lag
  else if (strcmp(argv[P_ENV], "carrandomlag") == 0){
    if (PRINTS) cout << "Environment: Car Velocity Random Velocities with lag\n";
    e = new RobotCarVel(rng, true, false, false, true);
    statesPerDim.resize(6,0);
    statesPerDim[0] = 12;
    statesPerDim[1] = 120; //48;
    statesPerDim[2] = 4;
    statesPerDim[3] = 10;
    statesPerDim[4] = 4;
    statesPerDim[5] = 10;
    MAXSTEPS = 100;//200;
    NUMEPISODES = 401;
  }
  // car vel, 7 to 2 with lag
  else if (strcmp(argv[P_ENV], "car7to2lag") == 0){
    if (PRINTS) cout << "Environment: Car Velocity 7 to 2 m/s with lag\n";
    e = new RobotCarVel(rng, false, false, false, true);
    statesPerDim.resize(4,0);
    statesPerDim[0] = 12;
    statesPerDim[1] = 48;
    statesPerDim[2] = 4;
    statesPerDim[3] = 10;
    MAXSTEPS = 100;//200;
    NUMEPISODES = 201;
  }
  // car vel, 10 to 6 with lag
  else if (strcmp(argv[P_ENV], "car10to6lag") == 0){
    if (PRINTS) cout << "Environment: Car Velocity 10 to 6 m/s with lag\n";
    e = new RobotCarVel(rng, false, false, true, true);
    statesPerDim.resize(6,0);
    statesPerDim[0] = 12;
    statesPerDim[1] = 120; //48; //120;
    statesPerDim[2] = 4;
    statesPerDim[3] = 10;
    statesPerDim[4] = 4;
    statesPerDim[5] = 10;
    MAXSTEPS = 100; //200;
    NUMEPISODES = 201;
  }
  // car vel, 10 to 6 no lag
  else if (strcmp(argv[P_ENV], "car10to6") == 0){
    if (PRINTS) cout << "Environment: Car Velocity 10 to 6 m/s\n";
    e = new RobotCarVel(rng, false, false, true, false);
    statesPerDim.resize(4,0);
    statesPerDim[0] = 12;
    statesPerDim[1] = 48; //120;
    statesPerDim[2] = 4;
    statesPerDim[3] = 10;
    MAXSTEPS = 100;
    NUMEPISODES = 201;
  }
  
  // car stop from 6
  else if (strcmp(argv[P_ENV], "carstop") == 0){
    if (PRINTS) cout << "Environment: Car Stop from 6 m/s\n";
    e = new RobotCarStop(rng, false, false);
    statesPerDim.resize(4,0);
    statesPerDim[0] = 60;
    statesPerDim[1] = 48; //120;
    statesPerDim[2] = 4;
    statesPerDim[3] = 10;
    NUMEPISODES = 201;
  }
  // car stop from random
  else if (strcmp(argv[P_ENV], "carstoprandom") == 0){
    if (PRINTS) cout << "Environment: Car Stop from random vel\n";
    e = new RobotCarStop(rng, true, false);
    statesPerDim.resize(4,0);
    statesPerDim[0] = 60;
    statesPerDim[1] = 48; //120;
    statesPerDim[2] = 4;
    statesPerDim[3] = 10;
    NUMEPISODES = 201;
  }
  // car stop from random
  else if (strcmp(argv[P_ENV], "carstoprandomlag") == 0){
    if (PRINTS) cout << "Environment: Car Stop from random vel w lag\n";
    e = new RobotCarStop(rng, true, true);
    statesPerDim.resize(4,0);
    statesPerDim[0] = 60;
    statesPerDim[1] = 48; //120;
    statesPerDim[2] = 4;
    statesPerDim[3] = 10;
    NUMEPISODES = 201;
  }
  else if (strcmp(argv[P_ENV], "carstoplag") == 0){
    if (PRINTS) cout << "Environment: Car Stop from 6 m/s w lag\n";
    e = new RobotCarStop(rng, false, true);
    statesPerDim.resize(4,0);
    statesPerDim[0] = 60;
    statesPerDim[1] = 48; //120;
    statesPerDim[2] = 4;
    statesPerDim[3] = 10;
    NUMEPISODES = 201;
  }
  /*
  // trap room
  else if (*argv[P_ENV] == 'u'){
  if (PRINTS) cout << "Environment: TrapRoom\n";
  e = new TrapRoom(rng, stochastic);
  }
  */

  // four rooms
  else if (strcmp(argv[P_ENV], "fourroom") == 0){
    if (PRINTS) cout << "Environment: FourRooms\n";
    e = new FourRooms(rng, stochastic, true, false);
  }

  // trick rooms
  else if (strcmp(argv[P_ENV], "trick") == 0){
    if (PRINTS) cout << "Environment: Trick Room\n";
    e = new TrickRoom(rng, stochastic, false, true);
  }
  else if (strcmp(argv[P_ENV], "revtrick") == 0){
    if (PRINTS) cout << "Environment: Reverse Trick Room\n";
    e = new TrickRoom(rng, stochastic, true, true);
  }
  else if (strcmp(argv[P_ENV], "notrick") == 0){
    if (PRINTS) cout << "Environment: No Trick Room\n";
    e = new TrickRoom(rng, stochastic, false, false);
  }
  else if (strcmp(argv[P_ENV], "revnotrick") == 0){
    if (PRINTS) cout << "Environment: Reverse no Trick Room\n";
    e = new TrickRoom(rng, stochastic, true, false);
  }

  // teleport room
  else if (strcmp(argv[P_ENV], "teleport") == 0){
    if (PRINTS) cout << "Environment: Teleport Room\n";
    e = new Teleport(rng, stochastic);
  }

  // radio jamming
  else if (strcmp(argv[P_ENV], "radio") == 0){
    if (PRINTS) cout << "Environment: Radio Jamming\n";
    e = new RadioJamming(rng);
  }

  else if (strcmp(argv[P_ENV], "radio-np") == 0){
    if (PRINTS) cout << "Environment: Radio Jamming - Block Only\n";
    e = new RadioJamming(rng, 0);
  }



  /*
  // four rooms with guiding rewards
  else if (*argv[P_ENV] == 'h'){
  if (PRINTS) cout << "Environment: FourRooms with guiding reward\n";
  e = new FourRooms(rng, stochastic, true, true);
  }

  // four rooms with wall distances
  else if (*argv[P_ENV] == 'd'){
  if (PRINTS) cout << "Environment: FourRooms with wall distances\n";
  e = new FourRooms(rng, stochastic, true);
  }

  // four rooms with wall distances and reward sensor
  else if (*argv[P_ENV] == 'y'){
  if (PRINTS) cout << "Environment: FourRooms with wall distances and reward sensor\n";
  e = new FourRooms(rng, stochastic);
  }
  */

  // four rooms with red herring state
  else if (strcmp(argv[P_ENV], "redherring") == 0){
    if (PRINTS) cout << "Environment: RedHerring\n";
    e = new RedHerring(rng, stochastic, true, false);
  }

  // four rooms with energy level
  else if (strcmp(argv[P_ENV], "energy") == 0){
    if (PRINTS) cout << "Environment: EnergyRooms\n";
    e = new EnergyRooms(rng, stochastic, true, false);
  }

  // gridworld with fuel (fuel stations on top and bottom with random costs)
  else if (strcmp(argv[P_ENV], "fuellow") == 0){
    if (PRINTS) cout << "Environment: FuelRooms, Little variation in fuel cost\n";
    e = new FuelRooms(rng, false, stochastic);
    NUMEPISODES = 100;
  }

  // gridworld with fuel (fuel stations on top and bottom with random costs)
  else if (strcmp(argv[P_ENV], "fuelhigh") == 0){
    if (PRINTS) cout << "Environment: FuelRooms, Large variation in fuel cost\n";
    e = new FuelRooms(rng, true, stochastic);
    NUMEPISODES = 100;
  }

  // explore domain
  else if (strcmp(argv[P_ENV], "explore") == 0){
    if (PRINTS) cout << "Environment: Explore\n";
    e = new Explore(rng, stochastic);
  }

  /*
  // webots sim
  else if (*argv[P_ENV] == 'x'){
  if (PRINTS) cout << "Environment: Penalty Kick\n";
  // random ball
  e = new PenaltyKick(rng, false, true, 4);
  // static ball
  //e = new PenaltyKick(rng, true, true, 4);
  }
  */

  /*
  // minesweeper
  else if (*argv[P_ENV] == 'm'){
  if (PRINTS) cout << "Environment: Minesweeper\n";
  e = new Minesweeper(rng);
  }
  */

  // multi room domain
  //else if (*argv[P_ENV] == 'm'){
  //  e = new MultiRoom(rng, stochastic);
  //}

  // large domain
  //else if (*argv[P_ENV] == 'l'){
  // e = new LargeDomain(rng, stochastic);
  //}

  // austin map domain
  else if (strcmp(argv[P_ENV], "map") == 0){
    e = new AustinMap(rng, stochastic);
  }

  // Lights domain
  else if (strcmp(argv[P_ENV], "lights") == 0){
    if (PRINTS) cout << "Environment: Lights\n";
    e = new Lights(rng, stochastic);
  }

  // Varying grid, normal
  else if (strcmp(argv[P_ENV], "varynormal") == 0){
    if (PRINTS) cout << "Environment: VaryingGrid, Normal\n";
    e = new VaryingGrid(rng, false, stochastic);
  }

  // Varying grid, extra variation
  else if (strcmp(argv[P_ENV], "varyextra") == 0){
    if (PRINTS) cout << "Environment: VaryingGrid, Extra variation\n";
    e = new VaryingGrid(rng, true, stochastic);
  }

  // Varying grid, varies every row
  else if (strcmp(argv[P_ENV], "varyevery") == 0){
    if (PRINTS) cout << "Environment: VaryingGrid, Every row\n";
    e = new VaryingGrid(rng, 120, true, stochastic, true);
  }

  // castle
  else if (strcmp(argv[P_ENV], "castle") == 0){
    if (PRINTS) cout << "Environment: Castle\n";
    e = new Castle(rng, stochastic);
  }


  else if (strcmp(argv[P_ENV], "chain") == 0){
    if (PRINTS) cout << "Environment: Chain\n";
    e = new Chain(rng, stochastic);
  }

  // castle with no energy and more terminal squares
  //else if (*argv[P_ENV] == 'g'){
  //  if (PRINTS) cout << "Environment: Castle w/o energy + terminals\n";
  //  e = new CastleR(rng, stochastic);
  //}

  // large gridworld
  else if (strcmp(argv[P_ENV], "large") == 0){
    MAXSTEPS = 100000;
    if (PRINTS) cout << "Environment: Large Gridworld\n";
    e = new LargeGrid(rng, stochastic, true);
  }

  // stocks
  else if (strcmp(argv[P_ENV], "stocks32") == 0){
    int nsectors = 3;
    int nstocks = 2;
    if (PRINTS) cout << "Enironment: Stocks with " << nsectors
                     << " sectors and " << nstocks << " stocks\n";
    e = new Stocks(rng, stochastic, nsectors, nstocks);
  }

  // stocks 2
  else if (strcmp(argv[P_ENV], "stocks43") == 0){
    int nsectors = 4;
    int nstocks = 3;
    if (PRINTS) cout << "Enironment: Stocks with " << nsectors
                     << " sectors and " << nstocks << " stocks\n";
    e = new Stocks(rng, stochastic, nsectors, nstocks);
  }

  // stocks 3
  else if (strcmp(argv[P_ENV], "stocks33") == 0){
    int nsectors = 3;
    int nstocks = 3;
    if (PRINTS) cout << "Enironment: Stocks with " << nsectors
                     << " sectors and " << nstocks << " stocks\n";
    e = new Stocks(rng, stochastic, nsectors, nstocks);
  }

  /*
  // rising optimum
  else if (*argv[P_ENV] == 'o'){
  if (PRINTS) cout << "Environment: Rising Optimum\n";
  e = new RisingOptimum(rng);
  }
  */

  // castle
  else if (strcmp(argv[P_ENV], "smallcastle") == 0){
    if (PRINTS) cout << "Environment: SmallCastle\n";
    e = new SmallCastle(rng, stochastic);
  }

  //else if (*argv[P_ENV] == 's'){
  // e = new SmallCastle(rng, stochastic);
  //}

  else if (strcmp(argv[P_ENV], "playroom") == 0){
    // last boolean tells if there's reward or not
    e = new PlayRoom(rng, stochastic, false, true); //false);
  }

  /*
    else if (*argv[P_ENV] == 'n'){
    if (PRINTS) cout << "Environment: NFL\n";
    // last boolean tells if there's reward or not
    e = new NFL(rng, stochastic); //false);
    }
  */

  else {
    std::cerr << "Invalid env type" << endl;
    exit(-1);
  }

  //exit(-1);

  const int numactions = e->getNumActions(); // Most agents will need this?
  float gamma = 0.99;
  const float initialvalue = 0.0; //60.0; //0.0; //0.0; //20.00;

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

  // default for gridworld like domains
  float lambda = 0.05;

  // higher lambda for stocks
  if (strcmp(argv[P_ENV], "stocks43") == 0 ||
      strcmp(argv[P_ENV], "stocks32") == 0 ||
      strcmp(argv[P_ENV], "stocks33") == 0)
    lambda = 0.3;


  // M - # visits required for RMax
  float M = 1;

  // higher M in stochastic domains
  if (stochastic){
    M = 5;
  }

  // tree models
  // svm model
  if (modelType == SVM){
    M = 0.5;
  }
  // single tree, higher # of visits to be known (since they're visits on a leaf)
  else if (nModels == 1 && (modelType == C45TREE || modelType == STUMP ||
                            modelType == M5MULTI || modelType == M5SINGLE ||
                            modelType == M5ALLMULTI || modelType == M5ALLSINGLE ||
                            modelType == LSTMULTI || modelType == LSTSINGLE ||
                            modelType == ALLM5TYPES)){
    M = M * 2;
  }

  // really high M for e-greedy, not actually doing unknown/known
  if (exploreType == EPSILONGREEDY){
    M = 100;
  }

  float rsum = 0;
#ifdef TIME
  float tsum = 0;
#endif

  int nstates = atoi(argv[P_NSTATES]);

  if (statesPerDim.size() == 0){
    cout << "set statesPerDim to " << nstates << " for all dim" << endl;
    statesPerDim.resize(minValues.size(), nstates);
  }

  for (unsigned j = 0; j < NUMTRIALS; ++j) {

    // Construct agent here.
    Agent* agent;

    if (*argv[P_AGENT] == 'q'){
      if (PRINTS) cout << "Agent: QLearner" << endl;
      agent = new QLearner(numactions,
                           gamma,
                           initialvalue, //0.0, // initialvalue
                           0.3, // alpha
                           0.1, // epsilon
                           rng);
    }

    else if (*argv[P_AGENT] == 'd'){
      if (PRINTS) cout << "Agent: dQLearner" << endl;
      agent = new dQLearner(numactions,
                            gamma,
                            initialvalue, //0.0, // initialvalue
                            0.3, // alpha
                            0.1, // epsilon
                            history,
                            rng);
    }

    else if (*argv[P_AGENT] == 'y'){
      if (PRINTS) cout << "Agent: Dyna, actrate:" << actrate << endl;
      agent = new Dyna(numactions,
                       gamma,
                       initialvalue, //0.0, // initialvalue
                       0.3, // alpha
                       1000, // k
                       0.1, // epsilon
                       rng);
    }

    else if (*argv[P_AGENT] == 's'){
      if (PRINTS) cout << "Agent: SARSA" << endl;
      agent = new Sarsa(numactions,
                           gamma,
                           initialvalue, //0.0, // initialvalue
                           0.3, // alpha
                           0.1, // epsilon
                           0,
                           rng);
    }

    else if (*argv[P_AGENT] == 'l'){
      if (PRINTS) cout << "Agent: SARSA(lambda) lambda = 0.9" << endl;
      agent = new Sarsa(numactions,
                           gamma,
                           initialvalue, //0.0, // initialvalue
                           0.3, // alpha
                           0.1, // epsilon
                           0.9,
                           rng);
    }

    /*
    else if (*argv[P_AGENT] == 'n'){
      if (PRINTS) cout << "Agent: Nick's Fitted R-max" << endl;
      agent = new NickAgent(numactions, gamma, 
                            maxValues, minValues, rMax);
    }
    */

    /*
      else if (*argv[P_AGENT] == 'o'){
      if (PRINTS) cout << "Agent: OptimalExplorer" << endl;
      agent = new OptimalExplorer(numactions,
      gamma,
      initialvalue-20.0, //0.0, // initialvalue
      0.3, // alpha
      0.1, // epsilon
      rng);
      }
    */

    else if (*argv[P_AGENT] == 'i'){
      if (PRINTS) cout << "Agent: Interactive Manual Agent" << endl;
      agent = new ManualAgent(numactions, *argv[P_ENV]);
    }


    else if (*argv[P_AGENT] == 'a'){
      if (PRINTS) cout << "Agent: QLearner with Tile Coding" << endl;
      agent = new QwithFA(numactions,
                          gamma,
                          0.3, // alpha
                          0.5, // beta
                          0.1, // epsilon
                          minValues, maxValues,
                          rng);
    }

    else if (*argv[P_AGENT] == 'f'){
      if (PRINTS) cout << "Agent: Flat RMax M: " << M << endl;
      agent = new FlatRMax(numactions,
                           gamma,
                           M,
                           rMax,
                           rng);
    }

    else if (*argv[P_AGENT] == 't'){
      if (PRINTS) cout << "Agent: RLDT M: " << M << endl;
      agent = new RLDT(numactions,
                       gamma,
                       rMax, rRange, M,
                       minValues, maxValues,
                       rng);
    }

    else if (*argv[P_AGENT] == 'm'){
      if (PRINTS) cout << "Agent: Model Based" << endl;
      if (PRINTS) cout << " M: " << M << endl;
      if (PRINTS) cout << "Lambda: " << lambda << endl;
      if (PRINTS) cout << "Act Rate: " << actrate << " Hz, seconds: " << (1.0/actrate) << endl;
      agent = new ModelBasedAgent(numactions,
                                  gamma,
                                  rMax, rRange,
                                  modelType,
                                  exploreType,
                                  predType,
                                  nModels,
                                  plannerType,
                                  0.1, // epsilon
                                  lambda,
                                  (1.0/actrate), //0.1, //0.1, //0.01, // max time
                                  M,
                                  minValues, maxValues,
                                  statesPerDim,//0,
                                  history, b,
                                  depTrans, relTrans, featPct, stochastic, episodic,
                                  rng);
    }

    else if (*argv[P_AGENT] == 'c'){
      if (PRINTS) cout << "Agent: ContinuousModel Based" << endl;
      if (PRINTS) cout << " M: " << M << endl;
      if (PRINTS) cout << "Lambda: " << lambda << endl;
      if (PRINTS) cout << "Act Rate: " << actrate << " Hz, seconds: " << (1.0/actrate) << endl;

      agent = new ModelBasedAgent(numactions,
                                  gamma,
                                  rMax, rRange,
                                  modelType,
                                  exploreType,
                                  predType,
                                  nModels,
                                  plannerType,
                                  0.1, // epsilon
                                  lambda,
                                  (1.0/actrate), //0.1, //0.1, //0.01, // max time
                                  M,
                                  minValues, maxValues,
                                  statesPerDim,
                                  history, b,
                                  depTrans, relTrans, featPct, stochastic, episodic,
                                  rng);
    }

    else if (*argv[P_AGENT] == 'p'){
      if (PRINTS) cout << "Agent: Saved Policy" << endl;
      agent = new SavedPolicy(numactions,"policy.pol");
    }

    /*
      else if (*argv[P_AGENT] == 'h'){
      if (PRINTS) cout << "Agent: Hand Coded" << endl;
      if (*argv[P_ENV] == 'f')
      agent = new HandCodedFuel(numactions);
      else if (*argv[P_ENV] == 'x')
      agent = new HandCodedExplore(numactions);
      else{
      std::cerr << "Hand coded not valid for env " << *argv[P_ENV] << endl;
      exit(-1);
      }
      }

      else if (*argv[P_AGENT] == 's'){
      agent = new TestRMax(numactions,
      gamma,
      M,
      rMax,
      rng);
      }
    */

    else {
      std::cerr << "ERROR: Invalid agent type" << endl;
      exit(-1);
    }

    int istep = 0;

#ifdef TIME
    if (*argv[P_AGENT] == 'm')
      ((ModelBasedAgent*)agent)->TIMEDEBUG = true;
    if (*argv[P_AGENT] == 't')
      ((RLDT*)agent)->TIMEDEBUG=true;
#endif

    // start discrete agent if we're discretizing (if nstates > 0 and not agent type 'c')
    int totalStates = 1;
    Agent* a2 = agent;
    if (nstates > 0 && *argv[P_AGENT] != 'c'){
      totalStates = powf(nstates,minValues.size());
      if (PRINTS) cout << "Discretize with " << nstates << ", total: " << totalStates << endl;
      //agent = new DiscretizationAgent(discAmt, a2, *argv[P_AGENT],
      //                              minValues, maxValues, PRINTS);
      agent = new DiscretizationAgent(statesPerDim, a2, *argv[P_AGENT],
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

    unsigned totalEpisodes = 0;

    // STEP BY STEP DOMAIN
    if (!episodic){

      // performance tracking
      unsigned counter = 0;
      float sum = 0;
      int steps = 0;
      float trialSum = 0.0;

      int a = 0;
      float r = 0;

      //////////////////////////////////
      // non-episodic
      //////////////////////////////////
      for (unsigned i = 0; i < MAXEPISODES; ++i){

        //std::cerr << "Trial " << j << ", episode " << i << ": ";
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

#ifdef VISUALIZE
        logExp(&vout, i, i, es, a, r);
#endif

        // update performance
        sum += r;
        ++counter;
        ++steps;
        istep++;

        // update performance info
#ifdef LOGDATA
        rdata[j][i] = r;
        sdata[j][i] = steps;
        rsums[i] += r;
        ssums[i] += steps;
#endif

        //  if (i % 1000 == 0)
        std::cerr << r << endl;

        if ((i+1)%10000 == 0){
          string filename("radio-policy");
          std::ostringstream epstring;
          epstring << (i+1);
          filename += argv[P_SEED];
          filename += "-";
          filename += argv[P_MODEL];
          filename += "-";
          filename += argv[P_ENV];
          filename += "-";
          filename += epstring.str();
          filename += ".pol";
          agent->savePolicy(filename.c_str());
        }

      }
      ///////////////////////////////////

      rsum += sum;
      trialSum += sum;
      if (PRINTS) cout << "Rsum(trial " << j << "): " << trialSum << " Avg: "
                       << (rsum / (float)(j+1))<< endl;

    }

    // EPISODIC DOMAINS
    else {

#ifdef TIMELOG
      double initTime = getSeconds();
      int lastTimeActions = 0;
      int nTimes = 0;
      int nActions = 0;
#endif
#ifdef EPTIME
      double initTime = getSeconds();
      int nTimes = 0;
#endif

      //      double initTime = getSeconds();
      float timeRSum = 0.0;
      float actRSum = 0.0;
      int totalActions = 0;

      //////////////////////////////////
      // episodic
      //////////////////////////////////
      for (unsigned i = 0; i < NUMEPISODES; ++i) {

        // performance tracking
        unsigned counter = 0;
        float sum = 0;
        int steps = 0;

        // first action
        std::vector<float> es = e->sensation();
        int a = agent->first_action(es);
        float r = e->apply(a);

#ifdef VISUALIZE
        // last episode
        if (i == NUMEPISODES-1)
          logExp(&vout, i, steps, es, a, r);
#endif

        // update performance
        sum += r;
        ++counter;
        ++steps;

        timeRSum += r;
        actRSum += r;
        totalActions++;

        while (!e->terminal() && counter < MAXSTEPS) {

          // perform an action
          es = e->sensation();
          a = agent->next_action(r, es);
          r = e->apply(a);

#ifdef VISUALIZE
          // last episode
          if (i == NUMEPISODES-1)
            logExp(&vout, i, steps, es, a, r);
#endif


          // update performance info
          sum += r;
          ++counter;
          ++steps;
          istep++;
          totalActions++;

          timeRSum += r;
          actRSum += r;

#ifdef TIMELOG
          // check if time or # actions is up
          if ((getSeconds() - initTime) > ((nTimes+1)*10.0)){
            // get cumulative reward since last
            timeRewards[nTimes] = timeRSum;
            // reset some things
            nTimes++;
            cout << (nTimes*10.0) << " sec passed. "
                 << (totalActions-lastTimeActions) << " actions. RSum: "
                 << timeRSum << endl;
            lastTimeActions = totalActions;
            timeRSum = 0;
          }
          if (totalActions >= ((nActions+1)*100)){
            // get cumulative reward since last
            actRewards[nActions] = actRSum;
            // reset some things
            nActions++;
            cout << (nActions*100.0) << " actions passed ("
                 << totalActions << "). RSum: "
                 << actRSum << endl;
            actRSum = 0;
          }
#endif //TIMELOG


#ifdef EPTIME
          if ((getSeconds() - initTime) > ((nTimes+1)*30.0)){
            if (nTimes < 5000)
              episodeTimes[nTimes] = i;
            nTimes++;
          }
#endif // EPTIME

        }

#ifdef VISUALIZE
        // last episode
        if (i == NUMEPISODES-1)
          logExp(&vout, i, steps, e->sensation(), a, r);
#endif

        // terminal/last state
        if (e->terminal()){
          agent->last_action(r);
        }else{
          agent->next_action(r, e->sensation());
        }

        // track model error
        //trackModelError(rng, agent, argv[2], i);

        // update performance info
#ifdef LOGDATA
        rdata[j][i] = sum;
        sdata[j][i] = steps;
        rsums[i] += sum;
        ssums[i] += steps;
#endif

        istep++;
        e->reset();
        //std::cerr << sum << "\t" << steps <<"\n";
        std::cerr << sum << endl;
        //std::cerr << i << "\t" << (getSeconds()-initTime) << "\t" << sum << endl;

        rsum += sum;

        // save policy
        /*
          if ((i+1)%100 == 0){
          string filename("msd-policy");
          std::ostringstream epstring;
          epstring << (i+1);
          filename += argv[P_SEED];
          filename += "-";
          filename += argv[P_NSTATES];
          filename += "-";
          filename += epstring.str();
          filename += ".pol";
          agent->savePolicy(filename.c_str());
          }*/

        // print some info about exploration
        /*
          if (*argv[P_ENV] == 'f' || *argv[P_ENV] == 'd'){
          ((FuelRooms*)e)->printVisits();
          }

          if (*argv[P_ENV] == 'f' || *argv[P_ENV] == 'd' ||
          *argv[P_ENV] == 'b' || *argv[P_ENV] == 'v' ||
          *argv[P_ENV] == 'x' || *argv[P_ENV] == 'n' ||
          *argv[P_ENV] == 'a'){
          if ((i+1) % 50 == 0 || (i+1) == 10 || (i+1) == 20 || (i+1) == 5){

          //cout << "print visitmap ep: " << i << endl;
          string filename("visitmap.");

          std::ostringstream expstring2;
          expstring2 << exploreType;

          std::ostringstream numstring2;
          numstring2 << nModels;

          std::ostringstream seedstring2;
          seedstring2 << std::atoi(argv[P_SEED]);

          std::ostringstream epstring2;
          epstring2 << (i+1);

          filename += *argv[1];
          filename += ".";
          filename += *argv[2];
          filename += ".";
          filename += *argv[3];
          filename += ".";
          filename += expstring2.str();
          filename += ".";
          filename += *argv[5];
          filename += ".";
          filename += *argv[6];
          filename += ".";
          filename += *argv[8];
          filename += ".";
          filename += numstring2.str();
          filename += ".";
          filename += seedstring2.str();
          filename += ".";
          filename += epstring2.str();

          if (*argv[P_ENV] == 'f' || *argv[P_ENV] == 'd' ||
          *argv[P_ENV] == 'a'){
          ((FuelRooms*)e)->printVisitMap(filename);
          }
          else if (*argv[P_ENV] == 'x'){
          ((Explore*)e)->printVisitMap(filename);
          }
          else {
          ((VaryingGrid*)e)->printVisitMap(filename);
          }
          }
          }
        */

#ifdef TIME
        if (*argv[P_AGENT] == 'm')
          tsum += ((ModelBasedAgent*)agent)->planningTime;
        else if (*argv[P_AGENT] == 't')
          tsum += ((RLDT*)agent)->planningTime;
#endif

        totalEpisodes++;

      }

      //if (PRINTS) cout << "Rsum(trial " << j << "): " << trialSum << " Avg: "
      //   << (rsum / (float)(j+1))<< endl;
#ifdef TIME
      if (PRINTS) cout << "Tsum: " << (tsum / (float)(j+1))<< endl;
#endif

    }

    if (NUMTRIALS > 1) delete agent;

  }

  if (PRINTS) cout << "Avg Rsum: " << (rsum / (float)NUMTRIALS) << endl;
#ifdef TIME
  if (PRINTS) cout << "Avg Tsum: " << (tsum / (float)NUMTRIALS) << endl;
#endif

#ifdef LOGDATA
  /*
  // print reward/steps to screen
  for (unsigned i = 0; i < NUMEPISODES; ++i) {
  for (unsigned j = 0; j < NUMTRIALS; ++j){

  // Output rewards
  if (PRINTS) cout << rdata[j][i] << "\t";

  // Output # steps
  //if (PRINTS) cout << sdata[j][i] << "\t";

  }
  if (PRINTS) cout << "\n";
  }
  */
#endif

#ifdef VISUALIZE
  vout.close();
#endif


#ifdef LOGDATA
  // print reward to a file
  // get filenames
  cout << "log data get file names" << endl << flush;

  std::string rstring = "";

  std::ostringstream numstring2;
  numstring2 << nModels;

  rstring += argv[1];
  rstring += ".";
  rstring += argv[2];
  rstring += ".";
  rstring += argv[3];
  rstring += ".";
  rstring += expstring2.str();
  rstring += ".";
  rstring += argv[5];
  rstring += ".";
  rstring += argv[6];
  rstring += ".";
  rstring += argv[8];
  rstring += ".";
  rstring += numstring2.str();
  rstring += ".";
  rstring += argv[10];
  rstring += ".";
  rstring += argv[11];
  rstring += ".";
  rstring += argv[12];
  rstring += ".";
  rstring += argv[13];
  rstring += ".";
  rstring += argv[14];
  rstring += ".condor.err.";
  rstring += argv[7];

  std::string stepstring = rstring;
  stepstring += "-steps.all";

  if (PRINTS || true) cout << "files: " << rstring << ", " << stepstring << endl << flush;

  fstream rfile(rstring.c_str(), ios::out);
  fstream sfile(stepstring.c_str(), ios::out);

  // write data
  for (unsigned i = 0; i < NUMEPISODES; ++i) {
    //    rfile << i << "\t";
    //    sfile << i << "\t";

    for (unsigned j = 0; j < NUMTRIALS; ++j){

      // Output rewards and # steps
      rfile << rdata[j][i] << "\t";
      sfile << sdata[j][i] << "\t";

      // print reward to std out
      //if (PRINTS) cout << rdata[j][i] << "\t";

    }
    rfile << "\n";
    sfile << "\n";
    //if (PRINTS) cout << "\n";
  }
  rfile.close();
  sfile.close();
#endif


#ifdef TIMELOG
  // print reward to a file
  // get filenames
  cout << "log data get file names" << endl << flush;

  std::string timestring = "";

  std::ostringstream numstring2;
  numstring2 << nModels;

  timestring += *argv[1];
  timestring += ".";
  timestring += *argv[2];
  timestring += ".";
  timestring += *argv[3];
  timestring += ".";
  timestring += expstring2.str();
  timestring += ".";
  timestring += argv[5];
  timestring += ".";
  timestring += *argv[6];
  timestring += ".";
  timestring += *argv[7];
  timestring += ".";
  timestring += *argv[8];
  timestring += ".";
  timestring += numstring2.str();
  timestring += ".";
  timestring += argv[10];
  timestring += ".";
  timestring += argv[11];
  timestring += ".";
  timestring += argv[12];
  timestring += ".";
  timestring += argv[13];
  timestring += ".";
  timestring += argv[14];
  timestring += ".desc";

  std::string actstring = timestring;
  timestring += ".10s";
  actstring += ".100a";

  if (PRINTS || true) cout << "files: " << timestring << ", " << actstring << endl << flush;

  fstream tfile(timestring.c_str(), ios::out);
  fstream afile(actstring.c_str(), ios::out);

  // write data
  for (int i = 0; i < 2000; i++){
    if (timeRewards[i] == 0 && actRewards[i] == 0)
      break;
    tfile << timeRewards[i] << endl;
    afile << actRewards[i] << endl;
  }

  tfile.close();
  afile.close();
#endif


#ifdef EPTIME
  // print reward to a file
  // get filenames
  cout << "log data get file names" << endl << flush;

  std::string timestring = "";

  std::ostringstream numstring2;
  numstring2 << nModels;

  timestring += *argv[1];
  timestring += ".";
  timestring += *argv[2];
  timestring += ".";
  timestring += *argv[3];
  timestring += ".";
  timestring += expstring2.str();
  timestring += ".";
  timestring += argv[5];
  timestring += ".";
  timestring += *argv[6];
  timestring += ".";
  timestring += *argv[7];
  timestring += ".";
  timestring += *argv[8];
  timestring += ".";
  timestring += numstring2.str();
  timestring += ".";
  timestring += argv[10];
  timestring += ".";
  timestring += argv[11];
  timestring += ".";
  timestring += argv[12];
  timestring += ".";
  timestring += argv[13];
  timestring += ".";
  timestring += argv[14];
  timestring += ".eptimes";


  if (PRINTS || true) cout << "files: " << timestring << endl << flush;

  fstream tfile(timestring.c_str(), ios::out);

  // write data
  bool complete = false;
  for (int i = 0; i < 5000; i++){
    if (!complete and episodeTimes[i] > 0)
      complete = true;
    if (complete && episodeTimes[i] == 0)
      tfile << NUMEPISODES << endl;
    else
      tfile << episodeTimes[i] << endl;
  }

  tfile.close();
#endif


  /*

  // print averages to a file

  // get filenames
  std::string rstringavg = "-reward.avg";
  rstringavg.insert(0, 1, expstring2.str());
  rstringavg.insert(0, 1, *argv[3]);
  rstringavg.insert(0, 1, *argv[2]);
  rstringavg.insert(0, 1, *argv[1]);

  std::string stepstringavg = "-steps.avg";
  stepstringavg.insert(0, 1, expstring2.str());
  stepstringavg.insert(0, 1, *argv[3]);
  stepstringavg.insert(0, 1, *argv[2]);
  stepstringavg.insert(0, 1, *argv[1]);

  fstream rfileavg(rstringavg.c_str(), ios::out);
  fstream sfileavg(stepstringavg.c_str(), ios::out);

  // write data
  for (unsigned i = 0; i < NUMEPISODES; ++i) {

  // Output average rewards and # steps
  rfileavg << i << "\t" << (rsums[i] / (float)NUMTRIALS) << "\n";
  sfileavg << i << "\t" << (ssums[i] / (float)NUMTRIALS) << "\n";

  }

  rfileavg.close();
  sfileavg.close();
  */

  /* Don't save per step for now

  // filename
  std::string perstepavg = "-perstep.avg";
  perstepavg.insert(0, 1, expstring2.str());
  perstepavg.insert(0, 1, *argv[3]);
  perstepavg.insert(0, 1, *argv[2]);
  perstepavg.insert(0, 1, *argv[1]);

  fstream perstepavgf(perstepavg.c_str(), ios::out);

  // write data
  for (unsigned i = 0; i < (NUMEPISODES * MAXSTEPS); ++i) {

  // Output average rewards and # steps
  perstepavgf << i << "\t" << (rperstepsums[i] / (float)NUMTRIALS) << "\n";

  }

  perstepavgf.close();

  // filename
  std::string perstepall = "-perstep.all";
  perstepall.insert(0, 1, expstring2.str());
  perstepall.insert(0, 1, *argv[3]);
  perstepall.insert(0, 1, *argv[2]);
  perstepall.insert(0, 1, *argv[1]);

  fstream perstepallf(perstepall.c_str(), ios::out);

  for (unsigned i = 0; i < (NUMEPISODES * MAXSTEPS); ++i) {
  perstepallf << i << "\t";

  for (unsigned j = 0; j < NUMTRIALS; ++j){

  perstepallf << rperstep[j][i] << "\t";

  }
  perstepallf << "\n";
  }
  perstepallf.close();

  */

  // clean stuff up
  delete e;

}



void logExp(ofstream *vout, int i, int steps, std::vector<float> es, int a, float r){
  *vout << i << "\t" << steps << "\t";
  for (unsigned jj = 0; jj < es.size(); jj++){
    *vout << es[jj] << "\t";
  }
  *vout << a << "\t" << r << endl;
}




double getSeconds(){
  struct timezone tz;
  timeval timeT;
  gettimeofday(&timeT, &tz);
  return  timeT.tv_sec + (timeT.tv_usec / 1000000.0);
}
