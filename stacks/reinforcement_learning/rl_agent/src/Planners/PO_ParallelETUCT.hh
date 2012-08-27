/** \file PO_ParallelETUCT.hh
    Defines my real-time model-based RL architecture which uses UCT with eligiblity traces for planning. This version of UCT plans over states augmented with k-action histories.
    The modified version of UCT used is presented in:
    L. Kocsis and C. Szepesv´ari, "Bandit based monte-carlo planning," in
    ECML-06. Number 4212 in LNCS. Springer, 2006, pp. 282-293.
    The real-time architecture is presented in: 
    Hester, Quinlan, and Stone, "A Real-Time Model-Based Reinforcement Learning Architecture for Robot Control", arXiv 1105.1749, 2011.
    \author Todd Hester
*/


#ifndef _PO_ParallelETUCT_HH_
#define _PO_ParallelETUCT_HH_

#include <rl_common/Random.h>
#include <rl_common/core.hh>
#include <rl_common/ExperienceFile.hh>

#include "../Models/FactoredModel.hh"
#include "../Models/C45Tree.hh"

#include <ext/hash_map>
#include <set>
#include <vector>
#include <map>
#include <sstream>
#include <deque>

/** Parallel thread that continually does uct search from agent's current state. */
void* poParallelSearchStart(void* arg);

/** Thread that loops, continually updating model with new experiences. */
void* poParallelModelLearningStart(void* arg);

/** This class defines my real-time model-based RL architecture, which does model learning, planning, and acting on 3 separate threads. It uses a modified version of UCT for planning, which plans on a model using Monte Carlo rollouts. Unlike the original UCT, it does not separate values by tree depth, and it incorporates eligibility traces. This version plans over states augmented with k-action histories, for delayed or partially observable domains. */
class PO_ParallelETUCT: public Planner {
public:

  /** Standard constructor
      \param numactions, numactions in the domain
      \param gamma discount factor
      \param rrange range of one-step rewards in the domain
      \param lambda for use with eligibility traces
      \param MAX_ITER maximum number of MC rollouts to perform
      \param MAX_TIME maximum amount of time to run Monte Carlo rollouts
      \param MAX_DEPTH maximum depth to perform rollout to
      \param modelType specifies model type
      \param featmax maximum value of each feature
      \param featmin minimum value of each feature
      \param statesPerDim # of values to discretize each feature into
      \param trackActual track actual real-valued states (or just discrete states)
      \param historySize # of previous actions to use for delayed domains
      \param rng random number generator
  */
  PO_ParallelETUCT(int numactions, float gamma, float rrange, float lambda,
                   int MAX_ITER, float MAX_TIME, int MAX_DEPTH,  int modelType,
                   const std::vector<float> &featmax, const std::vector<float> &featmin,
                   const std::vector<int> &statesPerDim, bool trackActual, int historySize, 
                   Random rng = Random());
  
  /** Unimplemented copy constructor: internal state cannot be simply
      copied. */
  PO_ParallelETUCT(const PO_ParallelETUCT &);

  virtual ~PO_ParallelETUCT();

  virtual void setModel(MDPModel* model);
  virtual bool updateModelWithExperience(const std::vector<float> &last, 
                                         int act, 
                                         const std::vector<float> &curr, 
                                         float reward, bool term);
  virtual void planOnNewModel();
  virtual int getBestAction(const std::vector<float> &s);

  virtual void setSeeding(bool seed);
  virtual void setFirst();

  bool PLANNERDEBUG;
  bool POLICYDEBUG; //= false; //true;
  bool MODELDEBUG;
  bool ACTDEBUG;
  bool UCTDEBUG;
  bool PTHREADDEBUG;
  bool ATHREADDEBUG;
  bool MTHREADDEBUG;
  bool TIMINGDEBUG;
  bool REALSTATEDEBUG;
  bool HISTORYDEBUG;

  /** MDPModel that we're using with planning */
  MDPModel* model;
  
  /** Copy of our model used when updating the model, so the original can still be queried by the other threads. */
  MDPModel* modelcopy;

  /** The implementation maps all sensations to a set of canonical
      pointers, which serve as the internal representation of
      environment state. */
  typedef const std::vector<float> *state_t;



  ///////////////////////////////
  // parallel stuff
  ///////////////////////////////

  // tell if thread started
  bool modelThreadStarted;
  bool planThreadStarted;

  // the threads
  /** Thread that performs planning using UCT. */
  pthread_t planThread;
  
  /** Thread that performs model updates. */
  pthread_t modelThread;

  // some variables that are locked
  /** List of experiences from action thread to be added into model */
  std::vector<experience> expList;
  
  /** Agent's current discrete state that UCT rollouts should start from. */
  state_t discPlanState;
  
  /** Agent's current actual real-valued state that UCT rollouts should start from. */
  std::vector<float> actualPlanState;
  
  /** Canonical pointer to agent's current state that UCT rollouts should start from. */
  state_t startState;
  bool doRandom;

  // lock over simple objects
  /** Mutex around the last frame that the model was updated. */
  pthread_mutex_t update_mutex;
  /** Mutex around the counter of how many actions the agent has taken. */
  pthread_mutex_t nactions_mutex;
  /** Mutex around the agent's current state to plan from. */
  pthread_mutex_t plan_state_mutex;
  /** Mutex around the model. */
  pthread_mutex_t model_mutex;
  /** Mutex around the list of experiences to be added to the model. */
  pthread_mutex_t list_mutex;
  /** Mutex around the history of previous actions of the agent. */
  pthread_mutex_t history_mutex;

  /** Mutex around the set of states. */
  pthread_mutex_t statespace_mutex;

  // condition for when list is updated
  pthread_cond_t list_cond; 


  /** Perform UCT/Monte Carlo rollout from the given state.
      If terminal or at depth, return some reward.
      Otherwise, select an action based on UCB.
      Simulate action to get reward and next state.
      Call search on next state at depth+1 to get reward return from there on.
      Update q value towards new value: reward + gamma * searchReturn
      Update visit counts for confidence bounds
      Return q
      
      From "Bandit Based Monte Carlo Planning" by Kocsis and Szepesv´ari.
  */
  float uctSearch(const std::vector<float> &actS, state_t state, int depth);
  
  /** Select a random previously visited state. */
  std::vector<float> selectRandomState();
  
  /** Set a flag that we are in between episodes. */
  void setBetweenEpisodes();
  
  /** Start the parallel model learning thread. */
  void parallelModelLearning();
  
  /** Start the parallel UCT planning thread. */
  void parallelSearch();
  
  /** Load a policy from a file. */
  void loadPolicy(const char* filename);
  
  /** Output value function to a file */
  void logValues(ofstream *of, int xmin, int xmax, int ymin, int ymax);
  
  /** Add two vectors together. */
  std::vector<float> addVec(const std::vector<float> &a, const std::vector<float> &b);
  
  /** Subtract two vectors. */
  std::vector<float> subVec(const std::vector<float> &a, const std::vector<float> &b);

protected:


  struct state_info;
  struct model_info;

  /** A struct that contains a vector of possible next state samples, weighted by their probabilities. */
  struct state_samples {
    std::vector<state_t> samples;
  };

  /** State info struct. Maintains visit counts, models, and q-values for state-actions. */
  struct state_info {
    int id;

    // experience data
    std::vector<int> visits;
    int totalVisits;

    // data filled in from models
    StateActionInfo* model;

    // q values from policy creation
    std::vector<float> Q;

    // uct experience data
    int uctVisits;
    std::vector<int> uctActions;
    int visited;

    // needs update
    bool needsUpdate;

    // mutex for model info, samples, everything else
    pthread_mutex_t statemodel_mutex;
    pthread_mutex_t stateinfo_mutex;

  };

  /** Initialize state info struct */
  void initStateInfo(state_t s,state_info* info, int id);
  
  /** Produces a canonical representation of the given sensation.
      \param s The current sensation from the environment.
      \return A pointer to an equivalent state in statespace. */
  state_t canonicalize(const std::vector<float> &s);

  /** Delete a state_info struct */
  void deleteInfo(state_info* info);
  
  /** Compuate a policy from a model */
  void createPolicy();
  
  /** Print information for each state. */
  void printStates();
  
  /** Calculate which states are reachable from states the agent has actually visited. */
  void calculateReachableStates();
  
  /** Remove states from set that were deemed unreachable. */
  void removeUnreachableStates();

  /** Update the state_info copy of the model for the given state-action from the MDPModel */
  void updateStateActionFromModel(state_t s, int a, state_info* info);
  
  /** Update the state_info copy of the model for the given state-action and k-action history from the MDPModel. */
  void updateStateActionHistoryFromModel(const std::vector<float> modState, int a, StateActionInfo *newModel);

  /** Get the current time in seconds */
  double getSeconds();

  // uct stuff
  /** Reset UCT visit counts to some baseline level (to decrease our confidence in q-values because model has changed. */
  void resetAndUpdateStateActions();

  /** Return a sampled state from the next state distribution of the model. 
      Simulate the next state from the given state, action, and possibly history of past actions. */
  std::vector<float> simulateNextState(const std::vector<float> &actS, state_t state, state_info* info, int action, float* reward, bool* term);
  
  /** Select UCT action based on UCB1 algorithm. */
  int selectUCTAction(state_info* info);
  
  /** Canonicalize all the next states predicted by this model. */
  void canonNextStates(StateActionInfo* modelInfo);
  
  /** Initialize the states for this domain (based on featmin and featmax) */
  void initStates();
  
  /** Fill in a state based on featmin and featmax */
  void fillInState(std::vector<float>s, int depth);

  virtual void savePolicy(const char* filename);
  
  /** Return a discretized version of the input state. */
  std::vector<float> discretizeState(const std::vector<float> &s);

private:

  /** Set of all distinct sensations seen.  Pointers to elements of
      this set serve as the internal representation of the environment
      state. */
  std::set<std::vector<float> > statespace;

  /** Hashmap mapping state vectors to their state_info structs. */
  std::map<state_t, state_info> statedata;

  std::vector<float> featmax;
  std::vector<float> featmin;

  /** Current history of previous actions. */
  std::deque<float> saHistory;

  state_t prevstate;
  int prevact;
  state_info* previnfo;

  double planTime;
  double initTime;
  double setTime;
  bool seedMode;

  int nstates;
  int nsaved;
  int nactions;
  int lastUpdate;

  bool timingType;

  const int numactions;
  const float gamma;
  const float rrange;
  const float lambda;

  const int MAX_ITER;
  const float MAX_TIME;
  const int MAX_DEPTH;
  const int modelType;
  const std::vector<int> &statesPerDim;
  const bool trackActual;
  const int HISTORY_SIZE;
  const int HISTORY_FL_SIZE;

  ExperienceFile expfile;
};

#endif
