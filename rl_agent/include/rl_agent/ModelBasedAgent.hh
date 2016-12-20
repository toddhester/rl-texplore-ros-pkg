/** \file ModelBasedAgent.hh
    Defines the ModelBasedAgent class
    \author Todd Hester
*/

#ifndef _MODELBASED_HH_
#define _MODELBASED_HH_

#include <rl_common/Random.h>
#include <rl_common/core.hh>

#include <set>
#include <vector>
#include <map>

/** Code for a model based agent, that can use any model and planner that meet the interface requirements */
class ModelBasedAgent: public Agent {
public:
  /** Standard constructor
      \param numactions The number of possible actions
      \param gamma The discount factor
      \param rmax max reward value, given out for unknown states
      \param rrange range between max and min reward in domain
      \param modelType specifies model type
      \param exploreType specifies exploration type
      \param predType specifies how to combine multiple models
      \param nModels number of models in ensemble
      \param plannerType specifies planner type
      \param epsilon used for epsilon-greedy action selection
      \param lambda used for eligibility traces in uct planning
      \param MAX_TIME amount of time the uct planners are given for planning
      \param m # visits required for a state to become known when doing rmax exploration
      \param featmin min values of each feature
      \param featmax max values of each feature
      \param statesPerDim # of values to discretize each feature into
      \param history # of previous actions to use for delayed domains
      \param b\v bonus reward used when models disagree
      \param n bonus reward used for novel states
      \param depTrans assume dependent or indep. feature transitions
      \param relTrans model transitions relatively vs absolutely
      \param featPct pct of feature to remove from set used for each split in tree
      \param stoch is the domain stochastic?
      \param episodic is the domain episodic?
      \param rng Initial state of the random number generator to use */
  ModelBasedAgent(int numactions, float gamma, float rmax, float rrange, 
                  int modelType, int exploreType, 
                  int predType, int nModels, int plannerType,
                  float epsilon, float lambda, float MAX_TIME,
                  float m, const std::vector<float> &featmin, 
                  const std::vector<float> &featmax,
                  int statesPerDim, int history, float v, float n,
                  bool depTrans, bool relTrans, float featPct,
                  bool stoch, bool episodic, Random rng = Random());

  /** Standard constructor 
      \param numactions The number of possible actions
      \param gamma The discount factor
      \param rmax max reward value, given out for unknown states
      \param rrange range between max and min reward in domain
      \param modelType specifies model type
      \param exploreType specifies exploration type
      \param predType specifies how to combine multiple models
      \param nModels number of models in ensemble
      \param plannerType specifies planner type
      \param epsilon used for epsilon-greedy action selection
      \param lambda used for eligibility traces in uct planning
      \param MAX_TIME amount of time the uct planners are given for planning
      \param m # visits required for a state to become known when doing rmax exploration
      \param featmin min values of each feature
      \param featmax max values of each feature
      \param statesPerDim # of values to discretize each feature into
      \param history # of previous actions to use for delayed domains
      \param b bonus reward used when models disagree
      \param depTrans assume dependent or indep. feature transitions
      \param relTrans model transitions relatively vs absolutely
      \param featPct pct of feature to remove from set used for each split in tree
      \param stoch is the domain stochastic?
      \param episodic is the domain episodic?
      \param rng Initial state of the random number generator to use*/
  ModelBasedAgent(int numactions, float gamma, float rmax, float rrange, 
                  int modelType, int exploreType, 
                  int predType, int nModels, int plannerType,
                  float epsilon, float lambda, float MAX_TIME,
                  float m, const std::vector<float> &featmin, 
                  const std::vector<float> &featmax,
                  std::vector<int> statesPerDim, int history, float v, float n,
                  bool depTrans, bool relTrans, float featPct,
                  bool stoch, bool episodic, Random rng = Random());
  
  /** Init params for both constructors */
  void initParams();

  /** Unimplemented copy constructor: internal state cannot be simply
      copied. */
  ModelBasedAgent(const ModelBasedAgent &);

  virtual ~ModelBasedAgent();

  virtual int first_action(const std::vector<float> &s);
  virtual int next_action(float r, const std::vector<float> &s);
  virtual void last_action(float r);
  virtual void seedExp(std::vector<experience> seeds);
  virtual void setDebug(bool d);
  virtual void savePolicy(const char* filename);

  /** Output value function to a file */
  void logValues(ofstream *of, int xmin, int xmax, int ymin, int ymax);

  bool AGENTDEBUG;
  bool POLICYDEBUG; //= false; //true;
  bool ACTDEBUG;
  bool SIMPLEDEBUG;
  bool TIMEDEBUG;

  bool seeding;

  /** Model that we're using */
  MDPModel* model;

  /** Planner that we're using */
  Planner* planner;

  float planningTime;
  float modelUpdateTime;
  float actionTime;

  std::vector<float> featmin;
  std::vector<float> featmax;

protected:

  /** The implementation maps all sensations to a set of canonical
      pointers, which serve as the internal representation of
      environment state. */
  typedef const std::vector<float> *state_t;

  /** Saves state and action to use for update on next action */
  void saveStateAndAction(const std::vector<float> &s, int act);

  /** Select action from the given state */
  int chooseAction(const std::vector<float> &s);

  /** Initialize the model with the given # of features */
  void initModel(int nfactors);

  /** Initialize the planner */
  void initPlanner();

  /** Update the agent with the new s,a,s',r experience */
  void updateWithNewExperience(const std::vector<float> &last, 
                               const std::vector<float> & curr, 
                               int lastact, float reward, bool term);

  /** Get the current time in seconds */
  double getSeconds();

private:

  /** Previous state */
  std::vector<float> prevstate;
  /** Previous action */
  int prevact;

  int nstates;
  int nactions; 
 
  bool modelNeedsUpdate;
  int lastUpdate;

  int BATCH_FREQ;

  bool modelChanged;

  const int numactions;
  const float gamma;

  const float rmax;
  const float rrange;
  const float qmax;
  const int modelType;
  const int exploreType;
  const int predType;
  const int nModels;
  const int plannerType;

  const float epsilon;
  const float lambda;
  const float MAX_TIME;

  const float M;
  const std::vector<int> statesPerDim;
  const int history;
  const float v;
  const float n;
  const bool depTrans;
  const bool relTrans;
  const float featPct;
  const bool stoch;
  const bool episodic;

  Random rng;

};

#endif
