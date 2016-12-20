#ifndef _RLCORE_H_
#define _RLCORE_H_

#include "Random.h"
#include <vector>
#include <map>


/** \file
    Fundamental declarations for the universal concepts in the
    reinforcement learning framework.
    \author Nick Jong
    \author Todd Hester
*/


// types of models
#define RMAX        0
#define TABULAR     0
#define SLF         1
#define C45TREE     2
#define SINGLETREE  3
#define SVM         4
#define STUMP       5
#define M5MULTI     6
#define M5SINGLE    7
#define M5ALLMULTI  8
#define M5ALLSINGLE 9 
#define LSTMULTI    10
#define LSTSINGLE   11
#define ALLM5TYPES  12
#define GPREGRESS   13
#define GPTREE      14

const std::string modelNames[] = {
  "Tabular",
  "SLF",
  "C4.5 Tree",
  "Single Tree",
  "SVM",
  "Stump",
  "M5 Tree",
  "M5 Tree",
  "M5 Tree",
  "M5 Tree",
  "LS Tree",
  "LS Tree",
  "M5 Combo",
  "GP Regression",
  "GP Tree"
};

// types of model combos
#define AVERAGE        1
#define WEIGHTAVG      2
#define BEST           3
#define SEPARATE       4 // sep model for planning, and forest for uncertainty

const std::string comboNames[] = {
  "Average",
  "Weighted Average",
  "Best",
  "Separate"
};

// types of exploration
#define EXPLORE_UNKNOWN    0
#define TWO_MODE           1
#define TWO_MODE_PLUS_R    2
#define CONTINUOUS_BONUS   3
#define THRESHOLD_BONUS    4
#define CONTINUOUS_BONUS_R 5
#define THRESHOLD_BONUS_R  6
#define NO_EXPLORE         7
#define GREEDY             7
#define EPSILONGREEDY      8
#define VISITS_CONF        9
#define UNVISITED_BONUS    11
#define UNVISITED_ACT_BONUS 13
#define DIFF_AND_VISIT_BONUS 16
#define NOVEL_STATE_BONUS    18
#define DIFF_AND_NOVEL_BONUS 19

const std::string exploreNames[] = {
  "Explore Unknowns",
  "Two Modes",
  "Two Models +R",
  "Continuous Bonus",
  "Threshold Bonus",
  "Continuous Bonus +R",
  "Threshold Bonus +R",
  "Greedy",
  "Epsilon-Greedy",
  "Visits Confidence",
  "Type 10",
  "Unvisited State Bonus",
  "Type 12", 
  "Unvisited Action Bonus",
  "Type 14",
  "Type 15",
  "Model Diff & Visit Bonus",
  "Type 17",
  "FeatDist Bonus",
  "Model Diff & FeatDist Bonus"
};

// types of planners
#define VALUE_ITERATION    0
#define POLICY_ITERATION   1
#define PRI_SWEEPING       2
#define UCT                3
#define ET_UCT             4
#define ET_UCT_WITH_ENV    5
#define SWEEPING_UCT_HYBRID 6
#define CMAC_PLANNER       7
#define NN_PLANNER         8
#define MOD_PRI_SWEEPING   9
#define ET_UCT_L1          10
#define UCT_WITH_L         11
#define UCT_WITH_ENV       12
#define PARALLEL_ET_UCT    13
#define ET_UCT_ACTUAL      14
#define ET_UCT_CORNERS     15
#define PAR_ETUCT_ACTUAL   16
#define PAR_ETUCT_CORNERS  17
#define POMDP_ETUCT        18
#define POMDP_PAR_ETUCT    19
#define MBS_VI             20

const std::string plannerNames[] = {
  "Value Iteration",
  "Policy Iteration",
  "Prioritized Sweeping",
  "UCT",
  "UCT",
  "UCT",
  "Sweeping UCT Hybrid",
  "CMACs",
  "NN",
  "Mod. Pri Sweeping",
  "UCT L=1",
  "UCT L",
  "UCT Env",
  "Parallel UCT",
  "Real-Valued UCT",
  "Corner UCT",
  "Parallel Real-Valued UCT",
  "Parallel Corner UCT",
  "Delayed UCT",
  "Parallel Delayed UCT",
  "Model Based Simulation - VI"
};
  


#define EPSILON   1e-5

/** Experience <s,a,s',r> struct */
struct experience {
  std::vector<float> s;
  int act;
  float reward;
  std::vector<float> next;
  bool terminal;
};

/** Training instances for prediction models */
struct modelPair {
  std::vector<float> in;
  std::vector<float> out;
};

/** Training instances for classification models */
struct classPair {
  std::vector<float> in;
  float out;
};

/** Interface for an environment, whose states can be represented as
    vectors of floats and whose actions can be represented as ints.
    Implementations of the Environment interface determine how actions
    influence sensations.  Note that this design assumes only one
    agent: it would be more accurate to name this interface
    EnvironmentAsPerceivedByOneParticularAgent. */
class Environment {
public:
  /** Provides access to the current sensation that the environment
      gives to the agent.
      \return The current sensation. */
  virtual const std::vector<float> &sensation() const = 0;

  /** Allows an agent to affect its environment.
      \param action The action the agent wishes to apply.
      \return The immediate one-step reward caused by the action. */
  virtual float apply(int action) = 0;

  /** Determines whether the environment has reached a terminal state.
      \return true iff the task is episodic and the present episode
      has ended.  Nonepisodic tasks should simply always
      return false. */
  virtual bool terminal() const = 0;

  /** Resets the internal state of the environment according to some
      initial state distribution.  Typically the user calls this only
      for episodic tasks that have reached terminal states, but this
      usage is not required. */
  virtual void reset() = 0;

  /** Returns the number of actions available in this environment.
      \return The number of actions available */
  virtual int getNumActions() = 0;

  /** Gets the minimum and maximum of the features in the environment.
   */
  virtual void getMinMaxFeatures(std::vector<float> *minFeat,
                                 std::vector<float> *maxFeat) = 0;

  /** Gets the minimum and maximum one-step reward in the domain. */
  virtual void getMinMaxReward(float *minR, float *maxR) = 0;

  /** Returns if the domain is episodic (true by default). */
  virtual bool isEpisodic(){ return true; };

  /** Get seeding experiences for agent. */
  virtual std::vector<experience> getSeedings()
  {
    std::vector<experience> e;
    return e;
  } ;

  /** Set the current state for testing purposes. */
  virtual void setSensation(std::vector<float> s){};

  virtual ~Environment() {};

};

/** Interface for an agent.  Implementations of the Agent interface
    determine the choice of actions given previous sensations and
    rewards. */
class Agent {
public:
  /** Determines the first action that an agent takes in an
      environment.  This method implies that the environment is
      currently in an initial state.
      \param s The initial sensation from the environment.
      \return The action the agent wishes to take first. */
  virtual int first_action(const std::vector<float> &s) = 0;

  /** Determines the next action that an agent takes in an environment
      and gives feedback for the previous action.  This method may
      only be called if the last method called was first_action or
      next_action.
      \param r The one-step reward resulting from the previous action.
      \param s The current sensation from the environment.
      \return The action the agent wishes to take next. */
  virtual int next_action(float r, const std::vector<float> &s) = 0;

  /** Gives feedback for the last action taken.  This method may only
      be called if the last method called was first_action or
      next_action.  It implies that the task is episodic and has just
      terminated.  Note that terminal sensations (states) are not
      represented.
      \param r The one-step reward resulting from the previous action. */
  virtual void last_action(float r) = 0;

  /** Set some debug flags on/off */
  virtual void setDebug(bool d) = 0;

  /** Use the model seeds from the environment to initialize the agent or its model */
  virtual void seedExp(std::vector<experience> seeds) {};

  /** Save the current policy to a file */
  virtual void savePolicy(const char* filename) {};

  virtual ~Agent() {};
};

/** Interface for a model that predicts a vector of floats given a vector of floats as input. */
class Model {
public:
  /** Train the model on a vector of training instances */
  virtual bool trainInstances(std::vector<modelPair> &instances) = 0;

  /** Train the model on a single training instance */
  virtual bool trainInstance(modelPair &instance) = 0;

  /** Get the model's prediction for a given input */
  virtual std::vector<float> testInstance(const std::vector<float> &in) = 0;

  virtual ~Model() {};
};

/** Interface for a classification model that predicts a class given a vector of floats as input. */
class Classifier {
public:
  /** Train the model on a vector of training instances */
  virtual bool trainInstances(std::vector<classPair> &instances) = 0;

  /** Train the model on a single training instance */
  virtual bool trainInstance(classPair &instance) = 0;

  /** Get the model's prediction for a given input */
  virtual void testInstance(const std::vector<float> &in, std::map<float, float>* retval) = 0;

  /** Get the model's confidence in its predictions for a given input. */
  virtual float getConf(const std::vector<float> &in) = 0;

  /** Get a copy of the model */
  virtual Classifier* getCopy() = 0;

  virtual ~Classifier() {};
};

/** All the relevant information predicted by a model for a given state-action.
    This includes predicted reward, next state probabilities, probability of episod termination, and model confidence.
*/
struct StateActionInfo {
  bool known;
  float reward;
  float termProb;
  int frameUpdated;

  // map from outcome state to probability
  std::map< std::vector<float> , float> transitionProbs;

  StateActionInfo(){
    known = false;
    reward = 0.0;
    termProb = 0.0;
    frameUpdated = -1;
  };
};


/** Interface for a model of an MDP. */
class MDPModel {
public:
  /** Update the MDP model with a vector of experiences. */
  virtual bool updateWithExperiences(std::vector<experience> &instances) = 0;

  /** Update the MDP model with a single experience. */
  virtual bool updateWithExperience(experience &instance) = 0;

  /** Get the predictions of the MDP model for a given state action */
  virtual float getStateActionInfo(const std::vector<float> &state, int action, StateActionInfo* retval) = 0;

  /** Get a copy of the MDP Model */
  virtual MDPModel* getCopy() = 0;
  virtual ~MDPModel() {};
};

/** Interface for planners */
class Planner {
public:
  /** Give the planner the model being used with the agent */
  virtual void setModel(MDPModel* model) = 0;

  /** Update the given model with an experience <s,a,s',r>. */
  virtual bool updateModelWithExperience(const std::vector<float>& last,
                                         int act,
                                         const std::vector<float>& curr,
                                         float reward, bool terminal) = 0;

  /** Plan a new policy suing the current model. */
  virtual void planOnNewModel() = 0;

  /** Return the best action for a given state. */
  virtual int getBestAction(const std::vector<float> &s) = 0;

  /** Save the policy to a file. */
  virtual void savePolicy(const char* filename) {};

  /** Set whether the next experiences are seeds or actual experiences from the agent. */
  virtual void setSeeding(bool seeding) {};

  /** Set if this is the first experience of the agent. */
  virtual void setFirst() {};

  /** A method to return at random one of the maximum values in the vector. 
      Such that when more than one actions are optimal, we select one of them at random.
  */
  std::vector<float>::iterator
  random_max_element(std::vector<float>::iterator start,
		     std::vector<float>::iterator end) {
    const float Q_EPSILON = 1e-4;
    
    std::vector<float>::iterator max =
    std::max_element(start, end);

    // find # within epsilon of max
    int nfound = 0;
    for (std::vector<float>::iterator it = start; it != end; it++){
      if (fabs(*it - *max) < Q_EPSILON){
        nfound++;
      }
    }
    
    // only 1: take it
    if (nfound == 1)
      return max;

    // take one of close to max at random
    for (std::vector<float>::iterator it = start; it != end; it++){
      if (fabs(*it - *max) < Q_EPSILON){
        if (rng.uniform() < (1.0 / (float)nfound)){
          return it;
        }
        nfound--;
      }
    }
    
    return max;
  };

  virtual ~Planner() {};
  
  Random rng;

};


#endif
