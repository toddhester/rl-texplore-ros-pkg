#ifndef _POLICYITERATION_HH_
#define _POLICYITERATION_HH_

#include <rl_common/Random.h>
#include <rl_common/core.hh>

#include <set>
#include <vector>
#include <map>


class PolicyIteration: public Planner {
public:

  /** The implementation maps all sensations to a set of canonical
      pointers, which serve as the internal representation of
      environment state. */
  typedef const std::vector<float> *state_t;


  /** Standard constructor
      \param numactions, numactions in the domain
      \param gamma discount factor
      \param maxloops
      \param max time
      \param rng random
  */
  PolicyIteration(int numactions, float gamma,
                  int MAX_LOOPS, float MAX_TIME, int modelType,
                  const std::vector<float> &featmax, 
                  const std::vector<float> &featmin,
                   const std::vector<int> &statesPerDim,
                  Random rng = Random());

  /** Unimplemented copy constructor: internal state cannot be simply
      copied. */
  PolicyIteration(const PolicyIteration &);

  virtual ~PolicyIteration();

  virtual void setModel(MDPModel* model);
  virtual bool updateModelWithExperience(const std::vector<float> &last, 
                                         int act, 
                                         const std::vector<float> &curr, 
                                         float reward, bool term);
  virtual void planOnNewModel();
  virtual int getBestAction(const std::vector<float> &s);
  virtual void savePolicy(const char* filename);

  bool PLANNERDEBUG;
  bool POLICYDEBUG; //= false; //true;
  bool MODELDEBUG;
  bool ACTDEBUG;

  /** Model that we're using */
  MDPModel* model;



protected:


  struct state_info;
  struct model_info;

  /** State info struct */
  struct state_info {
    int id;

    int stepsAway;
    bool fresh;

    // experience data
    std::vector<int> visits;

    // data filled in from models
    StateActionInfo* modelInfo;

    //std::map<state_t, std::vector<float> > P;
    //std::vector<float> R;
    //std::vector<bool> known;

    // q values from policy creation

    float value;
    int bestAction;

  };



  // various helper functions that we need
  void initStateInfo(state_info* info);
  
  /** Produces a canonical representation of the given sensation.
      \param s The current sensation from the environment.
      \return A pointer to an equivalent state in statespace. */
  state_t canonicalize(const std::vector<float> &s);

  // Operational functions
  void deleteInfo(state_info* info);
  void initNewState(state_t s);
  void createPolicy();
  void printStates();
  void calculateReachableStates();
  void removeUnreachableStates();

  // functions to update our models and get info from them
  void updateStatesFromModel();
  void updateStateActionFromModel(const std::vector<float> &state, int j);

  double getSeconds();

  // for policy iter
  void policyEvaluation();
  float getActionValue(state_t s, state_info* info, int act);
  bool policyImprovement();
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

  std::vector<float> prevstate;
  int prevact;

  double planTime;
  int nstates;
  int nactions; 
  
  int MAX_STEPS;
  bool timingType;

  const int numactions;
  const float gamma;

  const int MAX_LOOPS;
  const float MAX_TIME;
  const int modelType;
  const std::vector<int> &statesPerDim;

};

#endif
