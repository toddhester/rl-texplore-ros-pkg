#ifndef _ETUCTCompleteModel_HH_
#define _ETUCTCompleteModel_HH_

#include "../Common/Random.h"
#include "../Common/core.hh"

#include "../Env/nfl.hh"
#include "../Env/Minesweeper.hh"
#include "../Env/stocks.hh"

#include <ext/hash_map>
#include <set>
#include <vector>
#include <map>


class ETUCTCompleteModel: public Planner {
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
  ETUCTCompleteModel(int numactions, float gamma, float rmax, float lambda,
      int MAX_ITER, float MAX_TIME, int MAX_DEPTH,
      char env,
      Random rng = Random());
  
  /** Unimplemented copy constructor: internal state cannot be simply
      copied. */
  ETUCTCompleteModel(const ETUCTCompleteModel &);

  virtual ~ETUCTCompleteModel();

  virtual void setModel(MDPModel* model);
  virtual bool updateModelWithExperience(const std::vector<float> &last, 
					 int act, 
					 const std::vector<float> &curr, 
					 float reward);
  virtual void planOnNewModel();
  virtual int getBestAction(const std::vector<float> &s);

  bool PLANNERDEBUG;
  bool POLICYDEBUG; //= false; //true;
  bool MODELDEBUG;
  bool ACTDEBUG;
  bool UCTDEBUG;

  Environment* domain;

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

    // q values from policy creation
    std::vector<float> Q;

    // uct experience data
    int uctVisits;
    std::vector<int> uctActions;
    bool visited;


  };



  // various helper functions that we need
  void initStateInfo(state_info* info);
  
  /** Produces a canonical representation of the given sensation.
      \param s The current sensation from the environment.
      \return A pointer to an equivalent state in statespace. */
  state_t canonicalize(const std::vector<float> &s);
  std::vector<float>::iterator random_max_element(std::vector<float>::iterator start,
						   std::vector<float>::iterator end);

  std::vector<float> modifyState(const std::vector<float> &s);

  // Operational functions
  void deleteInfo(state_info* info);
  void initNewState(state_t s);
  void createPolicy();
  void printStates();
  void calculateReachableStates();
  void removeUnreachableStates();

  // functions to update our models and get info from them
  void updateStateActionFromModel(state_t s, int a);

  double getSeconds();

  // uct stuff
  void resetUCTCounts();
  float uctSearch(std::vector<float> state, int depth);
  std::vector<float> simulateNextState(state_t s, int action);
  int selectUCTAction(state_info* info);
  bool simulateReward(state_t s, int action, float* reward);

  void setEnvironment();

private:

  /** Set of all distinct sensations seen.  Pointers to elements of
      this set serve as the internal representation of the environment
      state. */
  std::set<std::vector<float> > statespace;

  /** Hashmap mapping state vectors to their state_info structs. */
  std::map<state_t, state_info> statedata;


  int nstates;
  int nactions; 

  const int numactions;
  const float gamma;
  const float rmax;
  const float lambda;

  const int MAX_ITER;
  const float MAX_TIME;
  const int MAX_DEPTH;

  const char env;

  Random rng;

};

#endif
