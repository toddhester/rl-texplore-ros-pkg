#ifndef _PRIORITIZEDSWEEPING_HH_
#define _PRIORITIZEDSWEEPING_HH_

#include <rl_common/Random.h>
#include <rl_common/core.hh>

#include <set>
#include <vector>
#include <map>


class PrioritizedSweeping: public Planner {
public:

  /** The implementation maps all sensations to a set of canonical
      pointers, which serve as the internal representation of
      environment state. */
  typedef const std::vector<float> *state_t;


  /** Standard constructor
      \param numactions, numactions in the domain
      \param gamma discount factor
      \param rng random
  */
  PrioritizedSweeping(int numactions, float gamma, float MAX_TIME,
                      bool onlyAddLastSA,  int modelType,
                      const std::vector<float> &featmax, 
                      const std::vector<float> &featmin,
                      Random rng = Random());

  /** Unimplemented copy constructor: internal state cannot be simply
      copied. */
  PrioritizedSweeping(const PrioritizedSweeping &);

  virtual ~PrioritizedSweeping();

  virtual void setModel(MDPModel* model);
  virtual bool updateModelWithExperience(const std::vector<float> &last, 
                                         int act, 
                                         const std::vector<float> &curr, 
                                         float reward, bool term);
  virtual void planOnNewModel();
  virtual int getBestAction(const std::vector<float> &s);

  bool PLANNERDEBUG;
  bool POLICYDEBUG; //= false; //true;
  bool MODELDEBUG;
  bool ACTDEBUG;
  bool LISTDEBUG;

  /** Model that we're using */
  MDPModel* model;



protected:


  struct state_info;
  struct model_info;

  struct saqPair {
    std::vector<float> s;
    int a;
    float q;
  };

  /** State info struct */
  struct state_info {
    int id;

    bool fresh;

    // experience data
    std::vector<int> visits;

    // data filled in from models
    StateActionInfo* modelInfo;

    //std::map<state_t, std::vector<float> > P;
    //std::vector<float> R;
    //std::vector<bool> known;

    // q values from policy creation
    std::vector<float> Q;
    
    // which states lead to this state?
    std::list<saqPair> pred;
    std::vector<int> lastUpdate;

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

  // functions to update our models and get info from them
  void updateStatesFromModel();

  double getSeconds();

  // for prioritized sweeping
  void updatePriorityList(state_info* info, const std::vector<float> &next);
  bool saqPairMatch(saqPair a, saqPair b);
  float updateQValues(const std::vector<float> &state, int act);
  void addSAToList(const std::vector<float> &s, int act, float q);
  void updateStateActionFromModel(const std::vector<float> &state, int a);

private:

  /** Set of all distinct sensations seen.  Pointers to elements of
      this set serve as the internal representation of the environment
      state. */
  std::set<std::vector<float> > statespace;

  /** Hashmap mapping state vectors to their state_info structs. */
  std::map<state_t, state_info> statedata;

  /** priority list for prioritized sweeping */
  std::list< saqPair> priorityList;

  std::vector<float> featmax;
  std::vector<float> featmin;

  std::vector<float> prevstate;
  int prevact;
  
  double planTime;
  int nstates;
  int nactions; 
  int lastModelUpdate;

  int MAX_STEPS;
  bool timingType;

  const int numactions;
  const float gamma;
  const float MAX_TIME;
  const bool onlyAddLastSA;
  const int modelType;

};

#endif
