/** \file RMaxModel.hh
    Defines the RMaxModel class.
    \author Todd Hester
*/

#ifndef _RMAXMODEL_HH_
#define _RMAXMODEL_HH_

#include <rl_common/Random.h>
#include <rl_common/core.hh>
#include <vector>
#include <map>
#include <set>


/** MDPModel used for RMax. Tabular model with Maximum Likelihood model for each state-action. */
class RMaxModel: public MDPModel {

public:

  /** Default constructor
      \param m # of visits before a state-actions becomes known.
      \param nact # of actions in the domain
      \param rng Random Number Generator 
  */
  RMaxModel(int m, int nact, Random rng);
  
  /** Copy constructor */
  RMaxModel(const RMaxModel&);

  virtual ~RMaxModel();
  virtual RMaxModel* getCopy();

  virtual bool updateWithExperiences(std::vector<experience> &instances);
  virtual bool updateWithExperience(experience &e);
  virtual bool getStateActionInfo(const std::vector<float> &state, int act, StateActionInfo* retval);


  // structs to be defined
  struct state_info;


  /** State info struct. Maintaints visit counts, outcome counts,
      reward sums, terminal transitions, and whether the state-action is 
      considered 'known' (>= m visits) 
  */
  struct state_info {
    int id;

    // model data (visit counts, outcome counts, reward sums, known)
    std::vector<int> visits;

    std::map< std::vector<float> , std::vector<int> > outCounts;
    std::vector<float> Rsum;
    std::vector<int> terminations;

    std::vector<bool> known;

  };

protected:
  typedef const std::vector<float> *state_t;

  // various helper functions that we need

  /** Initialize a state_info struct */
  void initStateInfo(state_info* info);
  
  /** Add the given state to the state set. initializes state info for new states not yet in set. The pointer to the state in the set is used for map of state_info's */
  state_t canonicalize(const std::vector<float> &s);

  /** Make sure the transition count vector is sized properly before indexing into it. */
  void checkTransitionCountSize(std::vector<int> *transCounts);

  /** Initialize a new state */
  void initNewState(state_t s);



private:
  
  /** Set of all distinct sensations seen.  Pointers to elements of
      this set serve as the internal representation of the environment
      state. */
  std::set<std::vector<float> > statespace;

  /** Hashmap mapping state vectors to their state_info structs.    */
  std::map<state_t, state_info> statedata;

  int nstates;

  int M;
  int nact;
  Random rng;

  bool RMAX_DEBUG;
  
};



#endif
