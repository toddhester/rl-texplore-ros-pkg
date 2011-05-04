#ifndef _MBS_HH_
#define _MBS_HH_

#include <rl_common/Random.h>
#include <rl_common/core.hh>
#include "ValueIteration.hh"

#include <ext/hash_map>
#include <set>
#include <vector>
#include <map>
#include <deque>

class MBS: public Planner {
public:

  /** Standard constructor
      \param numactions, numactions in the domain
      \param gamma discount factor
      \param maxloops
      \param max time
      \param rng random
  */
  MBS(int numactions, float gamma,
      int MAX_LOOPS, float MAX_TIME, int modelType,
      const std::vector<float> &featmax, 
      const std::vector<float> &featmin, const std::vector<int> &statesPerDim,
      int delay,
      Random rng = Random());

  /** Unimplemented copy constructor: internal state cannot be simply
      copied. */
  MBS(const MBS &);

  virtual ~MBS();

  virtual void setModel(MDPModel* model);
  virtual bool updateModelWithExperience(const std::vector<float> &last, 
                                         int act, 
                                         const std::vector<float> &curr, 
                                         float reward, bool term);
  virtual void planOnNewModel();
  virtual int getBestAction(const std::vector<float> &s);
  virtual void savePolicy(const char* filename);
  virtual void setSeeding(bool seed);
  virtual void setFirst();

  bool DELAYDEBUG;
  
private:

  ValueIteration* vi;
  std::deque<int> actHistory;
  const unsigned k;
  MDPModel* model;
  bool seedMode;

};


#endif
