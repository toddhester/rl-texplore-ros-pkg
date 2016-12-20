/** \file tworooms.hh
    Defines a two room gridworld domain, with possible action delays or 
    multiple goals (with partial observability). 
    \author Todd Hester
*/

#ifndef _TWOROOMS_H_
#define _TWOROOMS_H_

#include <set>
#include <rl_common/Random.h>
#include <rl_common/core.hh>
#include "gridworld.hh"

#include <deque>


/** This class defines a two room gridworld domain. It can optionally be stochastic, have action
    delays, or multiple goals (with partial observability). */
class TwoRooms: public Environment {
public:

  /** Standard Constructor
      \param rand Random Number generator
      \param stochastic Make the domain stochastic 
      \param rewardType Create -1 per step and 0 on termination (vs 0 and 1)
      \param actDelay # of steps to delay actions
      \param multiGoal create mulitple goals that are randomly selected from each episode
  */
  TwoRooms(Random &rand, bool stochastic, bool rewardType, int actDelay, bool multiGoal);

  virtual ~TwoRooms();

  virtual const std::vector<float> &sensation() const;
  virtual float apply(int action);

  virtual bool terminal() const;
  virtual void reset();

  virtual int getNumActions();
  virtual void getMinMaxFeatures(std::vector<float> *minFeat, std::vector<float> *maxFeat);
  virtual void getMinMaxReward(float* minR, float* maxR);

  /** Create an experience tuple for the given state-action. */
  experience getExp(float s0, float s1, int a);

  virtual std::vector<experience> getSeedings();

protected:
  typedef std::pair<float,float> coord_t;
  enum room_action_t {NORTH, SOUTH, EAST, WEST};

private:
  const Gridworld *const grid;
  coord_t goal;
  coord_t goal2;
  std::deque<int> actHistory;
  bool useGoal2;

  const bool negReward;
  const bool noisy;
  const int actDelay;
  const bool multiGoal;

  Random &rng;

  coord_t doorway;

  std::vector<float> s;

  float &ns;
  float &ew;

  /** Create default two room gridworld */
  const Gridworld *create_default_map();

  /** Corrupts a movement action.
      \param action The intended action
      \return The action actually executed */
  room_action_t add_noise(room_action_t action);

  /** Randomly assigns the goal to any random 
      position in the world. */
  void randomize_goal();

  /** Return the correct reward based on the current state. */
  float reward();

};

#endif
