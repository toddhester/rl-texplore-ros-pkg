#ifndef _FOURROOMS_H_
#define _FOURROOMS_H_

#include <set>
#include <rl_common/Random.h>
#include <rl_common/core.hh>
#include "gridworld.hh"

/*
inline ostream &operator<<(ostream &out, const room_action_t &a) {
  switch(a) {
  case NORTH: return out << "NORTH";
  case SOUTH: return out << "SOUTH";
  case EAST: return out << "EAST";
  case WEST: return out << "WEST";
  }
  return out;
}
*/

class FourRooms: public Environment {
public:
  /** Creates a FourRooms domain using the specified map.
      \param rand Random number generator to use.
      \param gridworld The map to use.
      \param stochastic Whether to use nondeterministic actions. */
  FourRooms(Random &rand, const Gridworld *gridworld, bool stochastic);

  /** Creates a deterministic FourRooms domain.
      \param rand Random number generator used solely for random
      initial states.  
      \param negReward Whether negative rewards are on/off.
  */
  FourRooms(Random &rand);

  /** Creates a possibly noisy FourRooms domain. */
  FourRooms(Random &rand, bool stochastic, bool negReward, bool exReward);

  /** Creates a possibly noisy FourRooms domain with distances to walls. */
  FourRooms(Random &rand, bool stochastic, bool negReward);

  /** Creates a Four Rooms domain with distance and reward distances */
  FourRooms(Random &rand, bool stochastic);

  /** Creates a random FourRooms domain of the given size. */
  FourRooms(Random &rand, unsigned width, unsigned height, bool stochastic);

  virtual ~FourRooms();

  virtual const std::vector<float> &sensation() const;
  virtual float apply(int action);

  virtual bool terminal() const;
  virtual void reset();

  virtual int getNumActions();
  virtual void getMinMaxFeatures(std::vector<float> *minFeat, std::vector<float> *maxFeat);
  virtual void getMinMaxReward(float* minR, float* maxR);

  const Gridworld &gridworld() const { return *grid; }

  friend std::ostream &operator<<(std::ostream &out, const FourRooms &rooms);

  std::vector<std::vector<float> > getSubgoals();
  void calcWallDistances();

  void setSensation(std::vector<float> newS);

protected:
  typedef std::pair<float,float> coord_t;
  enum room_action_t {NORTH, SOUTH, EAST, WEST};

private:
  const Gridworld *const grid;
  coord_t goal;

  const bool negReward;
  const bool noisy;
  const bool extraReward;
  const bool rewardSensor;

  Random &rng;

  coord_t doorway;

  std::vector<float> s;
  std::vector<float> trash;

  float &ns;
  float &ew;

  float &distN;
  float &distS;
  float &distE;
  float &distW;

  float &rewardEW;
  float &rewardNS;

  const bool goalOption;

  const Gridworld *create_default_map();

  /** Corrupts a movement action.
      \param action The intended action
      \return The action actually executed */
  room_action_t add_noise(room_action_t action);

  /** Randomly assigns the goal to any random 
      position in the world. */
  void randomize_goal();

  /** Return the correct reward based on the current state. */
  float reward(int effect);

};

#endif
