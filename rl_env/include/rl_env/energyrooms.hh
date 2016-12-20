#ifndef _ENERGYROOMS_H_
#define _ENERGYROOMS_H_

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

class EnergyRooms: public Environment {
public:
  /** Creates a EnergyRooms domain using the specified map.
      \param rand Random number generator to use.
      \param gridworld The map to use.
      \param stochastic Whether to use nondeterministic actions. */
  EnergyRooms(Random &rand, const Gridworld *gridworld, bool stochastic);

  /** Creates a deterministic EnergyRooms domain.
      \param rand Random number generator used solely for random
      initial states.  
      \param negReward Whether negative rewards are on/off.
  */
  EnergyRooms(Random &rand, bool negReward);

  /** Creates a possibly noisy EnergyRooms domain. */
  EnergyRooms(Random &rand, bool stochastic, bool negReward, bool goalOption);

  /** Creates a possibly noisy EnergyRooms domain. */
  EnergyRooms(Random &rand, bool stochastic, bool negReward, bool goalOption, bool fuel);

  /** Creates a random EnergyRooms domain of the given size. */
  EnergyRooms(Random &rand, unsigned width, unsigned height, bool stochastic);

  virtual ~EnergyRooms();

  virtual const std::vector<float> &sensation() const;
  virtual float apply(int action);

  virtual bool terminal() const;
  virtual void reset();

  virtual int getNumActions();
  virtual void getMinMaxFeatures(std::vector<float> *minFeat, std::vector<float> *maxFeat);
  virtual void getMinMaxReward(float* minR, float* maxR);

  const Gridworld &gridworld() const { return *grid; }

  friend std::ostream &operator<<(std::ostream &out, const EnergyRooms &rooms);

  std::vector<std::vector<float> > getSubgoals();

protected:
  typedef std::pair<float,float> coord_t;
  enum room_action_t {NORTH, SOUTH, EAST, WEST};

private:
  const Gridworld *const grid;
  coord_t goal;

  const bool negReward;

  const bool noisy;
  Random &rng;

  coord_t doorway;

  std::vector<float> s;

  float &ns;
  float &ew;
  float &energy;

  const bool goalOption;
  const bool fuel;

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
