/** \file FuelRooms.hh
    Defines the Fuel World domain, with possible noise.
    From the paper:
    Hester and Stone, "Real Time Targeted Exploration in Large Domains", ICDL 2010.
    \author Todd Hester
*/

#ifndef _FUELROOMS_H_
#define _FUELROOMS_H_

#include <set>
#include <rl_common/Random.h>
#include <rl_common/core.hh>
#include "gridworld.hh"


/** This class defines the Fuel World gridworld domain */
class FuelRooms: public Environment {
public:

  /** Creates a deterministic FuelRooms domain.
      \param rand Random number generator.
      \param extraVariation the costs of fuel stations vary even more
      \param stoch Stochastic or deterministic
  */
  FuelRooms(Random &rand, bool extraVariation, bool stoch);

  virtual ~FuelRooms();

  virtual const std::vector<float> &sensation() const;
  virtual float apply(int action);

  virtual bool terminal() const;
  virtual void reset();

  virtual int getNumActions();
  virtual void getMinMaxFeatures(std::vector<float> *minFeat, std::vector<float> *maxFeat);
  virtual void getMinMaxReward(float* minR, float* maxR);

  virtual std::vector<experience> getSeedings();

  /** Get an example experience for this state-action. */
  experience getExp(int s0, int s1, int s2, int a);

  // stuff to check on exploration
  /** For heatmap plots: Keep track of visits to each state, total states, and fuel station states. */
  void checkVisits();

  /** For heatmap plots: Reset visit counts for states to 0. */
  void resetVisits();

  /** For heatmap plots: Print % of visits to different state types */
  void printVisits();

  /** For heatmap plots: Print map of visits to each state cell */
  void printVisitMap(string filename);

protected:
  typedef std::pair<float,float> coord_t;
  enum room_action_t {NORTH, NORTHEAST, EAST, SOUTHEAST, SOUTH, SOUTHWEST, WEST, NORTHWEST};

private:
  const int height;
  const int width;
  coord_t goal;

  const bool extraVar;
  const bool noisy;
  Random &rng;


  std::vector<float> s;

  float &ns;
  float &ew;
  float &energy;

  // keep track of the agent's exploration so we can look at it later
  int fuelVisited;
  int totalVisited;
  
  int** stateVisits;

  /** Corrupts a movement action.
      \param action The intended action
      \return The action actually executed */
  room_action_t add_noise(room_action_t action);

  /** Return the correct reward based on the current state. */
  float reward(int effect);

};

#endif
