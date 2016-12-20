/** \file taxi.hh
    Defines the taxi domain, from:
    Dietterich, "The MAXQ method for hierarchical reinforcement learning," ICML 1998. 
    \author Todd Hester
    \author Nick Jong
*/

#ifndef _TAXI_H_
#define _TAXI_H_

#include <set>
#include <rl_common/Random.h>
#include <rl_common/core.hh>
#include "gridworld.hh"

/** This class defines the Taxi domain. */
class Taxi: public Environment {
public:
  /** Creates a Taxi domain using the specified map.
      \param rand Random number generator to use.
      \param gridworld The map to use.
      \param stochastic Whether to use nondeterministic actions and
                        fickle passenger. */
  Taxi(Random &rand, const Gridworld *gridworld, bool stochastic);

  /** Creates a deterministic Taxi domain.
      \param rand Random number generator used solely for random
                  initial states.  */
  Taxi(Random &rand);

  /** Creates a possibly noisy Taxi domain. 
      \param rand Random number generator to use.
      \param stochastic Whether to use nondeterministic actions and
      fickle passenger. 
  */
  Taxi(Random &rand, bool stochastic);

  /** Creates a random Taxi domain of the given size. 
      \param rand Random number generator to use.
      \param width width of grid
      \param height height of grid
      \param stochastic Whether to use nondeterministic actions and
      fickle passenger. 
  */ 
  Taxi(Random &rand, unsigned width, unsigned height, bool stochastic);

  virtual ~Taxi();

  virtual const std::vector<float> &sensation() const;
  virtual float apply(int action);

  virtual bool terminal() const;
  virtual void reset();
  virtual int getNumActions();
  virtual void getMinMaxFeatures(std::vector<float> *minFeat, std::vector<float> *maxFeat);
  virtual void getMinMaxReward(float* minR, float* maxR);
  virtual std::vector<experience> getSeedings();

  /** Get an example experience for the given state-action. */
  experience getExp(float s0, float s1, float s2, float s3, int a);

  /** Set the current state (for debug purposes) */
  void setSensation(std::vector<float> newS);

protected:
  typedef std::pair<float,float> coord_t;
  enum taxi_action_t {NORTH, SOUTH, EAST, WEST, PICKUP, PUTDOWN};

  /** The default landmarks for the Taxi domain */
  class DefaultLandmarks: public std::vector<coord_t> {
  public:
    DefaultLandmarks();
  };

private:
  const Gridworld *const grid;
  std::vector<coord_t> landmarks; // not const because of randomize_landmarks
  const bool noisy;
  Random &rng;

  std::vector<float> s;
  bool fickle;

  float &ns;
  float &ew;
  float &pass;
  float &dest;

  /** Create the default gridworld */
  static const Gridworld *create_default_map();

  static const DefaultLandmarks defaultlandmarks;

  /** Corrupts a movement action.
      \param action The intended action
      \return The action actually executed */
  taxi_action_t add_noise(taxi_action_t action);

  /** If the domain is noisy and the taxi has just taken its first
      step since picking up the passenger, then potentially change the
      destination.  */
  void apply_fickle_passenger();

  /** Randomly assigns the four landmarks to any four distinct
      positions in the world. */
  void randomize_landmarks();

  /** Randomly assigns the landmarks to positions near the corners of
      the map. */
  void randomize_landmarks_to_corners();
};

#endif
