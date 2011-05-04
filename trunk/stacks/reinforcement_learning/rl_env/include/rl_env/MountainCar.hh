/** \file MountainCar.hh
    Defines the Mountain Car domain, with possible action delays or linearized 
      transition dynamics.
    \author Todd Hester
*/

#ifndef _MOUNTAINCAR_H_
#define _MOUNTAINCAR_H_

#include <set>
#include <deque>
#include <rl_common/Random.h>
#include <rl_common/core.hh>

/** This class defines the Mountain Car domain, with possible action delays or linearized 
      transition dynamics.
*/
class MountainCar: public Environment {
public:

  /** Creates a deterministic MountainCar domain.
      \param rand Random number generator used solely for random
      initial states.  
  */
  MountainCar(Random &rand);

  /** Creates a Mountain Car domain.
      \param rand Random number generator
      \param stochastic if transitions are noisy
      \param lin create linearized transition dynamics
      \param delay # of steps to delay state observations
  */
  MountainCar(Random &rand, bool stochastic, bool lin, int delay);

  virtual ~MountainCar();

  virtual const std::vector<float> &sensation() const;
  virtual float apply(int action);

  virtual bool terminal() const;
  virtual void reset();

  virtual int getNumActions();
  virtual void getMinMaxFeatures(std::vector<float> *minFeat, std::vector<float> *maxFeat);
  virtual void getMinMaxReward(float* minR, float* maxR);

  /** Set the state vector (for debug purposes) */
  void setSensation(std::vector<float> newS);

  virtual std::vector<experience> getSeedings();

  /** Get an experience for the given state-action */
  experience getExp(float s0, float s1, int a);
  
protected:
  enum car_action_t {REVERSE, ZERO, FORWARD};

private:

  std::deque<float> posHistory;
  std::deque<float> velHistory;

  const bool noisy;
  Random &rng;

  std::vector<float> s;
 
  float &pos;
  float &vel;
  const bool linear;
  int delay;

  float reward();

  float bound(float val, float min, float max);
  
};

#endif
