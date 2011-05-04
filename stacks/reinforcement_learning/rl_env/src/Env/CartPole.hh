/** \file CartPole.hh
    Defines the Cart-Pole balancing domain, with possible noise.
    \author Todd Hester
*/

#ifndef _CARTPOLE_H_
#define _CARTPOLE_H_

#include <set>
#include <rl_common/Random.h>
#include <rl_common/core.hh>

/** This class defines the Cart-Pole balancing domain. */
class CartPole: public Environment {
public:

  /** Creates a deterministic CartPole domain.
      \param rand Random number generator used solely for random
      initial states.  
  */
  CartPole(Random &rand);

  /** Creates a Cart-Pole domain, possibly with noise.
      \param rand Random number generator
      \param stochastic noisy transitions? 
  */
  CartPole(Random &rand, bool stochastic);

  virtual ~CartPole();

  virtual const std::vector<float> &sensation() const;
  virtual float apply(int action);

  /** Calculate the new state and reward for the given force */
  float transition(float force);

  virtual bool terminal() const;
  virtual void reset();

  virtual int getNumActions();
  virtual void getMinMaxFeatures(std::vector<float> *minFeat, std::vector<float> *maxFeat);
  virtual void getMinMaxReward(float* minR, float* maxR);

  /** Set the state vector (for debug purposes) */
  void setSensation(std::vector<float> newS);

  virtual std::vector<experience> getSeedings();

  /** Get an experience for the given state-action */
    experience getExp(float s0, float s1, float s2, float s3, int a);
  
protected:
  enum car_action_t {LEFT, RIGHT};

private:

  float GRAVITY;
  float MASSCART;
  float MASSPOLE;
  float TOTAL_MASS;
  float LENGTH;
  
  float POLEMASS_LENGTH;
  float FORCE_MAG;
  float TAU;
  
  float FOURTHIRDS;
  float DEG_T_RAD;
  float RAD_T_DEG;

  const bool noisy;
  Random &rng;

  std::vector<float> s;
 
  float &cartPos;
  float &cartVel;
  float &poleAngle;
  float &poleVel;

  float reward();

};

#endif
