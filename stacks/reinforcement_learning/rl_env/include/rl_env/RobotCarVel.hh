/** \file RobotCarVel.hh
    This domain is a simulation of velocity control for the Austin Robot 
    Technology autonomous vehicle. 
    This vehicle is described in:
    Beeson et al, "Multiagent Interactions in Urban Driving," Journal of Physical Agents, March 2008.
    The velocity control task is described in:
    Hester, Quinlan, and Stone, "A Real-Time Model-Based Reinforcement Learning Architecture for Robot Control", arXiv 1105.1749, 2011.
    \author Todd Hester
*/

#ifndef _ROBOTCAR_H_
#define _ROBOTCAR_H_

#include <set>
#include <rl_common/Random.h>
#include <rl_common/core.hh>

/** This class defines a domain that is a simulation of velocity control for the Austin Robot 
    Technology autonomous vehicle. */
class RobotCarVel: public Environment {
public:

  /** Creates a RobotCarVel domain.
      \param rand Random number generator 
      \param randomVel Use random starting and target velocities
      \param upVel For specific velocity pair (not random), do target > starting vel
      \param tenToSix Use 10 and 6 m/s for specific velocity pair, rather than 7 and 2
      \param lag Implements lag on the brake actuator to fully model the real vehicle.
  */
  RobotCarVel(Random &rand, bool randomVel, bool upVel, bool tenToSix, bool lag);

  virtual ~RobotCarVel();

  virtual const std::vector<float> &sensation() const;
  virtual float apply(int action);

  virtual bool terminal() const;
  virtual void reset();

  virtual int getNumActions();
  virtual void getMinMaxFeatures(std::vector<float> *minFeat, std::vector<float> *maxFeat);
  virtual void getMinMaxReward(float* minR, float* maxR);

  /** Set the state vector for debug purposes. */
  void setSensation(std::vector<float> newS);

  virtual std::vector<experience> getSeedings();

  /** Get an experience seed at a random velocity for the given target velocity. */
  experience getRandomVelSeed(float target);

  /** Get an example experience for the given state-action. */
  experience getExp(float s0, float s1, float s2, float s3, int a);

  //virtual bool isEpisodic() { return false; };

  /** Bound a value between the given min and max. */
  float bound(float val, float min, float max);

protected:
  enum car_action_t {NOTHING, THROTTLE_UP, THROTTLE_DOWN, BRAKE_UP, BRAKE_DOWN};

private:

  Random &rng;

  std::vector<float> s;
  std::vector<float> junk;
 
  float &targetVel;
  float &currVel;
  float &trueThrottle;
  float &trueBrake;
  float &throttleTarget;
  float &brakeTarget;

  const bool randomVel;
  const bool upVel;
  const bool tenToSix;
  const bool lag;

  float brakePosVel;
  int actNum;

};

#endif
