/** \file RobotCarVel.cc
    This domain is a simulation of velocity control for the Austin Robot 
    Technology autonomous vehicle. 
    This vehicle is described in:
    Beeson et al, "Multiagent Interactions in Urban Driving," Journal of Physical Agents, March 2008.
    The velocity control task is described in:
    Hester, Quinlan, and Stone, "A Real-Time Model-Based Reinforcement Learning Architecture for Robot Control", arXiv 1105.1749, 2011.
    \author Todd Hester
*/

#include <rl_env/RobotCarVel.hh>

// normal: true values of each
RobotCarVel::RobotCarVel(Random &rand, bool randomVel, bool upVel, bool tenToSix, bool lag):
  rng(rand),
  s(4),
  junk(2),
  targetVel(s[0]),
  currVel(s[1]),
  trueThrottle(junk[0]),
  trueBrake(junk[1]),
  throttleTarget(s[2]),
  brakeTarget(s[3]),
  randomVel(randomVel),
  upVel(upVel),
  tenToSix(tenToSix),
  lag(lag)
{
  reset();
}
 


RobotCarVel::~RobotCarVel() { }

const std::vector<float> &RobotCarVel::sensation() const { 
  return s; 
}

float RobotCarVel::apply(int action) {

  float HZ = 10.0;

  actNum++;

  // figure out reward based on target/curr vel
  float reward = -10.0 * fabs(currVel - targetVel);

  float throttleChangePct = 1.0;//0.9; //1.0;
  float brakeChangePct = 1.0;//0.9; //1.0;
  if (lag){
    brakeChangePct = brakePosVel / HZ;
    float brakeVelTarget = 3.0*(brakeTarget - trueBrake);
    brakePosVel += (brakeVelTarget - brakePosVel) * 3.0 / HZ;
    trueBrake += brakeChangePct;
  } else {
    trueBrake += (brakeTarget-trueBrake) * brakeChangePct;
  }
  trueBrake = bound(trueBrake, 0.0, 1.0);

  // figure out the change of true brake/throttle position based on last targets
  trueThrottle += (throttleTarget-trueThrottle) * throttleChangePct;
  trueThrottle = bound(trueThrottle, 0.0, 0.4);

  // figure out new velocity based on those positions 
  // from the stage simulation
  float g = 9.81;         // acceleration due to gravity
  float throttle_accel = g;
  float brake_decel = g;
  float rolling_resistance = 0.01 * g;
  float drag_coeff = 0.01;
  float idle_accel = (rolling_resistance
                      + drag_coeff * 3.1 * 3.1);
  float wind_resistance = drag_coeff * currVel * currVel;
  float accel = (idle_accel
                  + trueThrottle * throttle_accel
                  - trueBrake * brake_decel
                  - rolling_resistance
                  - wind_resistance);
  currVel += (accel / HZ);
  currVel = bound(currVel, 0.0, 12.0);
  

  // figure out action's adjustment to throttle/brake targets
  if (action == THROTTLE_UP){
    brakeTarget = 0.0;
    if (throttleTarget < 0.4)
      throttleTarget += 0.1;
  }
  else if (action == THROTTLE_DOWN){
    brakeTarget = 0.0;
    if (throttleTarget > 0.0)
      throttleTarget -= 0.1;
  }
  else if (action == BRAKE_UP){
    throttleTarget = 0.0;
    if (brakeTarget < 1.0)
      brakeTarget += 0.1;
  }
  else if (action == BRAKE_DOWN){
    throttleTarget = 0.0;
    if (brakeTarget > 0.0)
      brakeTarget -= 0.1;
  }
  else if (action != NOTHING){
    cout << "invalid action " << action << endl;
  }
  throttleTarget = bound(throttleTarget, 0.0, 0.4);
  brakeTarget = bound(brakeTarget, 0.0, 1.0);
  throttleTarget = 0.1 * (float)((int)(throttleTarget*10.0));
  brakeTarget = 0.1 * (float)((int)(brakeTarget*10.0));
  
  /*
  cout << action << ", throt: " << throttleTarget << ", brake: " << brakeTarget
       << ", trueBrake: " << trueBrake 
       << ", currVel: " << currVel << ", reward: " << reward << endl;
  */

  return reward;

}


bool RobotCarVel::terminal() const {
  return false;
}



void RobotCarVel::reset() {

  // for now
  if (randomVel){
    targetVel = rng.uniformDiscrete(0, 11);
    currVel = rng.uniformDiscrete(0, 11);
  } else {
    if (tenToSix){ // 10 to 6
      if (upVel){
        targetVel = 10.0;
        currVel = 6.0;
      } else {
        targetVel = 6.0;
        currVel = 10.0;
      } 
    } else { // 7 to 2
      if (upVel){
        targetVel = 7.0;
        currVel = 2.0;
      } else {
        targetVel = 2.0;
        currVel = 7.0;
      } 
    }
  }

  actNum = 0;
  throttleTarget = rng.uniformDiscrete(0,4) * 0.1;
  brakeTarget = 0.0;
  trueThrottle = throttleTarget;
  trueBrake = brakeTarget;
  brakePosVel = 0.0;

}



int RobotCarVel::getNumActions(){
  return 5;
}


void RobotCarVel::setSensation(std::vector<float> newS){
  if (s.size() != newS.size()){
    cerr << "Error in sensation sizes" << endl;
  }

  for (unsigned i = 0; i < newS.size(); i++){
    s[i] = newS[i];
  }
}

std::vector<experience> RobotCarVel::getSeedings() {

  // return seedings
  std::vector<experience> seeds;
  //return seeds;

  
  reset();

  /*
  // seeds of perfect velocity
  if (randomVel){
    for (int i = 0; i < 12; i++){
      // 3 random of each
      for (int j = 0; j < 3; j++)
        seeds.push_back(getRandomVelSeed(i));
    } 
  } else {
    // just for target velocity
    // 5 random seeds
    for (int j = 0; j < 5; j++){
      seeds.push_back(getRandomVelSeed(targetVel));
    }
  }
  */

  
  // some completely random (non target)
  for (int i = 0; i < 25; i++){
    float vel = rng.uniform(0,11);

    float throt = 0;
    float brake = 0;
    if (rng.bernoulli(0.5)){
      throt = rng.uniformDiscrete(0,4)*0.1;
    } else {
      brake = rng.uniformDiscrete(0,9)*0.1;
    } 
    int act = i%getNumActions();
    seeds.push_back(getExp(targetVel,vel,throt,brake,act));
  }

  reset();

  return seeds;
}

experience RobotCarVel::getRandomVelSeed(float target){
  float throt = 0;
  float brake = 0;
  if (rng.bernoulli(0.5)){
    throt = rng.uniformDiscrete(0,4)*0.1;
  } else {
    brake = rng.uniformDiscrete(0,4)*0.1;
  } 

  return getExp(target, target, throt, brake, rng.uniformDiscrete(0,4));
}

experience RobotCarVel::getExp(float s0, float s1, float s2, float s3, int a){

  if (!randomVel) s0 = targetVel;

  experience e;

  e.s.resize(4, 0.0);
  e.next.resize(4, 0.0);

  targetVel = s0;
  currVel = s1;
  throttleTarget = s2;
  brakeTarget = s3;
  trueThrottle = throttleTarget;
  trueBrake = brakeTarget;
  brakePosVel = 0.0;

  e.act = a;
  e.s = sensation();
  e.reward = apply(e.act);

  e.terminal = terminal();
  e.next = sensation();

  /*
  cout << "seed from state: ";
  for (unsigned i = 0; i < e.s.size(); i++){
    cout << e.s[i] << ", ";
  } 
  cout << "act: " << e.act << " reward: " << e.reward << " next: ";
  for (unsigned i = 0; i < e.next.size(); i++){
    cout << e.next[i] << ", ";
  } 
  cout << endl;
  */

  reset();

  return e;
}


void RobotCarVel::getMinMaxFeatures(std::vector<float> *minFeat,
                                    std::vector<float> *maxFeat){
  
  minFeat->resize(s.size(), 0.0);
  maxFeat->resize(s.size(), 12.0);

  (*maxFeat)[2] = 0.4;
  (*maxFeat)[3] = 1.0;

}

void RobotCarVel::getMinMaxReward(float *minR,
                                  float *maxR){
  
  *minR = -120.0;
  *maxR = 0.0;    
  
}

float RobotCarVel::bound(float val, float min, float max){
  if (val < min)
    return min;
  if (val > max)
    return max;
  return val;
}
