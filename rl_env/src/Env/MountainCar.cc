/** \file MountainCar.cc
    Implements the Mountain Car domain, with possible action delays or linearized 
      transition dynamics.
    \author Todd Hester
*/

#include <rl_env/MountainCar.hh>

 
MountainCar::MountainCar(Random &rand):
  noisy(false),
  rng(rand),
  s(2),
  pos(s[0]),
  vel(s[1]),
  linear(false),
  delay(0)
{
  reset();
  //cout << *this << endl;
}
 

MountainCar::MountainCar(Random &rand, bool stochastic, bool lin, int delay):
  noisy(stochastic),
  rng(rand),
  s(2),
  pos(s[0]),
  vel(s[1]),
  linear(lin),
  delay(delay)
{
  reset();
}


MountainCar::~MountainCar() { }

const std::vector<float> &MountainCar::sensation() const { 
  //cout << "At state " << s[0] << ", " << s[1] << endl;

  return s; 
}

float MountainCar::apply(int action) {

  //cout << "Taking action " << action << endl;

  float actVal = ((float)action-1.0);
  if (noisy){
    actVal += rng.uniform(-0.5, 0.5);
  }

  float newVel = vel;
  if (linear){
    // for now, make this linear
    newVel = vel + 0.001 * actVal + -0.0075*pos;
  } else {
    newVel = vel + 0.001 * actVal + -0.0025*cos(3.0*pos);
  }

  newVel = bound(newVel, -0.07, 0.07);

  float newPos = pos + vel;
  if (newPos < -1.2f && newVel < 0.0f)
    newVel = 0.0;
  newPos = bound(newPos, -1.2, 0.6);

  pos = newPos;
  vel = newVel;

  if (delay > 0){
    posHistory.push_back(newPos);
    pos = posHistory.front();
    posHistory.pop_front();
    velHistory.push_back(newVel);
    vel = velHistory.front();
    velHistory.pop_front();
    //    cout << "new pos: " << newPos << " observed: " << pos << endl;
    //cout << "new vel: " << newVel << " observed: " << vel << endl;
  }

  return reward();

}

float MountainCar::bound(float val, float min, float max){
  if (val < min)
    return min;
  if (val > max)
    return max;
  return val;
}


float MountainCar::reward() {
  
  // normally -1 and 0 on goal
  if (terminal())
    return 0;
  else 
    return -1;
  
}


bool MountainCar::terminal() const {
  // current position equal to goal??
  return (pos >= 0.6); 
}



void MountainCar::reset() {

  if (noisy){
    pos = rng.uniform(-1.2, 0.59);
    vel = rng.uniform(-0.07, 0.07);
  } else {
    pos = 0;
    vel = 0;
  }

  pos = rng.uniform(-1.2, 0.59);
  vel = rng.uniform(-0.07, 0.07);

  if (delay > 0){
    posHistory.clear();
    velHistory.clear();
    for (int i = 0; i < delay; i++){
      posHistory.push_back(pos);
      velHistory.push_back(vel);
    }
  }

}


int MountainCar::getNumActions(){
  return 3;
}


void MountainCar::setSensation(std::vector<float> newS){
  if (s.size() != newS.size()){
    cerr << "Error in sensation sizes" << endl;
  }

  for (unsigned i = 0; i < newS.size(); i++){
    s[i] = newS[i];
  }
}

std::vector<experience> MountainCar::getSeedings() {

  int origDelay = delay;
  delay = 0;

  // return seedings
  std::vector<experience> seeds;

  // two seeds of terminal state
  seeds.push_back(getExp(0.58, 0.03, 2));
  //seeds.push_back(getExp(0.57, 0.06, 2));

  // random seed of each action
  for (int i = 0; i < getNumActions(); i++){
    float p = rng.uniform(-1.2, 0.6);
    float v = rng.uniform(-0.07, 0.07);
    seeds.push_back(getExp(p, v, i));
  }

  delay = origDelay;

  reset();

  return seeds;

}

experience MountainCar::getExp(float s0, float s1, int a){

  experience e;

  e.s.resize(2, 0.0);
  e.next.resize(2, 0.0);

  pos = s0;
  vel = s1;

  e.act = a;
  e.s = sensation();
  e.reward = apply(e.act);

  e.terminal = terminal();
  e.next = sensation();

  reset();

  return e;
}


void MountainCar::getMinMaxFeatures(std::vector<float> *minFeat,
                                    std::vector<float> *maxFeat){
  
  minFeat->resize(s.size(), 0.0);
  maxFeat->resize(s.size(), 1.0);

  (*minFeat)[0] = -1.2;
  (*maxFeat)[0] = 0.6;

  (*minFeat)[1] = -0.07;
  (*maxFeat)[1] = 0.07;

}

void MountainCar::getMinMaxReward(float *minR,
                              float *maxR){
  
  *minR = -1.0;
  *maxR = 0.0;    
  
}
