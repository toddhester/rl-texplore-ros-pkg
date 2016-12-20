/** \file CartPole.cc
    Implements the Cart-Pole balancing domain, with possible noise.
    \author Todd Hester
*/

#include <rl_env/CartPole.hh>

 
CartPole::CartPole(Random &rand):
  noisy(false),
  rng(rand),
  s(4),
  cartPos(s[0]),
  cartVel(s[1]),
  poleAngle(s[2]),
  poleVel(s[3])
{
  reset();
  //cout << *this << endl;
}
 

CartPole::CartPole(Random &rand, bool stochastic):
  noisy(stochastic),
  rng(rand),
  s(4),
  cartPos(s[0]),
  cartVel(s[1]),
  poleAngle(s[2]),
  poleVel(s[3])
{
  reset();
}


CartPole::~CartPole() { }

const std::vector<float> &CartPole::sensation() const { 
  //cout << "At state " << s[0] << ", " << s[1] << endl;

  return s; 
}


float CartPole::transition(float force){

  // transition

  float xacc;
  float thetaacc;
  float costheta;
  float sintheta;
  float temp;
  
  //Noise of 1.0 means possibly halfway to opposite action
  if (noisy){
    float thisNoise=1.0*FORCE_MAG*(rng.uniform(-0.5, 0.5));  
    force+=thisNoise;
  }

  costheta = cos(poleAngle);
  sintheta = sin(poleAngle);

  temp = (force + POLEMASS_LENGTH * poleVel * poleVel * sintheta) / TOTAL_MASS;

  thetaacc = (GRAVITY * sintheta - costheta * temp) / (LENGTH * (FOURTHIRDS - MASSPOLE * costheta * costheta / TOTAL_MASS));

  xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS;

  // Update the four state variables, using Euler's method. 
  cartPos += TAU * cartVel;
  cartVel += TAU * xacc;
  poleAngle += TAU * poleVel;
  poleVel += TAU * thetaacc;
  
  // These probably never happen because the pole would crash 
  while (poleAngle >= M_PI) {
    poleAngle -= 2.0 * M_PI;
  }
  while (poleAngle < -M_PI) {
    poleAngle += 2.0 * M_PI;
  }

  // dont velocities go past ranges
  if (fabs(cartVel) > 3){
    //    cout << "cart velocity out of range: " << cartVel << endl;
    if (cartVel > 0)
      cartVel = 3;
    else
      cartVel = -3;
  }
  if (fabs(poleVel) > M_PI){
    //    cout << "pole velocity out of range: " << poleVel << endl;
    if (poleVel > 0)
      poleVel = M_PI;
    else
      poleVel = -M_PI;
  }

  return reward();

}


float CartPole::apply(int action) {

  float force = 0;
  if (action == 1) {
    force = FORCE_MAG;
  } else {
    force = -FORCE_MAG;
  }

  return transition(force);
}

 

float CartPole::reward() {

  // normally +1 and 0 on goal
  if (terminal())
    return 0.0;
  else
    return 1.0;
}



bool CartPole::terminal() const {
  // current position past termination conditions (off track, pole angle)
  return (fabs(poleAngle) > (DEG_T_RAD*12.0) || fabs(cartPos) > 2.4);
}



void CartPole::reset() {

  GRAVITY = 9.8;
  MASSCART = 1.0;
  MASSPOLE = 0.1;
  TOTAL_MASS = (MASSPOLE + MASSCART);
  LENGTH = 0.5;	  // actually half the pole's length 
  
  POLEMASS_LENGTH = (MASSPOLE * LENGTH);
  FORCE_MAG = 10.0;
  TAU = 0.02;	  // seconds between state updates 
  
  FOURTHIRDS = 4.0 / 3.0;
  DEG_T_RAD = 0.01745329;
  RAD_T_DEG = 1.0/DEG_T_RAD;

  if (noisy){
    cartPos = rng.uniform(-0.5, 0.5);
    cartVel = rng.uniform(-0.5, 0.5);
    poleAngle = rng.uniform(-0.0625, 0.0625);
    poleVel = rng.uniform(-0.0625, 0.0625);
  } else {
    cartPos = 0.0;
    cartVel = 0.0;
    poleAngle = 0.0;
    poleVel = 0.0;
  }

}



int CartPole::getNumActions(){
  return 2;
}


void CartPole::setSensation(std::vector<float> newS){
  if (s.size() != newS.size()){
    cerr << "Error in sensation sizes" << endl;
  }

  for (unsigned i = 0; i < newS.size(); i++){
    s[i] = newS[i];
  }
}

std::vector<experience> CartPole::getSeedings() {

  // return seedings
  std::vector<experience> seeds;

  // single seed of each 4 terminal cases
  seeds.push_back(getExp(-2.4, -0.1, 0, 0, 0));
  seeds.push_back(getExp(2.4, 0.2, 0.1, 0.2, 1));
  seeds.push_back(getExp(0.4, 0.3, 0.2, 0.3, 0));
  seeds.push_back(getExp(-.3, 0.05, -0.2, -0.4, 1));

  reset();

  return seeds;

}

experience CartPole::getExp(float s0, float s1, float s2, float s3, int a){

  experience e;

  e.s.resize(4, 0.0);
  e.next.resize(4, 0.0);

  cartPos = s0;
  cartVel = s1;
  poleAngle = s2;
  poleVel = s3;

  e.act = a;
  e.s = sensation();
  e.reward = apply(e.act);

  e.terminal = terminal();
  e.next = sensation();

  return e;
}

void CartPole::getMinMaxFeatures(std::vector<float> *minFeat,
                                 std::vector<float> *maxFeat){
  
  minFeat->resize(s.size(), 0.0);
  maxFeat->resize(s.size(), 1.0);

  (*minFeat)[0] = -2.5;//3;
  (*maxFeat)[0] = 2.5;//3;

  (*minFeat)[1] = -3.0;
  (*maxFeat)[1] = 3.0;

  (*minFeat)[2] = -12.0 * DEG_T_RAD;
  (*maxFeat)[2] = 12.0 * DEG_T_RAD;
  
  (*minFeat)[3] = -M_PI;
  (*maxFeat)[3] = M_PI;

}

void CartPole::getMinMaxReward(float *minR,
                              float *maxR){
  
  *minR = 0.0;
  *maxR = 1.0;    
  
}
