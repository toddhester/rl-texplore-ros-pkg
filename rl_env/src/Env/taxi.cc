/** \file taxi.cc
    Implements the taxi domain, from:
    Dietterich, "The MAXQ method for hierarchical reinforcement learning," ICML 1998.
    \author Todd Hester
    \author Nick Jong
*/

#include <rl_env/taxi.hh>

const Taxi::DefaultLandmarks Taxi::defaultlandmarks;

Taxi::DefaultLandmarks::DefaultLandmarks() {
  push_back(value_type(4.,0.));
  push_back(value_type(0.,3.));
  push_back(value_type(4.,4.));
  push_back(value_type(0.,0.));
}

Taxi::Taxi(Random &rand, const Gridworld *gridworld, bool stochastic):
  grid(gridworld), landmarks(4), noisy(stochastic), rng(rand),
  s(4),
  ns(s[0]),
  ew(s[1]),
  pass(s[2]),
  dest(s[3])
{
  randomize_landmarks_to_corners();
  reset();
}

Taxi::Taxi(Random &rand):
  grid(create_default_map()),
  landmarks(defaultlandmarks),
  noisy(false),
  rng(rand),
  s(4),
  ns(s[0]),
  ew(s[1]),
  pass(s[2]),
  dest(s[3])
{
  reset();
}

Taxi::Taxi(Random &rand, bool stochastic):
  grid(create_default_map()),
  landmarks(defaultlandmarks),
  noisy(stochastic),
  rng(rand),
  s(4),
  ns(s[0]),
  ew(s[1]),
  pass(s[2]),
  dest(s[3])
{
  reset();
}

Taxi::Taxi(Random &rand, unsigned width, unsigned height, bool stochastic):
  grid(new Gridworld(height, width, rand)),
  landmarks(4), noisy(stochastic), rng(rand),
  s(4),
  ns(s[0]),
  ew(s[1]),
  pass(s[2]),
  dest(s[3])
{
  randomize_landmarks_to_corners();
  reset();
}

Taxi::~Taxi() { delete grid; }

const std::vector<float> &Taxi::sensation() const { return s; }

float Taxi::apply(int action) {
  const taxi_action_t effect =
    noisy
    ? add_noise(static_cast<taxi_action_t>(action))
    : static_cast<taxi_action_t>(action);
  switch(effect) {
  case NORTH:
    if (!grid->wall(static_cast<unsigned>(ns),
                    static_cast<unsigned>(ew),
                    effect))
      {
        ++ns;
        apply_fickle_passenger();
      }
    return -1;
  case SOUTH:
    if (!grid->wall(static_cast<unsigned>(ns),
                    static_cast<unsigned>(ew),
                    effect))
      {
        --ns;
        apply_fickle_passenger();
      }
    return -1;
  case EAST:
    if (!grid->wall(static_cast<unsigned>(ns),
                    static_cast<unsigned>(ew),
                    effect))
      {
        ++ew;
        apply_fickle_passenger();
      }
    return -1;
  case WEST:
    if (!grid->wall(static_cast<unsigned>(ns),
                    static_cast<unsigned>(ew),
                    effect))
      {
        --ew;
        apply_fickle_passenger();
      }
    return -1;
  case PICKUP: {
    if (pass < landmarks.size()
        && coord_t(ns,ew) == landmarks[static_cast<unsigned>(pass)])
      {
        pass = landmarks.size();
        fickle = noisy;
        return -1;
      } else
      return -10;
  }
  case PUTDOWN:
    if (pass == landmarks.size()
        && coord_t(ns,ew) == landmarks[static_cast<unsigned>(dest)]) {
      pass = dest;
      return 20;
    } else
      return -10;
  }
  std::cerr << "Unreachable point reached in Taxi::apply!!!\n";
  return 0; // unreachable, I hope
}

bool Taxi::terminal() const {
  return pass == dest;
}

void Taxi::reset() {
  ns = rng.uniformDiscrete(1, grid->height()) - 1;
  ew = rng.uniformDiscrete(1, grid->width()) - 1;
  pass = rng.uniformDiscrete(1, landmarks.size()) - 1;
  do dest = rng.uniformDiscrete(1, landmarks.size()) - 1;
  while (dest == pass);
  fickle = false;
}



int Taxi::getNumActions() {
  return 6;
}


const Gridworld *Taxi::create_default_map() {
  std::vector<std::vector<bool> > nsv(5, std::vector<bool>(4,false));
  std::vector<std::vector<bool> > ewv(5, std::vector<bool>(4,false));
  ewv[0][0] = true;
  ewv[0][2] = true;
  ewv[1][0] = true;
  ewv[1][2] = true;
  ewv[3][1] = true;
  ewv[4][1] = true;
  return new Gridworld(5,5,nsv,ewv);
}

Taxi::taxi_action_t Taxi::add_noise(taxi_action_t action) {
  switch(action) {
  case NORTH:
  case SOUTH:
    return rng.bernoulli(0.8) ? action : (rng.bernoulli(0.5) ? EAST : WEST);
  case EAST:
  case WEST:
    return rng.bernoulli(0.8) ? action : (rng.bernoulli(0.5) ? NORTH : SOUTH);
  default:
    return action;
  }
}

void Taxi::apply_fickle_passenger() {

  if (fickle) {
    fickle = false;
    if (rng.bernoulli(0.3)) {
      dest += rng.uniformDiscrete(1, landmarks.size() - 1);
      dest = static_cast<int>(dest) % landmarks.size();
    }
  }

}

void Taxi::randomize_landmarks() {
  std::vector<unsigned> indices(landmarks.size());
  const unsigned n = grid->height() * grid->width();
  for (unsigned i = 0; i < indices.size(); ++i) {
    unsigned index;
    bool duplicate;
    do {
      index = rng.uniformDiscrete(1,n) - 1;
      duplicate = false;
      for (unsigned j = 0; j < i; ++j)
        if (index == indices[j])
          duplicate = true;
    } while (duplicate);
    indices[i] = index;
  }
  for (unsigned i = 0; i < indices.size(); ++i)
    landmarks[i] = coord_t(indices[i] / grid->width(),
                           indices[i] % grid->width());
}

void Taxi::randomize_landmarks_to_corners() {
  for (unsigned i = 0; i < landmarks.size(); ++i) {
    int ns = rng.uniformDiscrete(0,1);
    int ew = rng.uniformDiscrete(0,1);
    if (1 == i/2)
      ns = grid->height() - ns - 1;
    if (1 == i%2)
      ew = grid->width() - ew - 1;
    landmarks[i] = coord_t(ns,ew);
  }
}


void Taxi::setSensation(std::vector<float> newS){
  if (s.size() != newS.size()){
    cerr << "Error in sensation sizes" << endl;
  }

  for (unsigned i = 0; i < newS.size(); i++){
    s[i] = (int)newS[i];
  }
}

std::vector<experience> Taxi::getSeedings() {

  // return seedings
  std::vector<experience> seeds;

  if (true)
    return seeds;
  // REMOVE THIS TO USE SEEDINGS

  // single seed for each of 4 drop off and pickup cases
  for (int i = 0; i < 4; i++){
    // drop off
    seeds.push_back(getExp(landmarks[i].first, landmarks[i].second, 4, i, PUTDOWN));
    // pick up
    seeds.push_back(getExp(landmarks[i].first, landmarks[i].second, i, rng.uniformDiscrete(0,3), PICKUP));
  }

  reset();

  return seeds;

}

experience Taxi::getExp(float s0, float s1, float s2, float s3, int a){

  experience e;

  e.s.resize(4, 0.0);
  e.next.resize(4, 0.0);

  ns = s0;
  ew = s1;
  pass = s2;
  dest = s3;

  e.act = a;
  e.s = sensation();
  e.reward = apply(e.act);

  e.terminal = terminal();
  e.next = sensation();

  return e;
}


void Taxi::getMinMaxFeatures(std::vector<float> *minFeat,
                             std::vector<float> *maxFeat){

  minFeat->resize(s.size(), 0.0);
  maxFeat->resize(s.size(), 1.0);

  (*minFeat)[0] = 0.0;
  (*maxFeat)[0] = 4.0;
  (*minFeat)[1] = 0.0;
  (*maxFeat)[1] = 4.0;
  (*minFeat)[2] = 0.0;
  (*maxFeat)[2] = 4.0;
  (*minFeat)[3] = 0.0;
  (*maxFeat)[3] = 3.0;

}

void Taxi::getMinMaxReward(float *minR,
                           float *maxR){

  *minR = -10.0;
  *maxR = 20.0;

}
