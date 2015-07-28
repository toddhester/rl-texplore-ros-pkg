/** \file TwoRooms.cc
    Implements a two room gridworld domain, with possible action delays or 
    multiple goals (with partial observability). 
    \author Todd Hester
*/

#include <rl_env/tworooms.hh>


TwoRooms::TwoRooms(Random &rand, bool stochastic, bool rewardType, 
                   int actDelay, bool multiGoal):
  grid(create_default_map()),
  goal(coord_t(1.,1.)), 
  goal2(coord_t(4.,1.)),
  negReward(rewardType),
  noisy(stochastic),
  actDelay(actDelay),
  multiGoal(multiGoal),
  rng(rand),
  doorway(coord_t(2.,5.)),
  s(2),
  ns(s[0]),
  ew(s[1])
{
  reset();
}


TwoRooms::~TwoRooms() { delete grid; }

const std::vector<float> &TwoRooms::sensation() const { 
  //cout << "At state " << s[0] << ", " << s[1] << endl;

  return s; 
}

float TwoRooms::apply(int action) {

  //cout << "Taking action " << static_cast<room_action_t>(action) << endl;

  int actUsed = action;

  if (actDelay > 0){
    actUsed = actHistory.front();
    actHistory.push_back(action);
    actHistory.pop_front();
  }

  if (actUsed > -1){

    const room_action_t effect =
      noisy
      ? add_noise(static_cast<room_action_t>(actUsed)) 
      : static_cast<room_action_t>(actUsed);
    switch(effect) {
    case NORTH:
      if (!grid->wall(static_cast<unsigned>(ns),
                      static_cast<unsigned>(ew),
                      effect))
        {
          ++ns;
        }
      return reward();
    case SOUTH:
      if (!grid->wall(static_cast<unsigned>(ns),
                      static_cast<unsigned>(ew),
                      effect))
        {
          --ns;
        }
      return reward();
    case EAST:
      if (!grid->wall(static_cast<unsigned>(ns),
                      static_cast<unsigned>(ew),
                      effect))
        {
          ++ew;
        }
      return reward();
    case WEST:
      if (!grid->wall(static_cast<unsigned>(ns),
                      static_cast<unsigned>(ew),
                      effect))
        {
          --ew;
        }
      return reward();
    }

    std::cerr << "Unreachable point reached in TwoRooms::apply!!!\n";
  }
  
  return 0; 
}

// return the reward for this move
float TwoRooms::reward() {

  /*
  if (coord_t(ns,ew) == goal2)
    cout << "At goal 2, " << useGoal2 << endl;
  if (coord_t(ns,ew) == goal)
    cout << "At goal 1, " << !useGoal2 << endl;
  */

  if (negReward){
    // normally -1 and 0 on goal
    if (terminal())
      return 0;
    else 
      return -1;
    
  }else{

    // or we could do 0 and 1 on goal
    if (terminal())
      return 1;
    else 
      return 0;
  }
}



bool TwoRooms::terminal() const {
  // current position equal to goal??
  if (useGoal2)
    return coord_t(ns,ew) == goal2;
  else
    return coord_t(ns,ew) == goal;
}


void TwoRooms::reset() {
  // start randomly in right room
  ns = rng.uniformDiscrete(0, grid->height() - 1 );
  ew = rng.uniformDiscrete(6, grid->width() - 1);

  // a history of no_acts
  actHistory.clear();
  actHistory.assign(actDelay, -1);

  if (multiGoal){
    useGoal2 = rng.bernoulli(0.5);
    //cout << "goal2? " << useGoal2 << endl;
  }
  else {
    useGoal2 = false;
  }

  //ns = 4;
  //ew = 9;
}



int TwoRooms::getNumActions(){
  return 4;
}


const Gridworld *TwoRooms::create_default_map() {
  int width = 11;
  int height = 5;
  std::vector<std::vector<bool> > nsv(width, std::vector<bool>(height-1,false));
  std::vector<std::vector<bool> > ewv(height, std::vector<bool>(width-1,false));

  // put the wall between the two rooms
  for (int j = 0; j < height; j++){
    // skip doorway
    if (j == 2)
      continue;
    ewv[j][4] = true;
    ewv[j][5] = true;
  }

  nsv[5][1] = true;
  nsv[5][2] = true;

  // add a doorway
  doorway = coord_t(2, 5);

  return new Gridworld(height, width, nsv, ewv);
}

TwoRooms::room_action_t TwoRooms::add_noise(room_action_t action) {
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


void TwoRooms::randomize_goal() {
  const unsigned n = grid->height() * grid->width();
  unsigned index = rng.uniformDiscrete(1,n) - 1;
  goal = coord_t(index / grid->width(), index % grid->width());
}


void TwoRooms::getMinMaxFeatures(std::vector<float> *minFeat,
                                 std::vector<float> *maxFeat){
  
  minFeat->resize(s.size(), 0.0);
  maxFeat->resize(s.size(), 10.0);

  (*maxFeat)[0] = 5.0;

}

void TwoRooms::getMinMaxReward(float *minR,
                              float *maxR){
  if (negReward){
    *minR = -1.0;
    *maxR = 0.0;    
  }else{
    *minR = 0.0;
    *maxR = 1.0;
  }
}


std::vector<experience> TwoRooms::getSeedings() {

  // return seedings
  std::vector<experience> seeds;

  //if (true)
  // return seeds;
  // REMOVE THIS TO USE SEEDINGS

  // single seed of terminal state
  useGoal2 = false;
  actHistory.clear();
  actHistory.assign(actDelay, SOUTH);
  seeds.push_back(getExp(2,1,SOUTH));
  
  // possible seed of 2nd goal
  if (multiGoal){
    useGoal2 = true;
    actHistory.clear();
    actHistory.assign(actDelay, NORTH);
    seeds.push_back(getExp(3,1,NORTH));
  }

  // single seed of doorway
  actHistory.clear();
  actHistory.assign(actDelay, WEST);
  seeds.push_back(getExp(2,6,WEST));

  reset();

  return seeds;

}

experience TwoRooms::getExp(float s0, float s1, int a){

  experience e;

  e.s.resize(2, 0.0);
  e.next.resize(2, 0.0);

  ns = s0;
  ew = s1;

  e.act = a;
  e.s = sensation();
  e.reward = apply(e.act);

  e.terminal = terminal();
  e.next = sensation();

  /*
  cout << "Seed from " << e.s[0] << "," << e.s[1] << " a: " << e.act
       << " r: " << e.reward << " term: " << e.terminal << endl;
  */

  reset();

  return e;
}
