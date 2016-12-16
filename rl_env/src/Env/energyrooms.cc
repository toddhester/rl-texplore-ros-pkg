#include <rl_env/energyrooms.hh>

/*
  EnergyRooms::EnergyRooms(Random &rand, const Gridworld *gridworld, bool stochastic):
  grid(gridworld), goal(coord_t(2.,2.)), noisy(stochastic), rng(rand),
  s(2),
  ns(s[0]),
  ew(s[1])
  {
  randomize_goal();
  reset();
  }
*/

EnergyRooms::EnergyRooms(Random &rand, bool negReward):
  grid(create_default_map()),
  goal(coord_t(1.,10.)),
  negReward(negReward),
  noisy(false),
  rng(rand),
  doorway(coord_t(2.,4.)),
  s(3),
  ns(s[0]),
  ew(s[1]),
  energy(s[2]),
  goalOption(false),
  fuel(false)
{
  reset();
  //cout << *this << endl;
}

EnergyRooms::EnergyRooms(Random &rand, bool stochastic, bool negReward,
                         bool goalOption):
  grid(create_default_map()),
  goal(coord_t(1.,10.)),
  negReward(negReward),
  noisy(stochastic),
  rng(rand),
  doorway(coord_t(2.,4.)),
  s(3),
  ns(s[0]),
  ew(s[1]),
  energy(s[2]),
  goalOption(goalOption),
  fuel(false)
{
  reset();
}

EnergyRooms::EnergyRooms(Random &rand, bool stochastic, bool negReward,
                         bool goalOption, bool fuel):
  grid(create_default_map()),
  goal(coord_t(1.,10.)),
  negReward(negReward),
  noisy(stochastic),
  rng(rand),
  doorway(coord_t(2.,4.)),
  s(3),
  ns(s[0]),
  ew(s[1]),
  energy(s[2]),
  goalOption(goalOption),
  fuel(fuel)
{
  reset();
}


/*
  EnergyRooms::EnergyRooms(Random &rand, unsigned width, unsigned height, bool stochastic):
  grid(new Gridworld(height, width, rand)),
  goal(coord_t(2.,2.)),
  noisy(stochastic), rng(rand),
  doorway(NULL),
  s(2),
  ns(s[0]),
  ew(s[1])
  {
  randomize_goal();
  reset();
  }
*/

EnergyRooms::~EnergyRooms() { delete grid; }

const std::vector<float> &EnergyRooms::sensation() const {
  //cout << "At state " << s[0] << ", " << s[1] << endl;

  return s;
}

float EnergyRooms::apply(int action) {

  //cout << "Taking action " << static_cast<room_action_t>(action) << endl;

  // 80% of the time, lose one energy
  if (!noisy || rng.bernoulli(0.8))
    energy--;

  // many fuel squares, with varying amounts of fuel and reward
  if (fuel){
    if ((int)ns % 3 == 0 && (int)ew % 3 == 0){
      energy += (int)(5 + ns/3 + (11-ew)/3);
    }
    if (energy > 20.0)
      energy = 20.0;
  }
  else {
    // certain squares reset you to 10
    if (ns == 7 && ew == 3)
      energy = 10;
    if (ns == 7 && ew == 7)
      energy = 10;
    if (ns == 3 && ew == 3)
      energy = 10;
    if (ns == 7 && ew == 7)
      energy = 10;
  }

  // never go below 0
  if (energy < 0.0)
    energy = 0;


  const room_action_t effect =
    noisy
    ? add_noise(static_cast<room_action_t>(action))
    : static_cast<room_action_t>(action);
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

  std::cerr << "Unreachable point reached in EnergyRooms::apply!!!\n";
  return 0; // unreachable, I hope
}


float EnergyRooms::reward() {

  if (negReward){
    // normally -1 and 0 on goal
    if (terminal())
      return 0;
    else if (energy <= 0.1){
      if (fuel)
        return -10;
      else
        return -2;
    }
    // normal square and normal energy
    else{
      // many squares of random cost
      if (fuel){
        if ((int)ns % 3 == 0 && (int)ew % 3 == 0)
          return -2 + (int)(-ew/5 -((int)ns%4));

        else
          return -1;
      }
      else
        return -1;
    }

  } // not negReward
  else{

    // or we could do 0 and 1 on goal
    if (terminal())
      return 1;
    else if (energy <= 0.1){
      if (fuel)
        return -10;
      else
        return -2;
    }else
      return 0;
  }
}


bool EnergyRooms::terminal() const {
  // current position equal to goal??
  return coord_t(ns,ew) == goal;
}


void EnergyRooms::reset() {
  // start randomly in upper left room (goal is lower right)
  ns = rng.uniformDiscrete(6, grid->height()-1);
  ew = rng.uniformDiscrete(0, 4);
  energy = 10;

  // if fuel, start with random amount of fuel
  if (fuel) {
    energy = rng.uniformDiscrete(1, 20);
  }

  //ns = 8;
  //ew = 2;

  //ns = 4;
  //ew = 9;
}


std::vector<std::vector<float> >  EnergyRooms::getSubgoals(){

  //cout << "Getting room subgoals " << endl;

  // Create vector of state representations, each is a subgoal
  std::vector<std::vector<float> > subgoals;


  std::vector<float> subgoal(2);

  // between two left rooms
  subgoal[0] = 5;
  subgoal[1] = 1;
  subgoals.push_back(subgoal);

  // between two right rooms
  subgoal[0] = 4;
  subgoal[1] = 8;
  subgoals.push_back(subgoal);

  // between two top rooms
  subgoal[0] = 8;
  subgoal[1] = 5;
  subgoals.push_back(subgoal);

  // between two lower rooms
  subgoal[0] = 1;
  subgoal[1] = 5;
  subgoals.push_back(subgoal);

  if (goalOption){
    // actual goal
    subgoal[0] = 1;
    subgoal[1] = 10;
    subgoals.push_back(subgoal);
  }

  return subgoals;

}


int EnergyRooms::getNumActions(){
  return 4;
}


std::ostream &operator<<(std::ostream &out, const EnergyRooms &rooms) {
  out << "Map:\n" << *rooms.grid;

  // print goal
  out << "Goal: row " << rooms.goal.first
      << ", column " << rooms.goal.second << "\n";

  // print doorway
  out << "Doorway: row " << rooms.doorway.first
      << ", column " << rooms.doorway.second << "\n";

  return out;
}

const Gridworld *EnergyRooms::create_default_map() {
  int width = 11;
  int height = 11;
  std::vector<std::vector<bool> > nsv(width, std::vector<bool>(height-1,false));
  std::vector<std::vector<bool> > ewv(height, std::vector<bool>(width-1,false));

  // put the vertical wall between the two rooms
  for (int j = 0; j < height; j++){
    // skip doorways at 1 and 8
    if (j == 1 || j == 8)
      continue;
    ewv[j][4] = true;
    ewv[j][5] = true;
  }

  nsv[5][0] = true;
  nsv[5][1] = true;
  nsv[5][7] = true;
  nsv[5][8] = true;

  // put the horizontal wall for the left room
  for (int i = 0; i < 6; i++){
    // skip doorway at 1
    if (i == 1)
      continue;
    nsv[i][4] = true;
    nsv[i][5] = true;
  }

  ewv[5][0] = true;
  ewv[5][1] = true;

  // put the horizontal wall for the right room
  for (int i = 5; i < width; i++){
    // skip doorway at 8
    if (i == 8)
      continue;
    nsv[i][3] = true;
    nsv[i][4] = true;
  }

  ewv[4][7] = true;
  ewv[4][8] = true;

  return new Gridworld(height, width, nsv, ewv);
}

EnergyRooms::room_action_t EnergyRooms::add_noise(room_action_t action) {
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


void EnergyRooms::randomize_goal() {
  const unsigned n = grid->height() * grid->width();
  unsigned index = rng.uniformDiscrete(1,n) - 1;
  goal = coord_t(index / grid->width(), index % grid->width());
}


void EnergyRooms::getMinMaxFeatures(std::vector<float> *minFeat,
                                    std::vector<float> *maxFeat){

  minFeat->resize(s.size(), 0.0);
  maxFeat->resize(s.size(), 10.0);

  (*maxFeat)[2] = 20.0;

}

void EnergyRooms::getMinMaxReward(float *minR,
                                 float *maxR){

  *minR = -10.0;
  *maxR = 1.0;

}
