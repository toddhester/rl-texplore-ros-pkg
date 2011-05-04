#include <rl_env/fourrooms.hh>

/*
FourRooms::FourRooms(Random &rand, const Gridworld *gridworld, bool stochastic):
  grid(gridworld), goal(coord_t(2.,2.)), noisy(stochastic), rng(rand),
  s(2),
  ns(s[0]),
  ew(s[1])
{
  randomize_goal();
  reset();
}
*/

 
FourRooms::FourRooms(Random &rand):
  grid(create_default_map()),
  goal(coord_t(1.,10.)), 
  negReward(true),
  noisy(false),
  extraReward(false),
  rewardSensor(false),
  rng(rand),
  doorway(coord_t(2.,4.)),
  s(2),
  trash(6),
  ns(s[0]),
  ew(s[1]),
  distN(trash[0]),
  distS(trash[1]),
  distE(trash[2]),
  distW(trash[3]),
  rewardEW(trash[4]),
  rewardNS(trash[5]),
  goalOption(false)
{
  reset();
  //cout << *this << endl;
}
 

FourRooms::FourRooms(Random &rand, bool stochastic, bool negReward, 
		     bool exReward):
  grid(create_default_map()),
  goal(coord_t(1.,10.)), 
  negReward(negReward),
  noisy(stochastic),
  extraReward(exReward),
  rewardSensor(false),
  rng(rand),
  doorway(coord_t(2.,4.)),
  s(2),
  trash(6),
  ns(s[0]),
  ew(s[1]),
  distN(trash[0]),
  distS(trash[1]),
  distE(trash[2]),
  distW(trash[3]),
  rewardEW(trash[4]),
  rewardNS(trash[5]),
  goalOption(goalOption)
{
  reset();
}

// Create the version with extra state features for wall distances
FourRooms::FourRooms(Random &rand, bool stochastic, bool negReward):
  grid(create_default_map()),
  goal(coord_t(1.,10.)), 
  negReward(negReward),
  noisy(stochastic),
  extraReward(true), //false),
  rewardSensor(false),
  rng(rand),
  doorway(coord_t(2.,4.)),
  s(6),
  trash(2),
  ns(s[0]),
  ew(s[1]),
  distN(s[2]),
  distS(s[3]),
  distE(s[4]),
  distW(s[5]),
  rewardEW(trash[0]),
  rewardNS(trash[1]),
  goalOption(false)
{
  reset();
}


// Create the version with extra state features for wall distances and 
// reward distance
FourRooms::FourRooms(Random &rand, bool stochastic):
  grid(create_default_map()),
  goal(coord_t(1.,10.)),
  negReward(true),
  noisy(stochastic),
  extraReward(false),
  rewardSensor(false),
  rng(rand),
  doorway(coord_t(2.,4.)),
  s(8),
  trash(0),
  ns(s[0]),
  ew(s[1]),
  distN(s[2]),
  distS(s[3]),
  distE(s[4]),
  distW(s[5]),
  rewardEW(s[6]),
  rewardNS(s[7]),
  goalOption(false)
{
  //  cout <<  "Four room with wall dist and reward sensor" << endl;
  reset();
}



/*
FourRooms::FourRooms(Random &rand, unsigned width, unsigned height, bool stochastic):
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

FourRooms::~FourRooms() { delete grid; }

const std::vector<float> &FourRooms::sensation() const { 
  //cout << "At state " << s[0] << ", " << s[1] << endl;

  return s; 
}

float FourRooms::apply(int action) {

  //cout << "Taking action " << static_cast<room_action_t>(action) << endl;

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
	calcWallDistances();
      }
    return reward(effect);
  case SOUTH:
    if (!grid->wall(static_cast<unsigned>(ns),
		    static_cast<unsigned>(ew),
		    effect))
      {
	--ns;
	calcWallDistances();
      }
    return reward(effect);
  case EAST:
    if (!grid->wall(static_cast<unsigned>(ns),
		    static_cast<unsigned>(ew),
		    effect))
      {
	++ew;
	calcWallDistances();
      }
    return reward(effect);
  case WEST:
    if (!grid->wall(static_cast<unsigned>(ns),
		    static_cast<unsigned>(ew),
		    effect))
      {
	--ew;
	calcWallDistances();
      }
    return reward(effect);
  }
  std::cerr << "Unreachable point reached in FourRooms::apply!!!\n";
  return 0; // unreachable, I hope
}


float FourRooms::reward(int effect) {
  
  if (extraReward){
    // 0 on goal
    if (terminal())
      return 0;

    // 0 when heading right dir towards door
    // towards top middle door
    if (ew < 6 && ns == 8 && effect == EAST){
      return -1;
    }

    // towards left door
    if (ew == 1 && ns > 4 && effect == SOUTH){
      return -1;
    }

    // towards right door
    if (ew == 8 && ns > 3 && effect == SOUTH){
      return -1;
    }

    // towrads bottom door
    if (ew < 6 && ns == 1 && effect == EAST){
      return -1;
    }

    // 0 when heading towards goal
    if (ns == 1 && effect == EAST){
      return -1;
    }
    if (ew == 10 && ns < 4 && effect == SOUTH){
      return -1;
    }

    // normally -2
    return -2;

  }


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


bool FourRooms::terminal() const {
  // current position equal to goal??
  return coord_t(ns,ew) == goal;
}


void FourRooms::calcWallDistances(){

  // calculate distances East and West
  // if we're not in the same row as a doorway
  if (ns != 1 && ns != 8){
    // left side of wall
    if (ew < 5){
      distW = ew;
      distE = 4 - ew;
    }
    // right side of wall
    else {
      distW = ew - 6;
      distE = 10 - ew;
    }
  } 
  // doorway
  else {
    distW = ew;
    distE = 10 - ew;
  }

  // in a vertical doorway
  if (ns == 5 && ew == 1){
    distW = 0;
    distE = 0;
  }
  if (ns == 4 && ew == 8){
    distW = 0;
    distE = 0;
  }

  // calculate NS
  // left side
  if (ew < 5){
    // not in doorway column
    if (ew != 1){
      // top room
      if (ns > 5){
	distN = 10 - ns;
	distS = ns - 6;
      }
      // bottom room
      else {
	distN = 4 - ns;
	distS = ns;
      }
    }
    // doorway column
    else {
      distN = 10 - ns;
      distS = ns;
    }
  }
  // right side
  else {
    // not in doorway column
    if (ew != 8){
      // top room
      if (ns > 4){
	distN = 10-ns;
	distS = ns - 5;
      }
      // bottom room
      else {
	distN = 3 - ns;
	distS = ns;
      }
    }
    // doorway column
    else {
      distN = 10-ns;
      distS = ns;
    }
  }

  // in horiz doorway
  if (ew == 5 && (ns == 1 || ns == 8)){
    distN = 0;
    distS = 0;
  }


  // calculate reward distances
  // can see it e/w
  if (ns == 1){
    rewardEW = 10 - ew;
  }
  else {
    rewardEW = 100;
  }

  // can see ns
  if (ew == 10 && ns < 4){
    rewardNS = 1 - ns;
  }
  else {
    rewardNS = 100;
  }
  
  /*
  cout << "x,y: " << ew << ", " << ns << " N,S,E,W: " 
       << distN << ", " << distS << ", " 
       << distE << ", " << distW << " reward EW, NS: " 
       << rewardEW << ", " << rewardNS << endl;
  */
}


void FourRooms::reset() {
  // start randomly in upper left room (goal is lower right)
  ns = rng.uniformDiscrete(6, grid->height()-1);
  ew = rng.uniformDiscrete(0, 4);

  //ns = 8;
  //ew = 2;

  //ns = 4;
  //ew = 9;

  calcWallDistances();
}


std::vector<std::vector<float> >  FourRooms::getSubgoals(){

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


int FourRooms::getNumActions(){
  return 4;
}


std::ostream &operator<<(std::ostream &out, const FourRooms &rooms) {
  out << "Map:\n" << *rooms.grid;

  // print goal
  out << "Goal: row " << rooms.goal.first
      << ", column " << rooms.goal.second << "\n";

  // print doorway
  out << "Doorway: row " << rooms.doorway.first
      << ", column " << rooms.doorway.second << "\n";

  return out;
}

const Gridworld *FourRooms::create_default_map() {
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

FourRooms::room_action_t FourRooms::add_noise(room_action_t action) {
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


void FourRooms::randomize_goal() {
  const unsigned n = grid->height() * grid->width();
  unsigned index = rng.uniformDiscrete(1,n) - 1;
  goal = coord_t(index / grid->width(), index % grid->width());
}


/** For special use to test true transitions */
void FourRooms::setSensation(std::vector<float> newS){
  if (s.size() != newS.size()){
    cerr << "Error in sensation sizes" << endl;
  }

  for (unsigned i = 0; i < newS.size(); i++){
    s[i] = (int)newS[i];
  }
}


void FourRooms::getMinMaxFeatures(std::vector<float> *minFeat,
                                  std::vector<float> *maxFeat){
  
  minFeat->resize(s.size(), 0.0);
  maxFeat->resize(s.size(), 10.0);
  
  if (s.size() > 2) {
    for (unsigned i = 2; i < s.size(); i++){
      (*minFeat)[i] = -10.0;
    }
  }

}

void FourRooms::getMinMaxReward(float *minR,
                               float *maxR){
  
  if (extraReward){
    *minR = -2.0;
    *maxR = 0.0;
  }
  else if (negReward){
    *minR = -1.0;
    *maxR = 0.0;    
  }else{
    *minR = 0.0;
    *maxR = 1.0;
  }

}
