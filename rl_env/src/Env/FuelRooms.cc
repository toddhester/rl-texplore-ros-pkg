/** \file FuelRooms.cc
    Implements the Fuel World domain, with possible noise.
    From the paper:
    Hester and Stone, "Real Time Targeted Exploration in Large Domains", ICDL 2010.
    \author Todd Hester
*/

#include <rl_env/FuelRooms.hh>


FuelRooms::FuelRooms(Random &rand, bool extraVariation, bool stoch):
  height(20), width(30),
  goal(coord_t(11.0,24.0)), 
  extraVar(extraVariation),
  noisy(stoch),
  rng(rand),
  s(3),
  ns(s[0]),
  ew(s[1]),
  energy(s[2])
{

  fuelVisited = 0;
  totalVisited = 0;

  stateVisits = new int*[21];
  for (int i = 0; i < 21; i++){
    stateVisits[i] = new int[31];
  }

  reset();
  resetVisits();
}



FuelRooms::~FuelRooms() { 
  for (int i = 0; i < 21; i++){
    delete [] stateVisits[i];
  }
  delete [] stateVisits;
}

const std::vector<float> &FuelRooms::sensation() const { 
  return s; 
}

float FuelRooms::apply(int action) {

  checkVisits();

  if (terminal())
    return 0.0;

  //cout << "Taking action " << static_cast<room_action_t>(action) << endl;

  //cout << "state: " << s[0] << ", " << s[1] << ", " << s[2] 
  //    << " act: " << action << endl;

  // 20% lose none
  // 20% lose two
  // 80% of the time, lose one energy
  //  if (!noisy || rng.bernoulli(0.8))
  energy--;

  // many fuel squares, with varying amounts reward
  if ((int)ns == 0 || (int)ns == height){
    energy += 20.0;
  }
  

  if (energy > 60.0) 
    energy = 60.0; 

  const room_action_t effect =
    noisy
    ? add_noise(static_cast<room_action_t>(action)) 
    : static_cast<room_action_t>(action);

  float r = reward(effect);

  if (effect == NORTH || effect == NORTHWEST || effect == NORTHEAST)
    if (ns < height)
      ns++;

  if (effect == SOUTH || effect == SOUTHWEST || effect == SOUTHEAST)
    if (ns > 0)
      ns--;

  if (effect == EAST || effect == SOUTHEAST || effect == NORTHEAST)
    if (ew < width)
      ew++;

  if (effect == WEST || effect == SOUTHWEST || effect == NORTHWEST)
    if (ew > 0)
      ew--;
  
  return r;
  
  std::cerr << "Unreachable point reached in FuelRooms::apply!!!\n";
  return 0; // unreachable, I hope
}



float FuelRooms::reward(int effect) {
  
  if (energy < 0.0){
    return -400.0;
  }

  if (terminal()){
    //cout << "Found goal!!!!" << endl;
    return 0.0;
  }

  // extra cost at fuel stations
  /*
  if (ns == 0 || ns == height){
    return -5.0;
  }
  */

  if (ns == 0 || ns == height){
    float base = -10.0;
    if (ns == 0)
      base = -13.0;
  
    // extra variation
    float var = 1.0;
    if (extraVar)
      var = 5.0;
    else
      base -= 8.0;

    return base - (((int)ew % 5) * var);
      
  }
 
  if (effect == NORTH || effect == SOUTH || effect == EAST || effect == WEST)
    return -1.0;
  else
    return -1.4;
  
}


bool FuelRooms::terminal() const {
  // current position equal to goal??
  // or out of fuel
  return (coord_t(ns,ew) == goal) || (energy < 0.0);
}


void FuelRooms::reset() {
  // start randomly in left region
  ns = rng.uniformDiscrete(7, 12);
  ew = rng.uniformDiscrete(0, 4);

  // start with random amount of fuel
  // enough to get to gas stations, not enough to get to goal
  // gas stations up to 9 steps away, goal at least 20 steps away
  energy = rng.uniformDiscrete(14, 18);

}


void FuelRooms::resetVisits(){
  fuelVisited = 0;
  totalVisited = 0;

  for (int i = 0; i < 21; i++)
    for (int j = 0; j < 31; j++)
      stateVisits[i][j] = 0;
}

void FuelRooms::checkVisits(){
  stateVisits[(int)ns][(int)ew]++;
  // first visit to a state
  if (stateVisits[(int)ns][(int)ew] == 1){
    totalVisited++;
    if (ns == 0 || ns == 20)
      fuelVisited++;
  }
}

void FuelRooms::printVisits(){
  float totalStates = 31.0 * 21.0;
  float fuelStates = 31.0 * 2.0;
  float otherStates = totalStates - fuelStates;
  cout << (fuelVisited/fuelStates) << endl << ((totalVisited-fuelVisited)/otherStates) << endl << (totalVisited/totalStates) << endl;
}

void FuelRooms::printVisitMap(string filename){
 ofstream fout(filename.c_str());
  for (int i = 0; i < 21; i++){
    for (int j = 0; j < 31; j++){
      fout << stateVisits[i][j] << "\t";
    }
    fout << endl;
  }
  fout.close();
}


int FuelRooms::getNumActions(){
  return 8;
}


FuelRooms::room_action_t FuelRooms::add_noise(room_action_t action) {

  int newAct = rng.bernoulli(0.8) ? action : (rng.bernoulli(0.5) ? action+1 : action-1);

  if (newAct < 0)
    newAct = getNumActions()-1;
  if (newAct >= getNumActions())
    newAct = 0;

  return (room_action_t)newAct;
}



std::vector<experience> FuelRooms::getSeedings() {

  // return seedings
  std::vector<experience> seeds;

  // how many copies of each?
  for (int i = 0; i < 1; i++){

    // single seed of terminal state
    seeds.push_back(getExp(11,24,rng.uniformDiscrete(2,40),rng.uniformDiscrete(0,7)));
    
    // one seed from each fuel row
    seeds.push_back(getExp(0, rng.uniformDiscrete(1,29),rng.uniformDiscrete(2,40),rng.uniformDiscrete(0,7)));  
    seeds.push_back(getExp(20,rng.uniformDiscrete(1,29),rng.uniformDiscrete(2,40),rng.uniformDiscrete(0,7)));

    // seed of terminal
    seeds.push_back(getExp(rng.uniformDiscrete(1,19),rng.uniformDiscrete(1,22),0,rng.uniformDiscrete(0,7)));
  }

    /*
  // two seeds around the goal state
  seeds.push_back(getExp(10,24,4,NORTH));
  seeds.push_back(getExp(11,25,42,EAST));

  // one of death
  seeds.push_back(getExp(9,15,0,SOUTHEAST));
  */

  // lots of seeds of various shit
  /*
    for (int i = 0; i < 3; i++){

    // each wall
    //seeds.push_back(getExp(0,11,10,SOUTH));
    //seeds.push_back(getExp(0,27,14,SOUTH));
    //seeds.push_back(getExp(0,17,10,NORTH));
    
    //seeds.push_back(getExp(20,12,20,SOUTH));
    //seeds.push_back(getExp(20,28,24,NORTH));
    //seeds.push_back(getExp(20,18,20,NORTH));
    
    //seeds.push_back(getExp(10,30,30,EAST));
    //seeds.push_back(getExp(12,30,34,EAST));
    //seeds.push_back(getExp(10,30,30,WEST));
    
    //seeds.push_back(getExp(13,0,2,WEST));
    //seeds.push_back(getExp(17,0,4,WEST));
    //seeds.push_back(getExp(13,0,2,EAST));
    
    // experiences showing where the goal state is (11,24)
    seeds.push_back(getExp(10,24,20,NORTH));
    seeds.push_back(getExp(12,24,44,SOUTH));
    seeds.push_back(getExp(11,23,52,EAST));
    seeds.push_back(getExp(11,25,7,WEST));
    seeds.push_back(getExp(10,23,11,NORTHEAST));
    seeds.push_back(getExp(10,25,16,NORTHWEST));
    seeds.push_back(getExp(12,23,21,SOUTHEAST));
    seeds.push_back(getExp(12,25,36,SOUTHWEST));

    // near the goal state
    seeds.push_back(getExp(11,23,45,NORTH));
    seeds.push_back(getExp(10,24,31,NORTHEAST));
    seeds.push_back(getExp(11,25,11,SOUTH));
    seeds.push_back(getExp(12,24,18,WEST));
    seeds.push_back(getExp(10,23,18,SOUTHWEST));

    // a few normal
    seeds.push_back(getExp(17,14,52,SOUTH));
    seeds.push_back(getExp(1,6,43,EAST));
    seeds.push_back(getExp(9,18,24,NORTH));
    seeds.push_back(getExp(12,8,3,WEST));
    seeds.push_back(getExp(7,1,42,SOUTHEAST));
    seeds.push_back(getExp(6,9,7,NORTHEAST));
    seeds.push_back(getExp(19,28,28,NORTHWEST));
    seeds.push_back(getExp(2,18,33,SOUTHWEST));

    // actions do different things from one state!
    seeds.push_back(getExp(9,3,19,SOUTHEAST));
    seeds.push_back(getExp(9,3,19,EAST));
    seeds.push_back(getExp(9,3,19,NORTHWEST));
    seeds.push_back(getExp(9,3,19,WEST));
    seeds.push_back(getExp(9,3,19,NORTH));
    seeds.push_back(getExp(9,3,19,SOUTHWEST));

    // and fuel running out
    seeds.push_back(getExp(7,4,0,NORTH));
    seeds.push_back(getExp(19,21,0,SOUTHWEST));
    seeds.push_back(getExp(3,16,0,NORTHEAST));
    seeds.push_back(getExp(13,1,0,SOUTH));
    
    // general gas stations 
    seeds.push_back(getExp(0,5,12,EAST));
    seeds.push_back(getExp(0,23,9,WEST));
    seeds.push_back(getExp(0,26,3,NORTHWEST));
    seeds.push_back(getExp(20,7,7,SOUTHEAST));
    seeds.push_back(getExp(20,18,4,SOUTH));
    seeds.push_back(getExp(20,25,4,SOUTHWEST));

    // terminal states
    seeds.push_back(getExp(3,24,-1,WEST));
    seeds.push_back(getExp(9,14,-1,SOUTH));
    seeds.push_back(getExp(7,4,-1,EAST));
    seeds.push_back(getExp(14,18,-1,NORTH));
    seeds.push_back(getExp(7,23,-1,NORTHWEST));
    seeds.push_back(getExp(16,5,-1,SOUTHWEST));
    seeds.push_back(getExp(17,14,-1,NORTHEAST));
    seeds.push_back(getExp(4,28,-1,SOUTHEAST));

    seeds.push_back(getExp(11,24,12,NORTH));
    seeds.push_back(getExp(11,24,22,WEST));
    seeds.push_back(getExp(11,24,32,EAST));
    seeds.push_back(getExp(11,24,2,SOUTH));
    seeds.push_back(getExp(11,24,17,NORTHWEST));
    seeds.push_back(getExp(11,24,27,SOUTHWEST));
    seeds.push_back(getExp(11,24,37,NORTHEAST));
    seeds.push_back(getExp(11,24,7,SOUTHEAST));
  }
  */

  /*
  // a bunch of random seeds
  for (int j = 0; j < 1000; j++){
    seeds.push_back(getExp(rng.uniformDiscrete(0,20),rng.uniformDiscrete(0,30),rng.uniformDiscrete(-1,60),rng.uniformDiscrete(0,3)));
  }
  */

  reset();
  resetVisits();

  return seeds;

}


experience FuelRooms::getExp(int s0, int s1, int s2, int a){

  experience e;

  e.s.resize(3, 0.0);
  e.next.resize(3, 0.0);

  ns = s0;
  ew = s1;
  energy = s2;
  e.act = a;
  e.s = sensation();
  e.reward = apply(e.act);

  e.terminal = terminal();
  e.next = sensation();

  /*
  cout << "Seed experience from state (" << e.s[0] << ", "
       << e.s[1] << ", " << e.s[2] << ") action: " << e.act
       << " to (" << e.next[0] << ", " << e.next[1] << ", " << e.next[2] 
       << ") with reward " << e.reward << " and term: " << e.terminal << endl;
  */

  return e;
}

void FuelRooms::getMinMaxFeatures(std::vector<float> *minFeat,
                                    std::vector<float> *maxFeat){
  
  minFeat->resize(s.size(), 0.0);
  maxFeat->resize(s.size(), 10.0);

  (*maxFeat)[0] = 20.0;
  (*maxFeat)[1] = 30.0;
  (*maxFeat)[2] = 60.0;
  (*minFeat)[2] = -1.0;
}

void FuelRooms::getMinMaxReward(float *minR,
                               float *maxR){
  
  *minR = -400.0;
  *maxR = 0.0;    
  
}
