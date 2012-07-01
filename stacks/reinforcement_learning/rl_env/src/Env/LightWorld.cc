/**
   The LightWorld domain from
   "Building Portable Options: Skill Transfer in Reinforcement Learning"
   by Konidaris and Barto
*/

#include <rl_env/LightWorld.hh>


LightWorld::LightWorld(Random &rand, bool stochastic, int nrooms):
  noisy(stochastic),
  nrooms(nrooms),
  rng(rand),
  s(17),
  ns(s[0]),
  ew(s[1]),
  have_key(s[2]),
  door_open(s[3]),
  room_id(s[4]),
  key_n(s[5]),
  key_e(s[6]),
  key_w(s[7]),
  key_s(s[8]),
  lock_n(s[9]),
  lock_e(s[10]),
  lock_w(s[11]),
  lock_s(s[12]),
  door_n(s[13]),
  door_e(s[14]),
  door_w(s[15]),
  door_s(s[16])
{

  LWDEBUG = false;
  MAX_SENSE = 10;


  totalVisited = 0;
  keyVisited = 0;
  lockVisited = 0;
  doorVisited = 0;
  haveKey = 0;
  doorOpen = 0;
  leaveRoom = 0;
  pressKey = 0;
  pressLockCorrect = 0;
  pressLockIncorrect = 0;
  pressDoor = 0;
  pressOther = 0;
  pickupKeyCorrect = 0;
  pickupKeyIncorrect = 0;
  pickupLock = 0;
  pickupDoor = 0;
  pickupOther = 0;

  reset();
}



LightWorld::~LightWorld() {  }


const std::vector<float> &LightWorld::sensation() const {
  if (LWDEBUG) print_map();
  return s;
}


int LightWorld::applyNoise(int action){
  switch(action) {
  case NORTH:
  case SOUTH:
    return rng.bernoulli(0.9) ? action : (rng.bernoulli(0.5) ? EAST : WEST);
  case EAST:
  case WEST:
    return rng.bernoulli(0.9) ? action : (rng.bernoulli(0.5) ? NORTH : SOUTH);
  case PRESS:
  case PICKUP:
    return rng.bernoulli(0.9) ? action : -1;
  default:
    return action;
  }
}

float LightWorld::apply(int origAction) {

  int reward = 0;

  int action = origAction;
  if (noisy)
    action = applyNoise(origAction);

  if (action == NORTH){
    if (((ns < rooms[room_id].height-2) || (rooms[room_id].lock_ns == rooms[room_id].height-1 && ew == rooms[room_id].lock_ew) || (rooms[room_id].door_ns == rooms[room_id].height-1 && ew == rooms[room_id].door_ew)) && ew > 0 && ew < rooms[room_id].width-1 && ns < rooms[room_id].height-1) {
      ns++;
    } else if (door_open && ns == rooms[room_id].door_ns && ew == rooms[room_id].door_ew && rooms[room_id].door_ns == rooms[room_id].height-1){
      leaveRoom++;
      room_id++;
      if (room_id >= nrooms) room_id = 0;
      have_key = false;
      door_open = false;
      if (rooms[room_id].key_ns < 0) have_key = true;
      ns = 0;
      resetKey();
      reward+=10;
    }
  }
  if (action == EAST){
    if (((ew < rooms[room_id].width-2) || (rooms[room_id].lock_ew == rooms[room_id].width-1 && ns == rooms[room_id].lock_ns) || (rooms[room_id].door_ew == rooms[room_id].width-1 && ns == rooms[room_id].door_ns)) && ns > 0 && ns < rooms[room_id].height-1 && ew < rooms[room_id].width-1) {
      ew++;
    } else if (door_open && ew == rooms[room_id].door_ew && ns == rooms[room_id].door_ns && rooms[room_id].door_ew == rooms[room_id].width-1){
      leaveRoom++;
      room_id++;
      if (room_id >= nrooms) room_id = 0;
      have_key = false;
      door_open = false;
      if (rooms[room_id].key_ns < 0) have_key = true;
      ew = 0;
      if (room_id == 1) ns = 3;
      resetKey();
      reward+=10;
    }
  }
  if (action == SOUTH){
    if (((ns > 1) || (rooms[room_id].lock_ns == 0 && ew == rooms[room_id].lock_ew) || (rooms[room_id].door_ns == 0 && ew == rooms[room_id].door_ew)) && ew > 0 && ew < rooms[room_id].width-1 && ns > 0) {
      ns--;
    } else if (door_open && ns == rooms[room_id].door_ns && ew == rooms[room_id].door_ew && rooms[room_id].door_ns == 0){
      leaveRoom++;
      room_id++;
      if (room_id >= nrooms) room_id = 0;
      have_key = false;
      door_open = false;
      if (rooms[room_id].key_ns < 0) have_key = true;
      ns = rooms[room_id].height-1;
      resetKey();
      reward+=10;
    }
  }
  if (action == WEST){
    if (((ew > 1) || (rooms[room_id].lock_ew == 0 && ns == rooms[room_id].lock_ns) || (rooms[room_id].door_ew == 0 && ns == rooms[room_id].door_ns)) && ns > 0 && ns < rooms[room_id].height-1 && ew > 0) {
      ew--;
    } else if (door_open && ew == rooms[room_id].door_ew && ns == rooms[room_id].door_ns && rooms[room_id].door_ew == 0){
      leaveRoom++;
      room_id++;
      if (room_id >= nrooms) room_id = 0;
      have_key = false;
      door_open = false;
      if (rooms[room_id].key_ns < 0) have_key = true;
      ew = rooms[room_id].width-1;
      resetKey();
      reward+=10;
    }
  }


  if (action == PICKUP){
    if (ns == rooms[room_id].key_ns && ew == rooms[room_id].key_ew){
      if (!have_key) pickupKeyCorrect++;
      else pickupKeyIncorrect++;
      have_key = true;
    }
    else if (ns == rooms[room_id].lock_ns && ew == rooms[room_id].lock_ew)
      pickupLock++;
    else if (ns == rooms[room_id].door_ns && ew == rooms[room_id].door_ew)
      pickupDoor++;
    else
      pickupOther++;
  }

  if (action == PRESS){
    if (ns == rooms[room_id].lock_ns && ew == rooms[room_id].lock_ew){
      if (have_key) {
        if (!door_open) pressLockCorrect++;
        else pressLockIncorrect++;
        door_open = true;
      } else {
        pressLockIncorrect++;
      }
    }
    else if (ns == rooms[room_id].key_ns && ew == rooms[room_id].key_ew)
      pressKey++;
    else if (ns == rooms[room_id].door_ns && ew == rooms[room_id].door_ew)
      pressDoor++;
    else
      pressOther++;
  }

  updateSensors();

  return reward;

}

void LightWorld::updateSensors() {

  // set all to 0
  key_n = 0;
  key_e = 0;
  key_w = 0;
  key_s = 0;
  lock_n = 0;
  lock_e = 0;
  lock_w = 0;
  lock_s = 0;
  door_n = 0;
  door_e = 0;
  door_w = 0;
  door_s = 0;

  if (!have_key){
    if (rooms[room_id].key_ns <= ns){
      key_s = MAX_SENSE - (ns - rooms[room_id].key_ns);
    }
    if (rooms[room_id].key_ns >= ns){
      key_n = MAX_SENSE - (rooms[room_id].key_ns - ns);
    }
    if (rooms[room_id].key_ew <= ew){
      key_w = MAX_SENSE - (ew - rooms[room_id].key_ew);
    }
    if (rooms[room_id].key_ew >= ew){
      key_e = MAX_SENSE - (rooms[room_id].key_ew - ew);
    }
  }

  if (door_open){
    if (rooms[room_id].door_ns <= ns){
      door_s = MAX_SENSE - (ns - rooms[room_id].door_ns);
    }
    if (rooms[room_id].door_ns >= ns){
      door_n = MAX_SENSE - (rooms[room_id].door_ns - ns);
    }
    if (rooms[room_id].door_ew <= ew){
      door_w = MAX_SENSE - (ew - rooms[room_id].door_ew);
    }
    if (rooms[room_id].door_ew >= ew){
      door_e = MAX_SENSE - (rooms[room_id].door_ew - ew);
    }
  }

  if (rooms[room_id].lock_ns <= ns){
    lock_s = MAX_SENSE - (ns - rooms[room_id].lock_ns);
  }
  if (rooms[room_id].lock_ns >= ns){
    lock_n = MAX_SENSE - (rooms[room_id].lock_ns - ns);
  }
  if (rooms[room_id].lock_ew <= ew){
    lock_w = MAX_SENSE - (ew - rooms[room_id].lock_ew);
  }
  if (rooms[room_id].lock_ew >= ew){
    lock_e = MAX_SENSE - (rooms[room_id].lock_ew - ew);
  }
}


void LightWorld::resetKey() {
  if (!have_key && rooms[room_id].key_ns > -1){
    rooms[room_id].key_ns = rng.uniformDiscrete(1, rooms[room_id].height-2);
    rooms[room_id].key_ew = rng.uniformDiscrete(1, rooms[room_id].width-2);
  }
}

void LightWorld::setKey(std::vector<float> testS){
  if (!have_key){
    float nsDist = 0;
    if (testS[5] > 0)
      nsDist = MAX_SENSE - testS[5];
    else
      nsDist = -MAX_SENSE + testS[8];
    rooms[room_id].key_ns = testS[0] + nsDist;
    float ewDist = 0;
    if (testS[6] > 0)
      ewDist = MAX_SENSE - testS[6];
    else
      ewDist = -MAX_SENSE + testS[7];
    rooms[room_id].key_ew = testS[1] + ewDist;
  }
}



bool LightWorld::terminal() const {
  // TODO: different terminal condition
  //return scream || (dangerous && (under_eye == 6 || under_hand == 6));
  return false;
}

void LightWorld::reset() {

  // init rooms
  rooms.resize(nrooms);

  rooms[0].height = 8;
  rooms[0].width = 7;
  rooms[0].key_ns = 3;
  rooms[0].key_ew = 1;
  rooms[0].lock_ns = 2;
  rooms[0].lock_ew = 6;
  rooms[0].door_ns = 5;
  rooms[0].door_ew = 6;

  rooms[1].height = 6;
  rooms[1].width = 5;
  rooms[1].key_ns = 3;
  rooms[1].key_ew = 2;
  rooms[1].lock_ns = 0;
  rooms[1].lock_ew = 3;
  rooms[1].door_ns = 0;
  rooms[1].door_ew = 1;

  rooms[2].height = 8;
  rooms[2].width = 5;
  rooms[2].key_ns = -1;
  rooms[2].key_ew = -1;
  rooms[2].lock_ns = 3;
  rooms[2].lock_ew = 4;
  rooms[2].door_ns = 3;
  rooms[2].door_ew = 0;

  rooms[3].height = 6;
  rooms[3].width = 7;
  rooms[3].key_ns = 1;
  rooms[3].key_ew = 1;
  rooms[3].lock_ns = 0;
  rooms[3].lock_ew = 4;
  rooms[3].door_ns = 5;
  rooms[3].door_ew = 2;

  for (int i = 4; i < nrooms; i++){
    rooms[i].height = i+3;
    rooms[i].width = i+2;
    rooms[i].key_ns = i;
    rooms[i].key_ew = i-1;
    rooms[i].lock_ns = 0;
    rooms[i].lock_ew = i-2;
    rooms[i].door_ns = 0;
    rooms[i].door_ew = i;
  }

  // random spot in first room
  room_id = 0;
  ns = rng.uniformDiscrete(1, rooms[0].height-2);
  ew = rng.uniformDiscrete(1, rooms[0].width-2);
  have_key = false;
  door_open = false;
  resetKey();
  updateSensors();

  if (LWDEBUG) print_map();

}


int LightWorld::getNumActions() {
  if (LWDEBUG) cout << "Return number of actions: " << NUM_ACTIONS << endl;
  return NUM_ACTIONS; //num_actions;
}


void LightWorld::print_map() const{
  // TODO: print map in rows, including symbols for all objects

  cout << "\nLightWorld, Room " << room_id << endl;

  // for each row
  for (int j = rooms[room_id].height-1; j >= 0; --j){
    // for each column
    for (int i = 0; i < rooms[room_id].width; i++){
      if (ns == j && ew == i) cout << "A";
      else if (j == rooms[room_id].key_ns && i == rooms[room_id].key_ew && !have_key) cout << "K";
      else if (j == rooms[room_id].lock_ns && i == rooms[room_id].lock_ew) cout << "L";
      else if (j == rooms[room_id].door_ns && i == rooms[room_id].door_ew) cout << "D";
      else if (j == 0 || i == 0 || j == rooms[room_id].height-1 || i == rooms[room_id].width-1) cout << "X";
      else cout << ".";
    } // last col of row
    cout << endl;
  } // last row

  cout << "at " << ns << ", " << ew << endl;
  cout << "Key: " << have_key << " door: "<< door_open << endl;
  cout << "NORTH: key: " << key_n << ", door: " << door_n << ", lock: " << lock_n << endl;
  cout << "EAST: key: " << key_e << ", door: " << door_e << ", lock: " << lock_e << endl;
  cout << "SOUTH: key: " << key_s << ", door: " << door_s << ", lock: " << lock_s << endl;
  cout << "WEST: key: " << key_w << ", door: " << door_w << ", lock: " << lock_w << endl;


}



void LightWorld::getMinMaxFeatures(std::vector<float> *minFeat,
                                   std::vector<float> *maxFeat){

  minFeat->resize(s.size(), 0.0);
  maxFeat->resize(s.size(), MAX_SENSE);

  // room id only goes to 2
  (*maxFeat)[4] = 2;

  // have_key and door_open are boolean
  (*maxFeat)[2] = 1;
  (*maxFeat)[3] = 1;

  // room sizes only go to 8
  (*maxFeat)[0] = 8;
  (*maxFeat)[1] = 8;

}

void LightWorld::getMinMaxReward(float *minR,
                                 float *maxR){

  *minR = 0.0;
  *maxR = 10.0;

}


