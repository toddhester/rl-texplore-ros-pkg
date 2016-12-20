#include "Stump.hh"



Stump::Stump(int id, int trainMode, int trainFreq, int m, float featPct, Random rng):
  id(id), mode(trainMode), freq(trainFreq), M(m), featPct(featPct), rng(rng)
{

  nOutput = 0;
  nExperiences = 0;

  // how close a split has to be to be randomly selected
  SPLIT_MARGIN = 0.02; //5; //01; //0.05; //0.2; //0.05;

  LOSS_MARGIN = 0.0; //0.02;
  REBUILD_RATIO = 0.05;

  MIN_GAIN_RATIO = 0.0001; //0.0004; //0.001; //0.0002; //0.001;

  ALLOW_ONLY_SPLITS = true; //false;

  STDEBUG = false;
  SPLITDEBUG = false; //true;

  if (STDEBUG) {
    cout << "Created decision stump " << id << endl;
    cout << " mode: " << mode
         << " freq: " << freq << endl;
  }

  initStump();

}

Stump::Stump(const Stump& s):
id(s.id), mode(s.mode), freq(s.freq), M(s.M), featPct(s.featPct), rng(s.rng)
{
  nOutput = s.nOutput;
  nExperiences = s.nExperiences;
  SPLIT_MARGIN = s.SPLIT_MARGIN;
  LOSS_MARGIN = s.LOSS_MARGIN;
  REBUILD_RATIO = s.REBUILD_RATIO;
  MIN_GAIN_RATIO = s.MIN_GAIN_RATIO;
  ALLOW_ONLY_SPLITS = s.ALLOW_ONLY_SPLITS;
  STDEBUG = s.STDEBUG;
  SPLITDEBUG = s.SPLITDEBUG;

  for (int i = 0; i < N_STUMP_EXP; i++){
    allExp[i] = s.allExp[i];
  }

  // set experience pointers
  experiences.resize(s.experiences.size());
  for (unsigned i = 0; i < s.experiences.size(); i++){
    experiences[i] = &(allExp[i]);
  }

  // actual split
  dim = s.dim;
  val = s.val;
  type = s.type;
  gainRatio = s.gainRatio;
  leftOutputs = s.leftOutputs;
  rightOutputs = s.rightOutputs;
}

Stump* Stump::getCopy(){
  Stump* copy = new Stump(*this);
  return copy;
}

Stump::~Stump() {
  for (unsigned i = N_STUMP_EXP; i < experiences.size(); i++){
    delete experiences[i];
  }
  experiences.clear();
}


// multiple instances at once
bool Stump::trainInstances(std::vector<classPair> &instances){
  if (STDEBUG) cout << "trainInstances: " << instances.size() << endl;

  bool modelChanged = false;
  bool doBuild = false;

  // loop through instances, possibly checking for errors
  for (unsigned a = 0; a < instances.size(); a++){
    classPair instance = instances[a];

    // take from static array until we run out
    stump_experience *e;
    if (nExperiences < N_STUMP_EXP){
      // from statically created set of experiences
      e = &(allExp[nExperiences]);

    } else {
      // dynamically create experience
      e = new stump_experience;
    }

    e->input = instance.in;
    e->output = instance.out;
    e->id = nExperiences;
    experiences.push_back(e);
    nExperiences++;

    if (nExperiences == 1000000){
      cout << "Reached limit of # experiences allowed." << endl;
      return false;
    }

    if (nExperiences != (int)experiences.size())
      cout << "ERROR: experience size mismatch: " << nExperiences << ", " << experiences.size() << endl;

    //cout << nExperiences << endl << flush;
    //if (nExperiences == 503 && id == 10){
    //  STDEBUG = true;
    //  SPLITDEBUG = true;
    //  INCDEBUG = true;
    //}

    if (STDEBUG) {
      cout << "Original input: ";
      for (unsigned i = 0; i < instance.in.size(); i++){
        cout << instance.in[i] << ", ";
      }
      cout << endl << " Original output: " << instance.out << endl;
      cout << "Added exp id: " << e->id << " output: " << e->output << endl;
      cout << "Address: " << e << " Input : ";
      for (unsigned i = 0; i < e->input.size(); i++){
        cout << e->input[i] << ", ";
      }
      cout << endl << " Now have " << nExperiences << " experiences." << endl;
    }

    // mode 0: re-build every step
    if (mode == 0 || mode == 1 || nExperiences <= 1){
      // build every step
      doBuild = true;
    }

    // mode 2: re-build every FREQ steps
    else if (mode == 2){
      // build every freq steps
      if (!modelChanged && (nExperiences % freq) == 0){
        doBuild = true;
      }
    }

  } // end instance loop

  if (doBuild){
    buildStump();
    modelChanged = true;
  }

  if (modelChanged){
    if (STDEBUG) cout << "ST " << id << " stump re-built." << endl;

    if (STDEBUG){
      cout << endl << "ST: " << id << endl;
      printStump();
      cout << "Done printing stump" << endl;
    }
  }

  return modelChanged;
}


// here the target output will be a single value
bool Stump::trainInstance(classPair &instance){
  if (STDEBUG) cout << "trainInstance" << endl;

  bool modelChanged = false;

  // simply add this instance to the set of experiences

  // take from static array until we run out
  stump_experience *e;
  if (nExperiences < N_STUMP_EXP){
    // from statically created set of experiences
    e = &(allExp[nExperiences]);

  } else {
    // dynamically create experience
    e = new stump_experience;
  }


  e->input = instance.in;
  e->output = instance.out;
  e->id = nExperiences;
  experiences.push_back(e);
  nExperiences++;

  if (nExperiences == 1000000){
    cout << "Reached limit of # experiences allowed." << endl;
    return false;
  }

  if (nExperiences != (int)experiences.size())
    cout << "ERROR: experience size mismatch: " << nExperiences << ", " << experiences.size() << endl;

  //cout << nExperiences << endl << flush;
  //if (nExperiences == 503 && id == 10){
  //  STDEBUG = true;
  //  SPLITDEBUG = true;
  //  INCDEBUG = true;
  //}

  if (STDEBUG) {
    cout << "Original input: ";
    for (unsigned i = 0; i < instance.in.size(); i++){
      cout << instance.in[i] << ", ";
    }
    cout << endl << " Original output: " << instance.out << endl;
    cout << "Added exp id: " << e->id << " output: " << e->output << endl;
    cout << "Address: " << e << " Input : ";
    for (unsigned i = 0; i < e->input.size(); i++){
      cout << e->input[i] << ", ";
    }
    cout << endl << " Now have " << nExperiences << " experiences." << endl;
  }

  // depending on mode/etc, maybe re-build stump

  // mode 0: re-build every step
  if (mode == 0 || mode == 1 || nExperiences <= 1){

    // build every step
    buildStump();
    modelChanged = true;

  }

  // mode 2: re-build every FREQ steps
  else if (mode == 2){
    // build every freq steps
    if (!modelChanged && (nExperiences % freq) == 0){
      buildStump();
      modelChanged = true;

    }
  }

  if (modelChanged){
    if (STDEBUG) cout << "ST " << id << " stump re-built." << endl;

    if (STDEBUG){
      cout << endl << "ST: " << id << endl;
      printStump();
      cout << "Done printing stump" << endl;
    }
  }

  return modelChanged;

}



// TODO: here we want to return the probability of the output value being each of the possible values, in the stochastic case
void Stump::testInstance(const std::vector<float> &input, std::map<float, float>* retval){
  if (STDEBUG) cout << "testInstance ST: " << id << endl;

  retval->clear();

  // in case the stump is empty
  if (experiences.size() == 0){
    (*retval)[0.0] = 1.0;
    if (STDEBUG) cout << "no experiences, return 1.0 prob of 0.0" << endl;
    return; 
  }

  // and return mapping of outputs and their probabilities
  if (passTest(dim, val, type, input))
    outputProbabilities(leftOutputs, retval);
  else
    outputProbabilities(rightOutputs, retval);
}

float Stump::getConf(const std::vector<float> &input){
  if (STDEBUG) cout << "numVisits" << endl;

  // in case the stump is empty
  if (experiences.size() == 0){
    return 0;
  }

  // and return #
  float conf = (float)nExperiences / (float)(2.0 *M);
  if (conf > 1.0)
    return 1.0;
  else
    return conf;

}


int Stump::findMatching(const std::vector<stump_experience*> &instances, int dim,
                        int val, int minConf){

  int count = 0;

  for (unsigned i = 0; i < instances.size(); i++){
    if (!passTest(dim, val, ONLY, instances[i]->input)){
      count++;
      // no need to continue if this won't be the new min
      if (count >= minConf)
        return count;
    }
  }
  return count;
}




// init the stump
void Stump::initStump(){
  if (STDEBUG) cout << "initStump()" << endl;

  dim = -1;
  val = -1;
  type = -1;

  // current data
  leftOutputs.clear();
  rightOutputs.clear();

  // just to ensure the diff models are on different random values
  for (int i = 0; i < id; i++){
    rng.uniform(0, 1);
  }

}


/** Decide if this passes the test */
bool Stump::passTest(int dim, float val, int type, const std::vector<float> &input){
  if (STDEBUG) cout << "passTest, dim=" << dim << ",val=" << val << ",type=" << type
                    << ",input["<<dim<<"]=" << input[dim] <<endl;

  // TODO: deal with categorical attributes in addition to continuous ones

  if (type == CUT){
    if (input[dim] > val)
      return false;
    else
      return true;
  } else if (type == ONLY){
    if (input[dim] == val)
      return false;
    else
      return true;
  } else {
    return false;
  }

}


/** Build the stump from this node down using this set of experiences. */
void Stump::buildStump(){
  if(STDEBUG) cout << "buildStump" << endl;

  if (experiences.size() == 0){
    cout << "Error: buildStump called on stump " << id << " with no instances." << endl;
    exit(-1);
  }

  if (SPLITDEBUG) cout << endl << "Creating new decision node" << endl;

  //node->nInstances++;

  float bestGainRatio = -1.0;
  int bestDim = -1;
  float bestVal = -1;
  int bestType = -1;

  testPossibleSplits(&bestGainRatio, &bestDim, &bestVal, &bestType);

  implementSplit(bestGainRatio, bestDim, bestVal, bestType);

}


void Stump::implementSplit(float bestGainRatio, int bestDim,
                           float bestVal, int bestType){
  if (STDEBUG) cout << "implementSplit gainRatio=" << bestGainRatio
                    << ",dim=" << bestDim
                    << ",val=" << bestVal << ",type=" << bestType << endl;


  // set the best split here
  dim = bestDim;
  val = bestVal;
  type = bestType;
  gainRatio = bestGainRatio;
  if (SPLITDEBUG) cout << "Best split was type " << type
                       << " with val " << val
                       << " on dim " << dim
                       << " with gainratio: " << bestGainRatio << endl;


  // split up the instances based on this split
  std::vector<stump_experience*> left;
  std::vector<stump_experience*> right;
  leftOutputs.clear();
  rightOutputs.clear();
  for (unsigned i = 0; i < experiences.size(); i++){
    if (STDEBUG) cout << "implmentSplit - Classify instance " << i << " id: "
                      << experiences[i]->id << " on new split " << endl;
    if (passTest(dim, val, type, experiences[i]->input)){
      left.push_back(experiences[i]);
      leftOutputs.insert(experiences[i]->output);
    }
    else{
      right.push_back(experiences[i]);
      rightOutputs.insert(experiences[i]->output);
    }
  }

  if (STDEBUG) cout << "Left has " << left.size()
                    << ", right has " << right.size() << endl;

}



void Stump::testPossibleSplits(float *bestGainRatio, int *bestDim,
                               float *bestVal, int *bestType) {
  if (STDEBUG) cout << "testPossibleSplits" << endl;


  // pre-calculate some stuff for these splits (namely I, P, C)
  float I = calcIforSet(experiences);
  //if (STDEBUG) cout << "I: " << I << endl;

  int nties = 0;

  // for each possible split, calc gain ratio
  for (unsigned idim = 0; idim < experiences[0]->input.size(); idim++){

    // we eliminate some random number of splits
    // here (decision is taken from the random set that are left)
    if (rng.uniform() < featPct)
      continue;


    float* sorted = sortOnDim(idim);

    for (int j = 0; j < (int)experiences.size(); j++){

      // splits that are cuts
      // if different from last value, try split in between
      if (j > 0 && sorted[j] != sorted[j-1]){
        float splitval = (sorted[j] + sorted[j-1]) / 2.0;
        float gainRatio = calcGainRatio(idim, splitval, CUT, I);

        if (SPLITDEBUG) cout << " CUT split val " << splitval
                             << " on dim: " << idim << " had gain ratio "
                             << gainRatio << endl;

        // see if this is the new best gain ratio
        compareSplits(gainRatio, idim, splitval, CUT, &nties,
                      bestGainRatio, bestDim, bestVal, bestType);


      } // if its a possible cut

      if (ALLOW_ONLY_SPLITS){
        // splits that are true only if this value is equal
        if (j == 0 || sorted[j] != sorted[j-1]){
          float splitval = sorted[j];

          float gainRatio = calcGainRatio(idim, splitval, ONLY, I);

          if (SPLITDEBUG) cout << " ONLY split val " << splitval
                               << " on dim: " << idim << " had gain ratio "
                               << gainRatio << endl;

          // see if this is the new best gain ratio
          compareSplits(gainRatio, idim, splitval, ONLY, &nties,
                        bestGainRatio, bestDim, bestVal, bestType);

        } // splits with only
      }

    } // j loop
    delete[] sorted;

  }
}


/** Decide if this split is better. */
void Stump::compareSplits(float gainRatio, int dim, float val, int type,
                          int *nties, float *bestGainRatio, int *bestDim,
                          float *bestVal, int *bestType){
  if (STDEBUG) cout << "compareSplits gainRatio=" << gainRatio << ",dim=" << dim
                    << ",val=" << val << ",type= " << type <<endl;


  bool newBest = false;

  // if its a virtual tie, break it randomly
  if (fabs(*bestGainRatio - gainRatio) < SPLIT_MARGIN){
    //cout << "Split tie, making random decision" << endl;

    (*nties)++;
    float randomval = rng.uniform(0,1);
    float newsplitprob = (1.0 / (float)*nties);

    if (randomval < newsplitprob){
      newBest = true;
      if (SPLITDEBUG) cout << "   Tie on split. ST: " << id << " rand: " << randomval
                           << " splitProb: " << newsplitprob << ", selecting new split " << endl;
    }
    else
      if (SPLITDEBUG) cout << "   Tie on split. ST: " << id << " rand: " << randomval
                           << " splitProb: " << newsplitprob << ", staying with old split " << endl;
  }

  // if its clearly better, set this as the best split
  else if (gainRatio > *bestGainRatio){
    newBest = true;
    *nties = 1;
  }


  // set the split features
  if (newBest){
    *bestGainRatio = gainRatio;
    *bestDim = dim;
    *bestVal = val;
    *bestType = type;
    if (SPLITDEBUG){
      cout << "  New best gain ratio: " << *bestGainRatio
           << ": type " << *bestType
           << " with val " << *bestVal
           << " on dim " << *bestDim << endl;
    }
  } // newbest
}

/** Calculate gain ratio for this possible split. */
float Stump::calcGainRatio(int dim, float val, int type,
                           float I){
  if (STDEBUG) cout << "calcGainRatio, dim=" << dim
                    << " val=" << val
                    << " I=" << I
                    << " nInstances= " << experiences.size() << endl;

  std::vector<stump_experience*> left;
  std::vector<stump_experience*> right;


  // array with percentage positive and negative for this test
  float D[2];

  // info(T) = I(P): float I;

  // Info for this split = Info(X,T)
  float Info;

  // Gain for this split = Gain(X,T)
  float Gain;

  // SplitInfo for this split = I(|pos|/|T|, |neg|/|T|)
  float SplitInfo;

  // GainRatio for this split = GainRatio(X,T) = Gain(X,T) / SplitInfo(X,T)
  float GainRatio;

  // see where the instances would go with this split
  for (unsigned i = 0; i < experiences.size(); i++){
    if (STDEBUG) cout << "calcGainRatio - Classify instance " << i << " id: "
                      << experiences[i]->id << " on new split " << endl;

    if (passTest(dim, val, type, experiences[i]->input)){
      left.push_back(experiences[i]);
    }
    else{
      right.push_back(experiences[i]);
    }
  }

  if (STDEBUG) cout << "Left has " << left.size()
                    << ", right has " << right.size() << endl;

  D[0] = (float)left.size() / (float)experiences.size();
  D[1] = (float)right.size() / (float)experiences.size();
  float leftInfo = calcIforSet(left);
  float rightInfo = calcIforSet(right);
  Info = D[0] * leftInfo + D[1] * rightInfo;
  Gain = I - Info;
  SplitInfo = calcIofP((float*)&D, 2);
  GainRatio = Gain / SplitInfo;

  if (STDEBUG){
    cout << "LeftInfo: " << leftInfo
         << " RightInfo: " << rightInfo
         << " Info: " << Info
         << " Gain: " << Gain
         << " SplitInfo: " << SplitInfo
         << " GainRatio: " << GainRatio
         << endl;
  }

  return GainRatio;

}

float Stump::calcIofP(float* P, int size){
  if (STDEBUG) cout << "calcIofP, size=" << size << endl;
  float I = 0;
  for (int i = 0; i < size; i++){
    I -= P[i] * log(P[i]);
  }
  return I;
}

/** Calculate I(P) for set. */
float Stump::calcIforSet(const std::vector<stump_experience*> &instances){
  if (STDEBUG) cout << "calcIforSet" << endl;

  std::vector<float> classes;
  std::vector<int> classCounts;

  // go through instances and figure count of each type
  for (unsigned i = 0; i < instances.size(); i++){

    float val = instances[i]->output;
    bool newValue = true;
    // see if this is a new val
    for (unsigned j = 0; j < classes.size(); j++){
      // not new, increment count for this class
      if (val == classes[j]){
        newValue = false;
        classCounts[j]++;
        break;
      }
    }

    // it is a new value
    if (newValue){
      classes.push_back(val);
      classCounts.push_back(1);
      if (STDEBUG) cout << "found new class with val " << val << endl;
    }
  }

  // now calculate P
  float *P = new float[classes.size()];
  for (int i = 0; i < (int)classCounts.size(); i++){
    P[i] = (float)classCounts[i] / (float)instances.size();
  }

  // calculate I(P)
  float I = calcIofP(P, classes.size());
  delete [] P;

  return I;

}

/** Returns a list of the attributes in this dimension sorted
    from lowest to highest. */
float* Stump::sortOnDim(int dim){
  if (STDEBUG) cout << "sortOnDim,dim = " << dim << endl;

  float* values = new float[experiences.size()];

  for (int i = 0; i < (int)experiences.size(); i++){
    //cout << "Instance " << i << endl;

    float val = experiences[i]->input[dim];
    //cout << " val: " << val << endl;

    // find where this should go
    for (int j = 0; j <= i; j++){
      //cout << " j: " << j << endl;

      // get to i, this is the spot then
      if (j==i){
        values[j] = val;
        //cout << "  At i, putting value in slot j: " << j << endl;
      }

      // if this is the spot
      else if (val < values[j]){
        //cout << "  Found slot at j: " << j << endl;

        // slide everything forward to make room
        for (int k = i; k > j; k--){
          //cout << "   k = " << k << " Sliding value from k-1 to k" << endl;
          values[k] = values[k-1];
        }

        // put value in its spot at j
        //cout << "  Putting value at slot j: " << j << endl;
        values[j] = val;

        // break
        break;
      }

    }
  }

  if (STDEBUG){
    cout << "Sorted array: " << values[0];
    for (int i = 1; i < (int)experiences.size(); i++){
      cout << ", " << values[i];
    }
    cout << endl;
  }

  return values;

}


/** Print the stump for debug purposes. */
void Stump::printStump(){

  cout << "Type: " << type
       << " Dim: " << dim << " Val: " << val
       << " nExperiences: " << nExperiences ;

  cout << " Left Outputs: ";
  for (std::multiset<float>::iterator j = leftOutputs.begin();
       j != leftOutputs.end(); j++){
    cout << *j << ", ";
  }
  cout << endl;
  cout << " Right Outputs: ";
  for (std::multiset<float>::iterator j = rightOutputs.begin();
       j != rightOutputs.end(); j++){
    cout << *j << ", ";
  }
  cout << endl;

}

/** See if the attributes are independent of the output class
    If they are all independent, we have no more inputs to split on */
/*
  std::vector<float> Stump::calcChiSquare(std::vector<experience> experiences){

  std::vector<float> chiSquare;
  chiSquare.resize(experiences[0].input.size());

  // for each input attribute
  for (int i = 0; i < (int)experiences[0].input.size(); i++){

  // attribute counts
  std::vector<float> attribValue;
  std::vector<int> attribCount;

  // outcome counts
  std::vector<float> outcomeValue;
  std::vector<int> outcomeCount;

  for (int k = 0; k < (int)experiences.size(); k++){

  experience* exp = &(experiences[k]);

  // match attribute value
  bool attribMatch = false;
  for (int l = 0; l < (int)attribValue.size(); l++){
  if (attribValue[l] == exp->input[i]){
  attribMatch = true;
  attribCount[l]++;
  break;
  }
  }
  // no match
  attribValue.push_back(exp->input[i]);
  attribCount.push_back(1);

  // match outcome value
  bool outcomeMatch = false;
  for (int l = 0; l < (int)outcomeValue.size(); l++){
  if (outcomeValue[l] == exp->output){
  outcomeMatch = true;
  outcomeCount[l]++;
  break;
  }
  }
  // no match
  outcomeValue.push_back(exp->output);
  outcomeCount.push_back(1);

  }

  // table
  // TODO

  float** actual = new float*[attribValue.size()];//[outcomeValue.size()];
  float** expected = new float*[attribValue.size()];//[outcomeValue.size()];
  for (int k = 0; k < (int)attribValue.size(); k++){
  actual[k] = new float[outcomeValue.size()];
  expected[k] = new float[outcomeValue.size()];
  }

  // calculate actual table
  for (int k = 0; k < (int)experiences.size(); k++){
  experience* exp = &(experiences[k]);

  // match
  for (int l = 0; l < (int)attribValue.size(); l++){
  if (attribValue[l] != exp->input[i])
  continue;
  for (int j = 0; j < (int)outcomeValue.size(); j++){
  if (outcomeValue[j] != exp->output)
  continue;
  actual[l][j]++;
  }
  }
  }

  // calculate expected table and compare
  float x2 = 0.0;
  for (int l = 0; l < (int)attribValue.size(); l++){
  for (int j = 0; j < (int)outcomeValue.size(); j++){

  // calculate expected
  expected[l][j] = attribCount[l] * outcomeCount[j] /
  (float)experiences.size();

  float cell = (actual[l][j] - expected[l][j]) *
  (actual[l][j] - expected[l][j]) / expected[i][j];

  x2 += cell;
  }
  }

  delete actual;
  delete expected;

  chiSquare[i] = x2;
  cout << " Feature " << i << " Independence: " << chiSquare[i] << endl;

  }
  return chiSquare;

  }
*/

// output a map of outcomes and their probabilities for this leaf node
void Stump::outputProbabilities(std::multiset<float> outputs, std::map<float, float>* retval){
  if (STDEBUG) cout << " Calculating output probs" << endl;

  // go through all possible output values
  for (std::multiset<float>::iterator it = outputs.begin();
       it != outputs.end(); it = outputs.upper_bound(*it)){

    // get count for this value
    float val = *it;
    int count = outputs.count(val);
    if (count > 0){
      (*retval)[val] = (float) count / (float)outputs.size();
      if (STDEBUG) cout << "  Output value " << val << " had count of " << count
                        << " on "
                        << outputs.size() << " experiences and prob of "
                        << (*retval)[val] << endl;
    }
  }

}
