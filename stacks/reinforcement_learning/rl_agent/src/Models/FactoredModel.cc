/** FactoredModel.cc
    Implements the FactoredModel class, which uses separate supervised learners to model each feature of an MDP.
    Please cite: Hester and Stone, "Real Time Targeted Exploration in Large Domains", ICDL 2010.
    \author Todd Hester
*/

#include "FactoredModel.hh"


FactoredModel::FactoredModel(int id, int numactions, int M, int modelType,
                 int predType, int nModels, float treeThreshold,
                 const std::vector<float> &featRange, float rRange,
                 bool needConf, bool dep, bool relTrans, float featPct, 
		 bool stoch, bool episodic, Random rng):
  rewardModel(NULL), terminalModel(NULL), 
  id(id), nact(numactions), M(M), modelType(modelType),
  predType(predType), nModels(nModels),
  treeBuildType(BUILD_ON_ERROR), // build tree after prediction error
  treeThresh(treeThreshold), featRange(featRange), rRange(rRange),
  needConf(needConf), dep(dep), relTrans(relTrans), FEAT_PCT(featPct), 
  stoch(stoch), episodic(episodic), rng(rng)
{

  //cout << "MDP Tree explore type: " << predType << endl;
  MODEL_DEBUG = false; //true;
  COPYDEBUG = false;

  // percent of experiences to use for each model
  EXP_PCT = 0.55;//6; //0.4;

  // just to ensure the diff models are on different random values
  for (int i = 0; i < id; i++){
    rng.uniform(0, 1);
  }

}


FactoredModel::FactoredModel(const FactoredModel & m):
  rewardModel(NULL), terminalModel(NULL), 
  id(m.id), nact(m.nact), M(m.M), modelType(m.modelType),
  predType(m.predType), nModels(m.nModels),
  treeBuildType(m.treeBuildType),
  treeThresh(m.treeThresh), featRange(m.featRange), rRange(m.rRange),
  needConf(m.needConf), dep(m.dep), relTrans(m.relTrans), FEAT_PCT(m.FEAT_PCT),
  stoch(m.stoch), episodic(m.episodic), rng(m.rng)
{
  COPYDEBUG = m.COPYDEBUG;

  if (COPYDEBUG) cout << "MDP Tree copy constructor" << endl;
  MODEL_DEBUG = m.MODEL_DEBUG;
  EXP_PCT = m.EXP_PCT;
  nfactors = m.nfactors;


  if (m.outputModels.size() > 0){
    if (COPYDEBUG) cout << " FactoredModel copy trees" << endl;
    rewardModel = m.rewardModel->getCopy();
    if (m.terminalModel != NULL) terminalModel = m.terminalModel->getCopy();
    if (COPYDEBUG) cout << " copy output trees" << endl;
    outputModels.resize(m.outputModels.size());
    for (unsigned i = 0; i < m.outputModels.size(); i++){
      outputModels[i] = m.outputModels[i]->getCopy();
    }
    if (COPYDEBUG) cout << " FactoredModel trees copied" << endl;
  }
  if (COPYDEBUG) cout << "FactoredModel copy complete " << endl;
}

FactoredModel* FactoredModel::getCopy(){

  FactoredModel* copy = new FactoredModel(*this);
  return copy;

}


FactoredModel::~FactoredModel() {
  if (rewardModel != NULL) delete rewardModel;
  if (terminalModel != NULL) delete terminalModel;
  for (unsigned i = 0; i < outputModels.size(); i++){
    delete outputModels[i];
  }
  outputModels.clear();
}



// init the trees
bool FactoredModel::initMDPModel(int nfactors){
  if (MODEL_DEBUG) cout << "Init trees for each state factor and reward" << endl;

  outputModels.resize(nfactors);

  bool simpleRegress = false;
  if (modelType == M5SINGLE || modelType == M5ALLSINGLE || modelType == LSTSINGLE)
    simpleRegress = true;

  // institute a model for each state factor, depending on model type
  for (int i = 0; i < nfactors; i++){
    if (modelType == C45TREE && nModels == 1){
      outputModels[i] = new C45Tree((id * (nfactors+1)) + i, treeBuildType, 5, M, 0, rng);
      if (i == 0){
        rewardModel = new C45Tree((id*(nfactors+1))+nfactors, treeBuildType,5, M, 0, rng);
        if (episodic) terminalModel = new C45Tree((id*(nfactors+1))+nfactors+1, treeBuildType,5, M, 0, rng);
      }
    }
    else if ((modelType == M5MULTI || modelType == M5SINGLE) && nModels == 1){
      outputModels[i] = new M5Tree((id * (nfactors+1)) + i, treeBuildType, 5, M, 0, simpleRegress, false, treeThresh *featRange[i], rng);
      if (i == 0){
        rewardModel = new M5Tree((id * (nfactors+1)) + nfactors, treeBuildType, 5, M, 0, simpleRegress, false, treeThresh *rRange, rng);
        if (episodic) terminalModel = new M5Tree((id * (nfactors+1)) + 1+nfactors, treeBuildType, 5, M, 0, simpleRegress, false, treeThresh, rng);
      }
    }
    else if ((modelType == M5ALLMULTI || modelType == M5ALLSINGLE) && nModels == 1){
      outputModels[i] = new M5Tree((id * (nfactors+1)) + i, treeBuildType, 5, M, 0, simpleRegress, true, treeThresh *featRange[i], rng);
      if (i == 0){
        rewardModel = new M5Tree((id * (nfactors+1)) + nfactors, treeBuildType, 5, M, 0, simpleRegress, true, treeThresh *rRange, rng);
        if (episodic) terminalModel = new M5Tree((id * (nfactors+1)) + 1+nfactors, treeBuildType, 5, M, 0, simpleRegress, true, treeThresh, rng);
      }
    }
    else if ((modelType == LSTMULTI || modelType == LSTSINGLE) && nModels == 1){
      outputModels[i] = new LinearSplitsTree((id * (nfactors+1)) + i, treeBuildType, 5, M, 0, simpleRegress, treeThresh *featRange[i], rng);
      if (i == 0){
        rewardModel = new LinearSplitsTree((id * (nfactors+1)) + nfactors, treeBuildType, 5, M, 0, simpleRegress, treeThresh *rRange, rng);
        if (episodic) terminalModel = new LinearSplitsTree((id * (nfactors+1)) + 1+nfactors, treeBuildType, 5, M, 0, simpleRegress, treeThresh, rng);
      }
    }
    else if (modelType == STUMP && nModels == 1){
      outputModels[i] = new Stump((id * (nfactors+1)) + i, 1, 5, M, 0, rng);
      if (i == 0){
        rewardModel = new Stump((id * (nfactors+1)) + nfactors, 1, 5, M, 0, rng);
        if (episodic) terminalModel = new Stump((id * (nfactors+1)) +1+ nfactors, 1, 5, M, 0, rng);
      }
    }
    else if (predType == SEPARATE && nModels > 1){
      outputModels[i] = new SepPlanExplore((id * (nfactors+1)) + i,
                                          modelType, predType,
                                          nModels, treeBuildType, 5,
                                          FEAT_PCT,
                                          EXP_PCT,
                                          treeThresh *featRange[i], stoch, rng);
      if (i == 0){
        rewardModel = new SepPlanExplore((id * (nfactors+1)) + nfactors,
                                        modelType, predType,
                                        nModels, treeBuildType, 5,
                                        FEAT_PCT, // remove this pct of feats
                                        EXP_PCT, treeThresh *rRange, stoch, rng);
	if (episodic){
	  terminalModel = new SepPlanExplore((id * (nfactors+1)) +1+ nfactors,
					    modelType, predType,
					    nModels, treeBuildType, 5,
					    FEAT_PCT, // remove this pct of feats
					    EXP_PCT, treeThresh, stoch, rng);
	}
      }
    }
    else if (nModels > 1 || modelType == ALLM5TYPES){
      outputModels[i] = new MultipleClassifiers((id * (nfactors+1)) + i,
                                               modelType, predType,
                                               nModels, treeBuildType, 5,
                                               FEAT_PCT,
                                               EXP_PCT,
                                               treeThresh *featRange[i], stoch, rng);
      if (i == 0){
        rewardModel = new MultipleClassifiers((id * (nfactors+1)) + nfactors,
                                             modelType, predType,
                                             nModels, treeBuildType, 5,
                                             FEAT_PCT, // remove this pct of feats
                                             EXP_PCT, treeThresh *rRange, stoch, rng);
	if (episodic){
	  terminalModel = new MultipleClassifiers((id * (nfactors+1)) +1+ nfactors,
						 modelType, predType,
						 nModels, treeBuildType, 5,
						 FEAT_PCT, // remove this pct of feats
						 EXP_PCT, treeThresh, stoch, rng);
	}
      }
    } else {
      cout << "Invalid model type for MDP TREE" << endl;
      exit(-1);
    }

  }

  return true;

}


// update all trees with multiple experiences
bool FactoredModel::updateWithExperiences(std::vector<experience> &instances){
  if (MODEL_DEBUG) cout << "FactoredModel updateWithExperiences : " << instances.size() << endl;

  bool changed = false;
  if (outputModels.size() == 0){
    nfactors = instances[0].next.size();
    initMDPModel(instances[0].next.size());
  }

  // make sure size is right
  if (outputModels.size() != instances[0].next.size()){
    if (MODEL_DEBUG)
      cout << "ERROR: size mismatch between input vector and # trees "
           << outputModels.size() << ", " << instances[0].next.size() << endl;
    return false;
    exit(-1);
  }

  // separate these experience instances into classPairs
  std::vector<std::vector<classPair> > stateData(outputModels.size());
  std::vector<classPair> rewardData(instances.size());
  std::vector<classPair> termData(instances.size());

  // count non-terminal experiences
  int nonTerm = 0;
  for (unsigned i = 0; i < instances.size(); i++){
    if (!instances[i].terminal)
      nonTerm++;
  }
  for (unsigned i = 0; i < outputModels.size(); i++){
    stateData[i].resize(nonTerm);
  }
  int nonTermIndex = 0;

  for (unsigned i = 0; i < instances.size(); i++){
    experience e = instances[i];

    std::vector<float> inputs(e.s.size() + nact);

    for (unsigned k = 0; k < e.s.size(); k++){
      inputs[k] = e.s[k];
    }
    // convert to binary vector of length nact
    for (int k = 0; k < nact; k++){
      if (e.act == k)
        inputs[e.s.size()+k] = 1;
      else
        inputs[e.s.size()+k] = 0;
    }

    // convert to rel
    if (relTrans)
      e.next = subVec(e.next, e.s);

    // reward and terminal models
    classPair cp;
    cp.in = inputs;
    cp.out = e.reward;
    rewardData[i] = cp;

    cp.out = e.terminal;
    termData[i] = cp;

    // add to each vector
    if (!e.terminal){
      for (unsigned j = 0; j < outputModels.size(); j++){
        classPair cp;
        cp.in = inputs;

        // split the outcome and rewards up
        // into each vector
        cp.out = e.next[j];
        stateData[j][nonTermIndex] = cp;

        // for dep trees, add this models target to next model's input
        if (dep){
          inputs.push_back(e.next[j]);
        }
      }
      nonTermIndex++;
    }

  }

  // build trees on all data
  for (unsigned k = 0; k < stateData.size(); k++){
    if (stateData[k].size() > 0){
      bool singleChange = outputModels[k]->trainInstances(stateData[k]);
      changed = changed || singleChange;
    }
  }

  bool singleChange = rewardModel->trainInstances(rewardData);
  changed = changed || singleChange;

  if (episodic){
    singleChange = terminalModel->trainInstances(termData);
    changed = changed || singleChange;
  }

  return changed;
}


// update all the trees, check if model has changed
bool FactoredModel::updateWithExperience(experience &e){
  if (MODEL_DEBUG) cout << "updateWithExperience " << &(e.s) << ", " << e.act
                       << ", " << &(e.next) << ", " << e.reward << endl;

  if (MODEL_DEBUG){
    cout << "From: ";
    for (unsigned i = 0; i < e.s.size(); i++){
      cout << e.s[i] << ", ";
    }
    cout << "Action: " << e.act << endl;;
    cout << "To: ";
    for (unsigned i = 0; i < e.next.size(); i++){
      cout << e.next[i] << ", ";
    }
    cout << "Reward: " << e.reward 
	 << " term: " << e.terminal << endl;
  }

  bool changed = false;

  if (outputModels.size() == 0){
    nfactors = e.next.size();
    initMDPModel(e.next.size());
  }

  // make sure size is right
  if (outputModels.size() != e.next.size()){
    if (MODEL_DEBUG) cout << "ERROR: size mismatch between input vector and # trees "
                         << outputModels.size() << ", " << e.next.size() << endl;
    return false;
    exit(-1);
  }

  std::vector<float> inputs(e.s.size() + nact);
  for (unsigned i = 0; i < e.s.size(); i++){
    inputs[i] = e.s[i];
  }
  // convert to binary vector of length nact
  for (int k = 0; k < nact; k++){
    if (e.act == k)
      inputs[e.s.size()+k] = 1;
    else
      inputs[e.s.size()+k] = 0;
  }

  // convert to rel
  if (relTrans)
    e.next = subVec(e.next, e.s);

  // split the outcome and rewards up
  // and train the trees
  classPair cp;
  cp.in = inputs;

  // reward model
  cp.out = e.reward;
  bool singleChange = rewardModel->trainInstance(cp);
  changed = changed || singleChange;

  // termination model
  if (episodic){
    cp.out = e.terminal;
    singleChange = terminalModel->trainInstance(cp);
    changed = changed || singleChange;
  }

  // if not a terminal transition
  if (!e.terminal){
    for (unsigned i = 0; i < e.next.size(); i++){
      cp.in = inputs;
      cp.out = e.next[i];

      bool singleChange = outputModels[i]->trainInstance(cp);
      changed = changed || singleChange;

      // add this model's target to input for next model
      if (dep){
        inputs.push_back(e.next[i]);
      }
    }
  }

  if (MODEL_DEBUG) cout << "Model updated, changed: " << changed << endl;
  return changed;

}


bool FactoredModel::getSingleSAInfo(const std::vector<float> &state, int act, StateActionInfo* retval){

  retval->transitionProbs.clear();

  if (outputModels.size() == 0){
    retval->reward = -0.001;

    // add to transition map
    retval->transitionProbs[state] = 1.0;
    retval->known = false;
    retval->conf = 0.0;
    retval->termProb = 0.0;
    return false;
  }

  // input we want predictions for
  std::vector<float> inputs(state.size() + nact);
  for (unsigned i = 0; i < state.size(); i++){
    inputs[i] = state[i];
  }
  // convert to binary vector of length nact
  for (int k = 0; k < nact; k++){
    if (act == k)
      inputs[state.size()+k] = 1;
    else
      inputs[state.size()+k] = 0;
  }

  // just pick one sample from each feature prediction
  std::vector<float>output(nfactors);
  for (int i = 0; i < nfactors; i++){

    // get prediction
    std::map<float, float> outputPreds;
    outputModels[i]->testInstance(inputs, &outputPreds);

    // sample a value
    float randProb = rng.uniform();
    float probSum = 0;
    for (std::map<float, float>::iterator it1 = outputPreds.begin(); it1 != outputPreds.end(); it1++){

      // get prob
      probSum += (*it1).second;

      if (randProb <= probSum){
	output[i] = (*it1).first;
	break;
      }
    }
  }

  retval->transitionProbs[output] = 1.0;

  // calculate reward and terminal probabilities
  // calculate expected reward
  float rewardSum = 0.0;
  // each value
  std::map<float, float> rewardPreds;
  rewardModel->testInstance(inputs, &rewardPreds);

  float totalVisits = 0.0;
  for (std::map<float, float>::iterator it = rewardPreds.begin(); it != rewardPreds.end(); it++){
    // get key from iterator
    float val = (*it).first;
    float prob = (*it).second;
    totalVisits += prob;
    if (MODEL_DEBUG) cout << "Reward value " << val << " had prob of " << prob << endl;
    rewardSum += (prob * val);
  }

  retval->reward = rewardSum / totalVisits;
  if (MODEL_DEBUG) cout << "Average reward was " << retval->reward << endl;

  if (isnan(retval->reward))
    cout << "FactoredModel setting model reward to NaN" << endl;


  // get termination prob
  std::map<float, float> termProbs;
  if (!episodic){
    termProbs[0.0] = 1.0;
  } else {
    terminalModel->testInstance(inputs, &termProbs);
  }
  // this needs to be a weighted sum.
  // discrete trees will give some probabilty of termination (outcome 1)
  // where continuous ones will give some value between 0 and 1
  float termSum = 0;
  float probSum = 0;
  for (std::map<float, float>::iterator it = termProbs.begin(); it != termProbs.end(); it++){
    // get key from iterator
    float val = (*it).first;
    if (val > 1.0) val = 1.0;
    if (val < 0.0) val = 0.0;
    float prob = (*it).second;
    if (MODEL_DEBUG) cout << "Term value " << val << " had prob of " << prob << endl;
    termSum += (prob * val);
    probSum += prob;
  }

  retval->termProb = termSum / probSum;
  if (retval->termProb < 0 || retval->termProb > 1){
    cout << "Invalid termination probability!!! " << retval->termProb << endl;
  }
  if (MODEL_DEBUG) cout << "Termination prob is " << retval->termProb << endl;

  return retval;

}


// fill in StateActionInfo struct and return it
bool FactoredModel::getStateActionInfo(const std::vector<float> &state, int act, StateActionInfo* retval){
  if (MODEL_DEBUG) cout << "getStateActionInfo, " << &state <<  ", " << act << endl;



  if (MODEL_DEBUG){
    for (unsigned i = 0; i < state.size(); i++){
      cout << state[i] << ", ";
    }
    //    cout << endl;
    cout << "a: " << act << " has " << retval->transitionProbs.size() << " outcomes already" << endl;
  }

  retval->transitionProbs.clear();

  if (outputModels.size() == 0){
    retval->reward = -0.001;

    // add to transition map
    retval->transitionProbs[state] = 1.0;
    retval->known = false;
    retval->conf = 0.0;
    retval->termProb = 0.0;
    return false;
  }

  // input we want predictions for
  std::vector<float> inputs(state.size() + nact);
  for (unsigned i = 0; i < state.size(); i++){
    inputs[i] = state[i];
  }
  // convert to binary vector of length nact
  for (int k = 0; k < nact; k++){
    if (act == k)
      inputs[state.size()+k] = 1;
    else
      inputs[state.size()+k] = 0;
  }


  // get the separate predictions for each outcome variable from the respective trees
  // combine together for outcome predictions

  // combine together and put into StateActionInfo struct
  retval->known = true;
  float confSum = 0.0;

  if (nModels == 1 &&
      (modelType == M5MULTI || modelType == M5SINGLE ||
       modelType == M5ALLMULTI || modelType == M5ALLSINGLE ||
       modelType == LSTMULTI || modelType == LSTSINGLE ||
       modelType == ALLM5TYPES)){
    //cout << "mdptree, combine deterministic outputs for each feature" << endl;
    ///////////////////////////////////////////
    // alternate version -> assuming one model that gives one prediction
    ///////////////////////////////////////////
    std::vector<float> MLnext(nfactors);
    std::vector<float> inputCopy = inputs;
    for (int i = 0; i < nfactors; i++){
      // get single outcome for this factor
      std::map<float, float> outputPreds;
      outputModels[i]->testInstance(inputCopy, &outputPreds);
      if (needConf && dep) confSum += outputModels[i]->getConf(inputCopy);
      float val = outputPreds.begin()->first;
      if (relTrans) val = val + inputs[i];
      MLnext[i] = val;
      if (dep){
        inputCopy.push_back(val);
      }
    }
    //add this one
    retval->transitionProbs[MLnext] = 1.0;
    if (MODEL_DEBUG){
      cout << "Final prob of outcome: ";
      for (int i = 0; i < nfactors; i++){
        cout << MLnext[i] << ", ";
      }
      cout << " is " << 1.0 << endl;
    }
    ////////////////////////////////////////////
  }

  else {
    //cout << "mdp tree, combine stochastic predictions for each feature" << endl;
    //////////////////////////////////////////////////////////////////////
    // Full version: assume possibly stochastic prediction for each model
    //////////////////////////////////////////////////////////////////////
    // grab predicted transition probs just once
    std::vector< std::map<float,float> > predictions(nfactors);
    if (!dep){
      for (int i = 0; i < nfactors; i++){
        outputModels[i]->testInstance(inputs, &(predictions[i]));
      }
    }

    ///////////////////////////////////////
    // get probability of each transition
    float* probs = new float[nfactors];
    std::vector<float> next(nfactors, 0);
    addFactorProb(probs, &next, inputs, retval, 0, predictions, &confSum);
    delete[] probs;

    /////////////////////////////////////////////////
  }

  // calculate expected reward
  float rewardSum = 0.0;
  // each value
  std::map<float, float> rewardPreds;
  rewardModel->testInstance(inputs, &rewardPreds);

  if (rewardPreds.size() == 0){
    //cout << "FactoredModel setting state known false" << endl;
    retval->known = false;
    return retval;
  }

  float totalVisits = 0.0;
  for (std::map<float, float>::iterator it = rewardPreds.begin(); it != rewardPreds.end(); it++){
    // get key from iterator
    float val = (*it).first;
    float prob = (*it).second;
    totalVisits += prob;
    if (MODEL_DEBUG) cout << "Reward value " << val << " had prob of " << prob << endl;
    rewardSum += (prob * val);

  }

  retval->reward = rewardSum / totalVisits;
  if (MODEL_DEBUG) cout << "Average reward was " << retval->reward << endl;

  if (isnan(retval->reward))
    cout << "FactoredModel setting model reward to NaN" << endl;


  // get termination prob
  std::map<float, float> termProbs;
  if (!episodic){
    termProbs[0.0] = 1.0;
  } else {
    terminalModel->testInstance(inputs, &termProbs);
  }
  // this needs to be a weighted sum.
  // discrete trees will give some probabilty of termination (outcome 1)
  // where continuous ones will give some value between 0 and 1
  float termSum = 0;
  float probSum = 0;
  for (std::map<float, float>::iterator it = termProbs.begin(); it != termProbs.end(); it++){
    // get key from iterator
    float val = (*it).first;
    if (val > 1.0) val = 1.0;
    if (val < 0.0) val = 0.0;
    float prob = (*it).second;
    if (MODEL_DEBUG) cout << "Term value " << val << " had prob of " << prob << endl;
    termSum += (prob * val);
    probSum += prob;
  }

  retval->termProb = termSum / probSum;
  if (retval->termProb < 0 || retval->termProb > 1){
    cout << "Invalid termination probability!!! " << retval->termProb << endl;
  }
  if (MODEL_DEBUG) cout << "Termination prob is " << retval->termProb << endl;

  // if we need confidence measure
  if (needConf){
    // conf is avg of each variable's model's confidence
    retval->conf = confSum;
    float rConf = rewardModel->getConf(inputs);
    float tConf = 1.0;
    if (episodic)
      tConf = terminalModel->getConf(inputs);

    //cout << "conf is " << confSum << ", r: " << rConf << ", " << tConf << endl;

    retval->conf += rConf + tConf;

    if (!dep){
      for (int i = 0; i < nfactors; i++){
        float featConf = outputModels[i]->getConf(inputs);
        retval->conf += featConf;
        //cout << "indep, conf for " << i << ": " << featConf << endl;
      }
    }
    retval->conf /= (float)(state.size() + 2.0);
  } else {
    retval->conf = 1.0;
  }

  if (MODEL_DEBUG) cout << "avg conf returned " << retval->conf << endl;

  //cout << "now has " << retval->transitionProbs.size() << " outcomes" << endl;

  // return filled-in struct
  retval->known = true;
  return true;

}



// gets the values/probs for index and adds them to the appropriate spot in the array
void FactoredModel::addFactorProb(float* probs, std::vector<float>* next, std::vector<float> x, StateActionInfo* retval, int index, std::vector< std::map<float,float> > predictions, float* confSum){

  // get values, probs etc for this index
  std::map<float, float> outputPreds = predictions[index];

  // get prediction each time for dep
  if (dep){
    outputModels[index]->testInstance(x, &outputPreds);
  }

  // sum up confidences
  if (dep && needConf){
    float conf = outputModels[index]->getConf(x);
    if (index > 0)
      (*confSum) += conf * probs[index-1];
    else
      (*confSum) += conf;
  }

  for (std::map<float, float>::iterator it1 = outputPreds.begin(); it1 != outputPreds.end(); it1++){
    // get key from iterator
    float val = (*it1).first;

    if (MODEL_DEBUG) cout << "Prob of outcome " << val << " on factor " << index << " is " << (*it1).second << endl;

    // ignore it if it has prob 0
    if ((*it1).second == 0){
      if (MODEL_DEBUG) cout << "Prob 0, ignore" << endl;
      continue;
    }

    if (dep){
      x.push_back(val);
    }

    if (relTrans)
      val = val + x[index];

    (*next)[index] = val;
    if (index == 0)
      probs[index] = (*it1).second;
    else
      probs[index] = probs[index-1] * (*it1).second;

    // if last one, lets set it in our transition prob map
    if (index == nfactors - 1 && probs[index] > 0.0){

      if (MODEL_DEBUG){
        cout << "Final prob of outcome: ";
        for (int i = 0; i < nfactors; i++){
          cout << (*next)[i] << ", ";
        }
        cout << " is " << probs[index] << endl;
        cout << " was " << retval->transitionProbs[*next] << endl;
        cout << " now " << (retval->transitionProbs[*next]+probs[index]) << endl;
      }

      retval->transitionProbs[*next] += probs[index];
      continue;
    }

    // next factors
    addFactorProb(probs, next, x, retval, index+1, predictions, confSum);

  }
}

std::vector<float> FactoredModel::addVec(const std::vector<float> &a, const std::vector<float> &b){
  //if (a.size() != b.size())
  // cout << "ERROR: vector sizes wrong" << endl;


  int smaller = a.size();
  if (b.size() < a.size())
    smaller = b.size();

  std::vector<float> c(smaller, 0.0);
  for (int i = 0; i < smaller; i++){
    c[i] = a[i] + b[i];
  }

  return c;
}

std::vector<float> FactoredModel::subVec(const std::vector<float> &a, const std::vector<float> &b){
  //if (a.size() != b.size())
  // cout << "ERROR: vector sizes wrong" << endl;

  int smaller = a.size();
  if (b.size() < a.size())
    smaller = b.size();

  std::vector<float> c(smaller, 0.0);
  for (int i = 0; i < smaller; i++){
    c[i] = a[i] - b[i];
  }

  return c;
}

