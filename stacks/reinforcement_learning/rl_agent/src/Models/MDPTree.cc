/** \file MDPTree.cc
    Implements the MDPTree class, which uses decision trees to model each feature of an MDP.
    Please cite: Hester and Stone, "Real Time Targeted Exploration in Large Domains", ICDL 2010.
    \author Todd Hester
*/

#include "MDPTree.hh"


MDPTree::MDPTree(int id, int numactions, int M, int modelType,
                 int predType, int nModels, float treeThreshold,
                 const std::vector<float> &featRange, float rRange,
                 bool needConf, bool dep, bool relTrans, float featPct, 
		 bool stoch, bool episodic, Random rng):
  rewardTree(NULL), terminalTree(NULL), 
  id(id), nact(numactions), M(M), modelType(modelType),
  predType(predType), nModels(nModels),
  treeBuildType(BUILD_ON_ERROR), // build tree after prediction error
  treeThresh(treeThreshold), featRange(featRange), rRange(rRange),
  needConf(needConf), dep(dep), relTrans(relTrans), FEAT_PCT(featPct), 
  stoch(stoch), episodic(episodic), rng(rng)
{

  //cout << "MDP Tree explore type: " << predType << endl;
  TREE_DEBUG = false; //true;
  COPYDEBUG = false;

  // percent of experiences to use for each model
  EXP_PCT = 0.55;//6; //0.4;

  // just to ensure the diff models are on different random values
  for (int i = 0; i < id; i++){
    rng.uniform(0, 1);
  }

}


MDPTree::MDPTree(const MDPTree & m):
  rewardTree(NULL), terminalTree(NULL), 
  id(m.id), nact(m.nact), M(m.M), modelType(m.modelType),
  predType(m.predType), nModels(m.nModels),
  treeBuildType(m.treeBuildType),
  treeThresh(m.treeThresh), featRange(m.featRange), rRange(m.rRange),
  needConf(m.needConf), dep(m.dep), relTrans(m.relTrans), FEAT_PCT(m.FEAT_PCT),
  stoch(m.stoch), episodic(m.episodic), rng(m.rng)
{
  COPYDEBUG = m.COPYDEBUG;

  if (COPYDEBUG) cout << "MDP Tree copy constructor" << endl;
  TREE_DEBUG = m.TREE_DEBUG;
  EXP_PCT = m.EXP_PCT;
  nfactors = m.nfactors;


  if (m.outputTrees.size() > 0){
    if (COPYDEBUG) cout << " MDPTree copy trees" << endl;
    rewardTree = m.rewardTree->getCopy();
    if (m.terminalTree != NULL) terminalTree = m.terminalTree->getCopy();
    if (COPYDEBUG) cout << " copy output trees" << endl;
    outputTrees.resize(m.outputTrees.size());
    for (unsigned i = 0; i < m.outputTrees.size(); i++){
      outputTrees[i] = m.outputTrees[i]->getCopy();
    }
    if (COPYDEBUG) cout << " MDPTree trees copied" << endl;
  }
  if (COPYDEBUG) cout << "MDPTree copy complete " << endl;
}

MDPTree* MDPTree::getCopy(){

  MDPTree* copy = new MDPTree(*this);
  return copy;

}


MDPTree::~MDPTree() {
  if (rewardTree != NULL) delete rewardTree;
  if (terminalTree != NULL) delete terminalTree;
  for (unsigned i = 0; i < outputTrees.size(); i++){
    delete outputTrees[i];
  }
  outputTrees.clear();
}



// init the trees
bool MDPTree::initMDPModel(int nfactors){
  if (TREE_DEBUG) cout << "Init trees for each state factor and reward" << endl;

  outputTrees.resize(nfactors);

  bool simpleRegress = false;
  if (modelType == M5SINGLE || modelType == M5ALLSINGLE || modelType == LSTSINGLE)
    simpleRegress = true;

  // institute a model for each state factor, depending on model type
  for (int i = 0; i < nfactors; i++){
    if (modelType == C45TREE && nModels == 1){
      outputTrees[i] = new C45Tree((id * (nfactors+1)) + i, treeBuildType, 5, M, 0, rng);
      if (i == 0){
        rewardTree = new C45Tree((id*(nfactors+1))+nfactors, treeBuildType,5, M, 0, rng);
        if (episodic) terminalTree = new C45Tree((id*(nfactors+1))+nfactors+1, treeBuildType,5, M, 0, rng);
      }
    }
    else if ((modelType == M5MULTI || modelType == M5SINGLE) && nModels == 1){
      outputTrees[i] = new M5Tree((id * (nfactors+1)) + i, treeBuildType, 5, M, 0, simpleRegress, false, treeThresh *featRange[i], rng);
      if (i == 0){
        rewardTree = new M5Tree((id * (nfactors+1)) + nfactors, treeBuildType, 5, M, 0, simpleRegress, false, treeThresh *rRange, rng);
        if (episodic) terminalTree = new M5Tree((id * (nfactors+1)) + 1+nfactors, treeBuildType, 5, M, 0, simpleRegress, false, treeThresh, rng);
      }
    }
    else if ((modelType == M5ALLMULTI || modelType == M5ALLSINGLE) && nModels == 1){
      outputTrees[i] = new M5Tree((id * (nfactors+1)) + i, treeBuildType, 5, M, 0, simpleRegress, true, treeThresh *featRange[i], rng);
      if (i == 0){
        rewardTree = new M5Tree((id * (nfactors+1)) + nfactors, treeBuildType, 5, M, 0, simpleRegress, true, treeThresh *rRange, rng);
        if (episodic) terminalTree = new M5Tree((id * (nfactors+1)) + 1+nfactors, treeBuildType, 5, M, 0, simpleRegress, true, treeThresh, rng);
      }
    }
    else if ((modelType == LSTMULTI || modelType == LSTSINGLE) && nModels == 1){
      outputTrees[i] = new LinearSplitsTree((id * (nfactors+1)) + i, treeBuildType, 5, M, 0, simpleRegress, treeThresh *featRange[i], rng);
      if (i == 0){
        rewardTree = new LinearSplitsTree((id * (nfactors+1)) + nfactors, treeBuildType, 5, M, 0, simpleRegress, treeThresh *rRange, rng);
        if (episodic) terminalTree = new LinearSplitsTree((id * (nfactors+1)) + 1+nfactors, treeBuildType, 5, M, 0, simpleRegress, treeThresh, rng);
      }
    }
    else if (modelType == STUMP && nModels == 1){
      outputTrees[i] = new Stump((id * (nfactors+1)) + i, 1, 5, M, 0, rng);
      if (i == 0){
        rewardTree = new Stump((id * (nfactors+1)) + nfactors, 1, 5, M, 0, rng);
        if (episodic) terminalTree = new Stump((id * (nfactors+1)) +1+ nfactors, 1, 5, M, 0, rng);
      }
    }
    else if (predType == SEPARATE && nModels > 1){
      outputTrees[i] = new SepPlanExplore((id * (nfactors+1)) + i,
                                          modelType, predType,
                                          nModels, treeBuildType, 5,
                                          FEAT_PCT,
                                          EXP_PCT,
                                          treeThresh *featRange[i], stoch, rng);
      if (i == 0){
        rewardTree = new SepPlanExplore((id * (nfactors+1)) + nfactors,
                                        modelType, predType,
                                        nModels, treeBuildType, 5,
                                        FEAT_PCT, // remove this pct of feats
                                        EXP_PCT, treeThresh *rRange, stoch, rng);
	if (episodic){
	  terminalTree = new SepPlanExplore((id * (nfactors+1)) +1+ nfactors,
					    modelType, predType,
					    nModels, treeBuildType, 5,
					    FEAT_PCT, // remove this pct of feats
					    EXP_PCT, treeThresh, stoch, rng);
	}
      }
    }
    else if (nModels > 1 || modelType == ALLM5TYPES){
      outputTrees[i] = new MultipleClassifiers((id * (nfactors+1)) + i,
                                               modelType, predType,
                                               nModels, treeBuildType, 5,
                                               FEAT_PCT,
                                               EXP_PCT,
                                               treeThresh *featRange[i], stoch, rng);
      if (i == 0){
        rewardTree = new MultipleClassifiers((id * (nfactors+1)) + nfactors,
                                             modelType, predType,
                                             nModels, treeBuildType, 5,
                                             FEAT_PCT, // remove this pct of feats
                                             EXP_PCT, treeThresh *rRange, stoch, rng);
	if (episodic){
	  terminalTree = new MultipleClassifiers((id * (nfactors+1)) +1+ nfactors,
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
bool MDPTree::updateWithExperiences(std::vector<experience> &instances){
  if (TREE_DEBUG) cout << "MDPTree updateWithExperiences : " << instances.size() << endl;

  bool changed = false;
  if (outputTrees.size() == 0){
    nfactors = instances[0].next.size();
    initMDPModel(instances[0].next.size());
  }

  // make sure size is right
  if (outputTrees.size() != instances[0].next.size()){
    if (TREE_DEBUG)
      cout << "ERROR: size mismatch between input vector and # trees "
           << outputTrees.size() << ", " << instances[0].next.size() << endl;
    return false;
    exit(-1);
  }

  // separate these experience instances into classPairs
  std::vector<std::vector<classPair> > stateData(outputTrees.size());
  std::vector<classPair> rewardData(instances.size());
  std::vector<classPair> termData(instances.size());

  // count non-terminal experiences
  int nonTerm = 0;
  for (unsigned i = 0; i < instances.size(); i++){
    if (!instances[i].terminal)
      nonTerm++;
  }
  for (unsigned i = 0; i < outputTrees.size(); i++){
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
      for (unsigned j = 0; j < outputTrees.size(); j++){
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
      bool singleChange = outputTrees[k]->trainInstances(stateData[k]);
      changed = changed || singleChange;
    }
  }

  bool singleChange = rewardTree->trainInstances(rewardData);
  changed = changed || singleChange;

  if (episodic){
    singleChange = terminalTree->trainInstances(termData);
    changed = changed || singleChange;
  }

  return changed;
}


// update all the trees, check if model has changed
bool MDPTree::updateWithExperience(experience &e){
  if (TREE_DEBUG) cout << "updateWithExperience " << &(e.s) << ", " << e.act
                       << ", " << &(e.next) << ", " << e.reward << endl;

  if (TREE_DEBUG){
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

  if (outputTrees.size() == 0){
    nfactors = e.next.size();
    initMDPModel(e.next.size());
  }

  // make sure size is right
  if (outputTrees.size() != e.next.size()){
    if (TREE_DEBUG) cout << "ERROR: size mismatch between input vector and # trees "
                         << outputTrees.size() << ", " << e.next.size() << endl;
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
  bool singleChange = rewardTree->trainInstance(cp);
  changed = changed || singleChange;

  // termination model
  if (episodic){
    cp.out = e.terminal;
    singleChange = terminalTree->trainInstance(cp);
    changed = changed || singleChange;
  }

  // if not a terminal transition
  if (!e.terminal){
    for (unsigned i = 0; i < e.next.size(); i++){
      cp.in = inputs;
      cp.out = e.next[i];

      bool singleChange = outputTrees[i]->trainInstance(cp);
      changed = changed || singleChange;

      // add this model's target to input for next model
      if (dep){
        inputs.push_back(e.next[i]);
      }
    }
  }

  if (TREE_DEBUG) cout << "Model updated, changed: " << changed << endl;
  return changed;

}


bool MDPTree::getSingleSAInfo(const std::vector<float> &state, int act, StateActionInfo* retval){

  retval->transitionProbs.clear();

  if (outputTrees.size() == 0){
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
    outputTrees[i]->testInstance(inputs, &outputPreds);

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
  rewardTree->testInstance(inputs, &rewardPreds);

  float totalVisits = 0.0;
  for (std::map<float, float>::iterator it = rewardPreds.begin(); it != rewardPreds.end(); it++){
    // get key from iterator
    float val = (*it).first;
    float prob = (*it).second;
    totalVisits += prob;
    if (TREE_DEBUG) cout << "Reward value " << val << " had prob of " << prob << endl;
    rewardSum += (prob * val);
  }

  retval->reward = rewardSum / totalVisits;
  if (TREE_DEBUG) cout << "Average reward was " << retval->reward << endl;

  if (isnan(retval->reward))
    cout << "MDPTree setting model reward to NaN" << endl;


  // get termination prob
  std::map<float, float> termProbs;
  if (!episodic){
    termProbs[0.0] = 1.0;
  } else {
    terminalTree->testInstance(inputs, &termProbs);
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
    if (TREE_DEBUG) cout << "Term value " << val << " had prob of " << prob << endl;
    termSum += (prob * val);
    probSum += prob;
  }

  retval->termProb = termSum / probSum;
  if (retval->termProb < 0 || retval->termProb > 1){
    cout << "Invalid termination probability!!! " << retval->termProb << endl;
  }
  if (TREE_DEBUG) cout << "Termination prob is " << retval->termProb << endl;

  return retval;

}


// fill in StateActionInfo struct and return it
bool MDPTree::getStateActionInfo(const std::vector<float> &state, int act, StateActionInfo* retval){
  if (TREE_DEBUG) cout << "getStateActionInfo, " << &state <<  ", " << act << endl;



  if (TREE_DEBUG){
    for (unsigned i = 0; i < state.size(); i++){
      cout << state[i] << ", ";
    }
    //    cout << endl;
    cout << "a: " << act << " has " << retval->transitionProbs.size() << " outcomes already" << endl;
  }

  retval->transitionProbs.clear();

  if (outputTrees.size() == 0){
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
      outputTrees[i]->testInstance(inputCopy, &outputPreds);
      if (needConf && dep) confSum += outputTrees[i]->getConf(inputCopy);
      float val = outputPreds.begin()->first;
      if (relTrans) val = val + inputs[i];
      MLnext[i] = val;
      if (dep){
        inputCopy.push_back(val);
      }
    }
    //add this one
    retval->transitionProbs[MLnext] = 1.0;
    if (TREE_DEBUG){
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
        outputTrees[i]->testInstance(inputs, &(predictions[i]));
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
  rewardTree->testInstance(inputs, &rewardPreds);

  if (rewardPreds.size() == 0){
    //cout << "MDPTree setting state known false" << endl;
    retval->known = false;
    return retval;
  }

  float totalVisits = 0.0;
  for (std::map<float, float>::iterator it = rewardPreds.begin(); it != rewardPreds.end(); it++){
    // get key from iterator
    float val = (*it).first;
    float prob = (*it).second;
    totalVisits += prob;
    if (TREE_DEBUG) cout << "Reward value " << val << " had prob of " << prob << endl;
    rewardSum += (prob * val);

  }

  retval->reward = rewardSum / totalVisits;
  if (TREE_DEBUG) cout << "Average reward was " << retval->reward << endl;

  if (isnan(retval->reward))
    cout << "MDPTree setting model reward to NaN" << endl;


  // get termination prob
  std::map<float, float> termProbs;
  if (!episodic){
    termProbs[0.0] = 1.0;
  } else {
    terminalTree->testInstance(inputs, &termProbs);
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
    if (TREE_DEBUG) cout << "Term value " << val << " had prob of " << prob << endl;
    termSum += (prob * val);
    probSum += prob;
  }

  retval->termProb = termSum / probSum;
  if (retval->termProb < 0 || retval->termProb > 1){
    cout << "Invalid termination probability!!! " << retval->termProb << endl;
  }
  if (TREE_DEBUG) cout << "Termination prob is " << retval->termProb << endl;

  // if we need confidence measure
  if (needConf){
    // conf is avg of each variable's model's confidence
    retval->conf = confSum;
    float rConf = rewardTree->getConf(inputs);
    float tConf = 1.0;
    if (episodic)
      tConf = terminalTree->getConf(inputs);

    //cout << "conf is " << confSum << ", r: " << rConf << ", " << tConf << endl;

    retval->conf += rConf + tConf;

    if (!dep){
      for (int i = 0; i < nfactors; i++){
        float featConf = outputTrees[i]->getConf(inputs);
        retval->conf += featConf;
        //cout << "indep, conf for " << i << ": " << featConf << endl;
      }
    }
    retval->conf /= (float)(state.size() + 2.0);
  } else {
    retval->conf = 1.0;
  }

  if (TREE_DEBUG) cout << "avg conf returned " << retval->conf << endl;

  //cout << "now has " << retval->transitionProbs.size() << " outcomes" << endl;

  // return filled-in struct
  retval->known = true;
  return true;

}



// gets the values/probs for index and adds them to the appropriate spot in the array
void MDPTree::addFactorProb(float* probs, std::vector<float>* next, std::vector<float> x, StateActionInfo* retval, int index, std::vector< std::map<float,float> > predictions, float* confSum){

  // get values, probs etc for this index
  std::map<float, float> outputPreds = predictions[index];

  // get prediction each time for dep
  if (dep){
    outputTrees[index]->testInstance(x, &outputPreds);
  }

  // sum up confidences
  if (dep && needConf){
    float conf = outputTrees[index]->getConf(x);
    if (index > 0)
      (*confSum) += conf * probs[index-1];
    else
      (*confSum) += conf;
  }

  for (std::map<float, float>::iterator it1 = outputPreds.begin(); it1 != outputPreds.end(); it1++){
    // get key from iterator
    float val = (*it1).first;

    if (TREE_DEBUG) cout << "Prob of outcome " << val << " on factor " << index << " is " << (*it1).second << endl;

    // ignore it if it has prob 0
    if ((*it1).second == 0){
      if (TREE_DEBUG) cout << "Prob 0, ignore" << endl;
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

      if (TREE_DEBUG){
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

std::vector<float> MDPTree::addVec(const std::vector<float> &a, const std::vector<float> &b){
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

std::vector<float> MDPTree::subVec(const std::vector<float> &a, const std::vector<float> &b){
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

