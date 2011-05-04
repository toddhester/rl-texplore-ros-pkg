/** \file M5Tree.cc
    Implements the M5 Decision tree, as described in:
    "Learning with Continuous Classes" by J.R. Quinlan
    "Inducing Model Trees for Continuous Classes" by Y. Wang and I.H. Witten
    \author Todd Hester
*/

#include "M5Tree.hh"


// Include stuff for newmat matrix libraries

#define WANT_MATH                    // include.h will get math fns
                                     // newmatap.h will get include.h
#include "../newmat/newmatap.h"      // need matrix applications
#ifdef use_namespace
using namespace NEWMAT;              // access NEWMAT namespace
#endif



M5Tree::M5Tree(int id, int trainMode, int trainFreq, int m,
               float featPct, bool simple, bool allowAllFeats, 
	       float min_sdr, Random rng):
  id(id), mode(trainMode), freq(trainFreq), M(m),
  featPct(featPct), SIMPLE(simple), ALLOW_ALL_FEATS(allowAllFeats),
  MIN_SDR(min_sdr), rng(rng)
{

  nnodes = 0;
  nOutput = 0;
  nExperiences = 0;
  hadError = false;
  totalnodes = 0;
  maxnodes = N_M5_NODES;


  // how close a split has to be to be randomly selected
  SPLIT_MARGIN = 0.0; //0.02; //5; //01; //0.05; //0.2; //0.05;

  LMDEBUG = false;
  DTDEBUG = false;///true;
  SPLITDEBUG = false;//true;
  STOCH_DEBUG = false; //true; //false; //true;
  INCDEBUG = false; //true; //false; //true;
  NODEDEBUG = false;
  COPYDEBUG = false; //true;
  nfeat = 4;

  cout << "Created m5 decision tree " << id;
  if (SIMPLE) cout << " simple regression";
  else cout << " multivariate regression";
  if (ALLOW_ALL_FEATS) cout << " (all feats)";
  else cout << " (subtree feats)";
  if (DTDEBUG){
    cout << " mode: " << mode << " freq: " << freq << endl;
  }
  cout << " MIN_SDR: " << MIN_SDR << endl;

  initNodes();
  initTree();


}

M5Tree::M5Tree(const M5Tree& m5):
  id(m5.id), mode(m5.mode), freq(m5.freq), M(m5.M),
  featPct(m5.featPct), SIMPLE(m5.SIMPLE), ALLOW_ALL_FEATS(m5.ALLOW_ALL_FEATS),
  MIN_SDR(m5.MIN_SDR), rng(m5.rng)
{
  COPYDEBUG = m5.COPYDEBUG;
  if (COPYDEBUG) cout << "m5 copy " << id << endl;
  nnodes = 0;
  nOutput = m5.nOutput;
  nExperiences = m5.nExperiences;
  hadError = m5.hadError;
  totalnodes = 0;
  maxnodes = m5.maxnodes;
  SPLIT_MARGIN = m5.SPLIT_MARGIN; 
  LMDEBUG = m5.LMDEBUG;
  DTDEBUG = m5.DTDEBUG;
  SPLITDEBUG = m5.SPLITDEBUG;
  STOCH_DEBUG = m5.STOCH_DEBUG; 
  INCDEBUG = m5.INCDEBUG; 
  NODEDEBUG = m5.NODEDEBUG;
  nfeat = m5.nfeat;

  if (COPYDEBUG) cout << "   M5 copy nodes, experiences, root, etc" << endl;
  // copy all experiences
  for (int i = 0; i < N_M5_EXP; i++){
    allExp[i] = m5.allExp[i];
  }
  if (COPYDEBUG) cout << "   M5 copied exp array" << endl;

  // set experience pointers
  experiences.resize(m5.experiences.size());
  for (unsigned i = 0; i < m5.experiences.size(); i++){
    experiences[i] = &(allExp[i]);
  }
  if (COPYDEBUG) cout << "   M5 set experience pointers" << endl;

  // now the tricky part, set the pointers inside the tree nodes correctly
  initNodes();

  if (COPYDEBUG) cout << "   M5 copy tree " << endl;
  root = allocateNode();
  lastNode = root;
  copyTree(root, m5.root);
  if (COPYDEBUG) cout << "   M5 tree copy done" << endl;
   
  if (COPYDEBUG) {
    cout << endl << "New tree: " << endl;
    printTree(root, 0);
    cout << endl;
    cout << "  m5 copy done" << endl;
  }

}

void M5Tree::copyTree(tree_node* newNode, tree_node* origNode){

  int nodeId = newNode->id;

  if (COPYDEBUG) {
    cout << "    Copy node " << newNode->id << " from node " << origNode->id << endl;
    cout << "    NewAddy " << newNode << ", old: " << origNode << endl;
  }

  // copy node from t
  *newNode = *origNode;
  newNode->id = nodeId;

  // if it has children, allocate and copy them too
  if (origNode->l != NULL && !newNode->leaf){
    newNode->l = allocateNode();
    if (COPYDEBUG) cout << "     Copy left node " << newNode->l->id << " from " << origNode->l->id << endl;
    copyTree(newNode->l, origNode->l);
  } else {
    newNode->l = NULL;
  }

  if (origNode->r != NULL && !newNode->leaf){
    newNode->r = allocateNode();
    if (COPYDEBUG) cout << "     Copy right node " << newNode->r->id << " from " << origNode->r->id << endl;
    copyTree(newNode->r, origNode->r);
  } else {
    newNode->r = NULL;
  }
}

M5Tree* M5Tree::getCopy(){
  M5Tree* copy = new M5Tree(*this);
  return copy;
}

M5Tree::~M5Tree() {
  deleteTree(root);
  for (unsigned i = N_M5_EXP; i < experiences.size(); i++){
    delete experiences[i];
  }
  experiences.clear();
}

// here the target output will be a single value
bool M5Tree::trainInstance(classPair &instance){

  if (DTDEBUG) cout << id << " trainInstance" << endl;

  //  featPct *= 0.99;

  nfeat = instance.in.size();

  bool modelChanged = false;

  // simply add this instance to the set of experiences

  // take from static array until we run out
  tree_experience *e;
  if (nExperiences < N_M5_EXP){
    // from statically created set of experiences
    e = &(allExp[nExperiences]);

  } else {
    // dynamically create experience
    e = new tree_experience;
  }


  e->input = instance.in;
  e->output = instance.out;
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
  //  DTDEBUG = true;
  //  SPLITDEBUG = true;
  //  INCDEBUG = true;
  //}

  if ( DTDEBUG) {
    cout << "Original input: ";
    for (unsigned i = 0; i < instance.in.size(); i++){
      cout << instance.in[i] << ", ";
    }
    cout << endl << " Original output: " << instance.out << endl;
    cout << "Added exp id: " << nExperiences << " output: " << e->output << endl;
    cout << "Address: " << e << " Input : ";
    for (unsigned i = 0; i < e->input.size(); i++){
      cout << e->input[i] << ", ";
    }
    cout << endl << " Now have " << nExperiences << " experiences." << endl;
  }

  // depending on mode/etc, maybe re-build tree

  // mode 0: re-build every step
  if (mode == BUILD_EVERY || nExperiences <= 1){
    rebuildTree();
    modelChanged = true;
  }

  // mode 1: re-build on error only
  else if (mode == BUILD_ON_ERROR){

    // build on misclassification
    // check for misclassify
    std::map<float, float> answer;
    testInstance(e->input, &answer);
    float val = answer.begin()->first;
    float error = fabs(val - e->output);

    if (error > 0.0){
      rebuildTree();
      modelChanged = true;
    }
  }

  // mode 2: re-build every FREQ steps
  else if (mode == BUILD_EVERY_N){
    // build every freq steps
    if (!modelChanged && (nExperiences % freq) == 0){
      rebuildTree();
      modelChanged = true;
    }
  }

  if (modelChanged){
    if (DTDEBUG) cout << "DT " << id << " tree re-built." << endl;

    if (DTDEBUG){
      cout << endl << "DT: " << id << endl;
      printTree(root, 0);
      cout << "Done printing tree" << endl;
    }
  }

  return modelChanged;

}


// here the target output will be a single value
bool M5Tree::trainInstances(std::vector<classPair> &instances){
  if (DTDEBUG) cout << id << " DT trainInstances: " 
                            << instances.size() 
                            << " nExp: " << nExperiences << endl;
  nfeat = instances[0].in.size();
  
  //  featPct *= 0.99;
  
  bool modelChanged = false;

  bool doBuild = false;

  // loop through instances, possibly checking for errors
  for (unsigned a = 0; a < instances.size(); a++){
    classPair instance = instances[a];

    // simply add this instance to the set of experiences

    // take from static array until we run out
    tree_experience *e;
    if (nExperiences < N_M5_EXP){
      // from statically created set of experiences
      e = &(allExp[nExperiences]);

    } else {
      // dynamically create experience
      e = new tree_experience;
    }


    e->input = instance.in;
    e->output = instance.out;
    experiences.push_back(e);
    nExperiences++;

    if (nExperiences == 1000000){
      cout << "Reached limit of # experiences allowed." << endl;
      return false;
    }

    if (nExperiences != (int)experiences.size())
      cout << "ERROR: experience size mismatch: " << nExperiences << ", " << experiences.size() << endl;

    if (DTDEBUG) {
      cout << "Original input: ";
      for (unsigned i = 0; i < instance.in.size(); i++){
        cout << instance.in[i] << ", ";
      }
      cout << endl << " Original output: " << instance.out << endl;
      cout << "Added exp id: " << nExperiences << " output: " << e->output << endl;
      cout << "Address: " << e << " Input : ";
      for (unsigned i = 0; i < e->input.size(); i++){
        cout << e->input[i] << ", ";
      }
      cout << endl << " Now have " << nExperiences << " experiences." << endl;
    }

    // depending on mode/etc, maybe re-build tree

    // don't need to check if we've already decided
    if (doBuild) continue;

    // mode 0: re-build every step
    if (mode == BUILD_EVERY || nExperiences <= 1){
      doBuild = true;
    }

    // mode 1: re-build on error only
    else if (mode == BUILD_ON_ERROR){

      // build on misclassification
      // check for misclassify
      std::map<float, float> answer;
      testInstance(e->input, &answer);
      float val = answer.begin()->first;
      float error = fabs(val - e->output);

      if (error > 0.0){
        doBuild = true;
      }

    }

    // mode 2: re-build every FREQ steps
    else if (mode == BUILD_EVERY_N){
      // build every freq steps
      if (!modelChanged && (nExperiences % freq) == 0){
        doBuild = true;
      }
    }

  } // loop of new instances

  if (DTDEBUG) cout << "Added " << instances.size() << " new instances. doBuild = " << doBuild << endl;

  if (doBuild){
    rebuildTree();
    modelChanged = true;
  }

  if (modelChanged){
    if (DTDEBUG) cout << "DT " << id << " tree re-built." << endl;

    if (DTDEBUG){
      cout << endl << "DT: " << id << endl;
      printTree(root, 0);
      cout << "Done printing tree" << endl;
    }
  }

  return modelChanged;

}


void M5Tree::rebuildTree(){
  //cout << "rebuild tree " << id << " on exp: " << nExperiences << endl;
  //  deleteTree(root);
  buildTree(root, experiences, false);
  //cout << "tree " << id << " rebuilt. " << endl;
}


// TODO: here we want to return the probability of the output value being each of the possible values, in the stochastic case
void M5Tree::testInstance(const std::vector<float> &input, std::map<float, float>* retval){
  if (DTDEBUG) cout << "testInstance" << endl;

  retval->clear();

  // in case the tree is empty
  if (experiences.size() == 0){
    (*retval)[0.0] = 1.0;
    return;
  }

  // follow through tree to leaf
  tree_node* leaf = traverseTree(root, input);
  lastNode = leaf;

  // and return mapping of outputs and their probabilities
  leafPrediction(leaf, input, retval);

}

float M5Tree::getConf(const std::vector<float> &input){
  if (DTDEBUG) cout << "numVisits" << endl;

  // in case the tree is empty
  if (experiences.size() == 0){
    return 0;
  }

  if (lastNode == NULL){
    return 0;
  }

  // follow through tree to leaf
  //tree_node* leaf = traverseTree(root, input);

  // and return # in this leaf
  float conf = (float)lastNode->nInstances / (float)(2.0*M);
  if (conf > 1.0)
    return 1.0;
  else
    return conf;

}

// check to see if this state is one we should explore
// to get more info on potential splits


// init the tree
void M5Tree::initTree(){
  if (DTDEBUG) cout << "initTree()" << endl;
  root = allocateNode();

  if (DTDEBUG) cout << "   root id = " << root->id << endl;

  // just to ensure the diff models are on different random values
  for (int i = 0; i < id; i++){
    rng.uniform(0, 1);
  }

}



// init a tree node
void M5Tree::initTreeNode(tree_node* node){
  if (DTDEBUG) cout << "initTreeNode()";

  node->id = nnodes++;
  if (DTDEBUG) cout << " id = " << node->id << endl;

  totalnodes++;
  if (totalnodes > maxnodes){
    maxnodes = totalnodes;
    if (DTDEBUG) cout << id << " M5 MAX nodes: " << maxnodes << endl;
  }

  // split criterion
  node->dim = -1;
  node->val = -1;

  // current data
  node->nInstances = 0;
  node->constant = 0;

  // coefficients will get resized later
  //  node->coefficients.resize(2,0);

  // next nodes in tree
  node->l = NULL;
  node->r = NULL;

  node->leaf = true;

}

void M5Tree::deleteTree(tree_node* node){
  if (DTDEBUG) cout << "deleteTree, node=" << node->id << endl;

  if (node==NULL)
    return;

  totalnodes--;

  node->nInstances = 0;
  node->coefficients.clear();

  //recursively call deleteTree on children
  // then delete them
  if (!node->leaf){
    // left child
    if (node->l != NULL){
      deleteTree(node->l);
      deallocateNode(node->l);
      node->l = NULL;
    }

    // right child
    if (node->r != NULL){
      deleteTree(node->r);
      deallocateNode(node->r);
      node->r = NULL;
    }
  }

  node->leaf  = true;
  node->dim = -1;
  node->val = -1;
  node->constant = 0;
}


M5Tree::tree_node* M5Tree::getCorrectChild(tree_node* node,
                                           const std::vector<float> &input){

  if (DTDEBUG) cout << "getCorrectChild, node=" << node->id << endl;

  if (passTest(node->dim, node->val, input))
    return node->l;
  else
    return node->r;

}

M5Tree::tree_node* M5Tree::traverseTree(tree_node* node,
                                        const std::vector<float> &input){

  if (DTDEBUG) cout << "traverseTree, node=" << node->id << endl;

  while (!node->leaf){
    node = getCorrectChild(node, input);
  }

  return node;
}


bool M5Tree::passTest(int dim, float val, const std::vector<float> &input){
  if (DTDEBUG) cout << "passTest, dim=" << dim << ",val=" << val 
                    << ",input["<<dim<<"]=" << input[dim] <<endl;

  if (input[dim] > val)
    return false;
  else
    return true;

}


void M5Tree::buildTree(tree_node *node,
                       const std::vector<tree_experience*> &instances,
                       bool changed){
  if(DTDEBUG) cout << "buildTree, node=" << node->id
                   << ",nInstances:" << instances.size()
                   << ",chg:" << changed << endl;

  if (instances.size() == 0){
    cout << "Error: buildTree called on tree " << id << " node " << node->id << " with no instances." << endl;
    exit(-1);
  }


  // TODO: what about stochastic data?
  //std::vector<float> chiSquare = calcChiSquare(instances);

  // first, add instances to tree
  node->nInstances = instances.size();

  bool allSame = true;
  float val0 = instances[0]->output;
  for (unsigned i = 1; i < instances.size(); i++){
    if (instances[i]->output != val0){
      allSame = false;
      break;
    }
  }

  // see if they're all the same
  if (allSame){
    makeLeaf(node);
    node->constant = instances[0]->output;
    if (DTDEBUG){
      cout << "Tree " << id << " node " << node->id 
           << " All " << node->nInstances
           << " classified with output "
           << instances[0]->output << endl;
    }
    return;
  }

  // check if this is a leaf and linear model has no error
  if (node->leaf && node->coefficients.size() > 0) {
    //cout << "Node " << node->id << " is leaf, checking lm" << endl;
    float errorSum = 0;
    for (unsigned i = 0; i < instances.size(); i++){
      // get prediction for instance and compare with actual output
      tree_node* leaf = traverseTree(node, instances[i]->input);
      std::map<float, float> retval;
      leafPrediction(leaf, instances[i]->input, &retval);
      float prediction = retval.begin()->first;
      float absError = fabs(prediction - instances[i]->output);
      errorSum += absError;
      //cout << "instance " << i << " leaf predicted: " << prediction
      //     << " actual: " << instances[i]->output
      //   << " error: " << absError << endl;
    }
    float avgError = errorSum / (float)instances.size();
    if (avgError < 0.001){
      //      cout << "stick with linear model" << endl;
      return;
    }
  }

  // if not, calculate SDR to determine best split
  if (SPLITDEBUG) cout << endl << "Creating new decision node" << endl;

  node->leaf = false;
  //node->nInstances++;

  float bestSDR = -1.0;
  int bestDim = -1;
  float bestVal = -1;
  std::vector<tree_experience*> bestLeft;
  std::vector<tree_experience*> bestRight;

  testPossibleSplits(instances, &bestSDR, &bestDim, &bestVal, &bestLeft, &bestRight);

  implementSplit(node, instances, bestSDR, bestDim, bestVal, bestLeft, bestRight, changed);

  // possibly replace split node with linear regression model
  pruneTree(node, instances);

}


void M5Tree::makeLeaf(tree_node* node){

  removeChildren(node);

  // make sure there are enough coefficients for all the features
  // and that they are 0
  if (node->coefficients.size() != (unsigned)nfeat){
    node->coefficients.resize(nfeat, 0);
  }
  for (unsigned i = 0; i < node->coefficients.size(); i++){
    node->coefficients[i] = 0;
  }

}


void M5Tree::removeChildren(tree_node* node){
  // check on children
  if (node->l != NULL){
    deleteTree(node->l);
    deallocateNode(node->l);
    node->l = NULL;
  }

  if (node->r != NULL){
    deleteTree(node->r);
    deallocateNode(node->r);
    node->r = NULL;
  }

  node->leaf = true;
}


void M5Tree::pruneTree(tree_node *node,
                       const std::vector<tree_experience*> &instances){
  if (LMDEBUG || DTDEBUG){
    printTree(root, 0);
    cout << "pruneTree, node=" << node->id
         << ",nInstances:" << instances.size() << endl;
  }

  // TODO: no pruning right now
  //  return;

  // calculate error of current subtree
  float subtreeErrorSum = 0;
  for (unsigned i = 0; i < instances.size(); i++){
    
    // get prediction for instance and compare with actual output
    tree_node* leaf = traverseTree(node, instances[i]->input);
    std::map<float, float> retval;
    leafPrediction(leaf, instances[i]->input, &retval);
    float prediction = retval.begin()->first;
    float absError = fabs(prediction - instances[i]->output);
    subtreeErrorSum += absError;
    if (LMDEBUG || DTDEBUG){
      cout << "instance " << i << " subtree predicted: " << prediction
           << " actual: " << instances[i]->output
           << " error: " << absError << endl;
    }
  }
  if (instances.size() < 3){
    if (LMDEBUG || DTDEBUG) cout << "instances size <= 2!!!" << endl;
    return;
  }

  float avgTreeError = subtreeErrorSum / (float)instances.size();

  // if this is zero error, we're not going to replace it
  if (false && avgTreeError <= 0.0001){
    if (LMDEBUG || DTDEBUG) 
      cout << "Sub-tree is perfect (" << avgTreeError
           << "), do not attempt LM" << endl;
    return;
  }
                                                                
  // figure out tree feats used
  std::vector<bool> treeFeatsUsed(instances[0]->input.size(), false);
  getFeatsUsed(node, &treeFeatsUsed);
  int nTreeFeatsUsed = 0;
  for (unsigned i = 0; i < treeFeatsUsed.size(); i++){
    if (treeFeatsUsed[i])
      nTreeFeatsUsed++;
  }

  // Just use them all... otherwise we ignore some that weren't good enough
  // for splitting, but are good here
  std::vector<bool> featsUsed(instances[0]->input.size(), true);
  int nFeatsUsed = featsUsed.size();

  // or just use ones allowed by subtree
  if (!ALLOW_ALL_FEATS){
    featsUsed = treeFeatsUsed;
    nFeatsUsed = nTreeFeatsUsed;
  }

  // no features to build linear model on
  if (nFeatsUsed == 0){
    if (LMDEBUG || DTDEBUG) cout << "No features for LM" << endl;
    return;
  }

  // add on error bonus based on nInstances and nParams
  float treeErrorEst = avgTreeError;
  float denom = (instances.size() - nTreeFeatsUsed);
  if (denom < 1){
    denom = 0.5;
    if (LMDEBUG) {
      cout << "denom of tree error factor is " << denom 
           << " with nInst " << instances.size() 
           << " nfeats: " << nTreeFeatsUsed << endl;
    }
  }
  treeErrorEst *= (instances.size() + nTreeFeatsUsed) / denom;

  // fit linear model to this set of instances
  float lmErrorSum = 0;
  int nlmFeats = 0;
  if (SIMPLE)
    nlmFeats = fitSimpleLinearModel(node, instances, featsUsed, nFeatsUsed, &lmErrorSum);
  else 
    nlmFeats = fitLinearModel(node, instances, featsUsed, nFeatsUsed, &lmErrorSum);

  float avgLMError = lmErrorSum / (float)instances.size();

  float lmErrorEst = avgLMError;
  float denom2 = (instances.size() - nlmFeats);
  if (denom2 < 1){
    denom2 = 0.5;
    if (LMDEBUG) {
      cout << "denom2 of lm error factor is " << denom2
           << " with nInst " << instances.size() 
           << " nfeats: " << nlmFeats << endl;
    }
  }
  lmErrorEst *= (instances.size() + nlmFeats) / denom2; 

  // replace subtree with linear model?
  if (LMDEBUG || DTDEBUG) {
    cout << "Sub-Tree Error: " << treeErrorEst << ", lm Error: "
         << lmErrorEst << endl;
  }
  //  if (lmErrorEst < (treeErrorEst + 0.0001)){
  if (lmErrorEst < (treeErrorEst + 0.1*MIN_SDR)){
  //if (lmErrorEst < treeErrorEst){
    if (LMDEBUG || DTDEBUG)
      cout << node->id << " replace tree with linear model" << endl;
    removeChildren(node);
  } else {
    // remove coefficients again, for memory
    //    node->coefficients.clear();
  }

}


int M5Tree::fitLinearModel(tree_node *node,
                           const std::vector<tree_experience*> &instances,
                           std::vector<bool> featureMask,
                           int nFeats, float* resSum){
  
  if(DTDEBUG || LMDEBUG) cout << "fitLinearModel, node=" << node->id
                              << ",nInstances:" << instances.size() << endl;

  // make sure there are enough coefficients for all the features
  if (node->coefficients.size() != instances[0]->input.size()){
    node->coefficients.resize(instances[0]->input.size(), 0);
  }

  node->constant = 0.0;
  bool doRegression = true;
  int ntimes = 0;
  int nlmFeats = 10;
  (*resSum) = 1000000;

  while (doRegression){
    //cout << id << " Attempt linear model " << ntimes << endl;
    ntimes++;

    int nObs = (int)instances.size();

    //cout << "with nObs: " << nObs << " and nFeats: " << nFeats << endl;

    // no feats or obs, no model to build
    if (nObs == 0 || nFeats == 0)
      break;

    Matrix X(nObs, nFeats);
    ColumnVector Y(nObs);

    std::vector<int> featIndices(nFeats);

    std::vector<bool> constants(nFeats,true);
    bool foundProblem = false;
    
    // load up matrices
    for (int i = 0; i < nObs; i++){
      tree_experience *e = instances[i];
      if (LMDEBUG) cout << "Obs: " << i;

      // go through all features
      int featIndex = 1;
      for (unsigned j = 0; j < featureMask.size(); j++){
        node->coefficients[j] = 0;
        if (!featureMask[j])
          continue;
      
        if (constants[j] && e->input[j] != instances[0]->input[j]){
          constants[j] = false;
        }

	/*
        if (rng.uniform() < featPct){
          featureMask[j] = false;
          nFeats--;
          if (nFeats > 0)
            continue;
          else
            break;
        }
	*/

        if (i == nObs-1 && constants[j]){
          //cout << "PROBLEM: feat " << j << " is constant!" << endl;
          foundProblem = true;
          featureMask[j] = false;
          nFeats--;
          if (nFeats > 0)
            continue;
          else
            break;
        }

        featIndices[featIndex-1] = j;
        // HACK: I'm adding random noise here to prevent colinear features
        X(i+1,featIndex) = e->input[j]; // + rng.uniform(-0.00001, 0.00001);
        if (LMDEBUG){
          cout << " Feat " << featIndex << " index " << j
               << " val " << X(i+1,featIndex) << ",";
        }
        featIndex++;
      }

      Y(i+1) = e->output;
      if (LMDEBUG) cout << " out: " << e->output << endl;
    }

    if (foundProblem || nFeats == 0)
      continue;

    // make vector of 1s
    ColumnVector Ones(nObs); Ones = 1.0;

    // calculate means (averages) of x1 and x2 [ .t() takes transpose]
    RowVector Mrow = Ones.t() * X / nObs;

    // and subtract means from x1 and x1
    Matrix XC(nObs,nFeats);
    XC = X - Ones * Mrow;

    // do the same to Y [use Sum to get sum of elements]
    ColumnVector YC(nObs);
    Real mval = Sum(Y) / nObs;
    YC = Y - Ones * mval;

    Try {

      // form sum of squares and product matrix
      //    [use << rather than = for copying Matrix into SymmetricMatrix]
      SymmetricMatrix SSQ;
      SSQ << XC.t() * XC;
      
      ///////////////////////////
      // Cholesky Method
      LowerTriangularMatrix L = Cholesky(SSQ);

      // calculate estimate
      ColumnVector A = L.t().i() * (L.i() * (XC.t() * YC));
      ///////////////////////////
      
      //////////////////////////
      // Least Squares Method
      // calculate estimate
      //    [bracket last two terms to force this multiplication first]
      //    [ .i() means inverse, but inverse is not explicity calculated]
      //ColumnVector A = SSQ.i() * (XC.t() * YC);
      //////////////////////////
  
  
      // calculate estimate of constant term
      //    [AsScalar converts 1x1 matrix to Real]
      Real a = mval - (Mrow * A).AsScalar();

      // Calculate fitted values and residuals
      //int npred1 = nFeats+1;
      ColumnVector Fitted = X * A + a;
      ColumnVector Residual = Y - Fitted;
      //Real ResVar = Residual.SumSquare() / (nObs-npred1);

      // print out answers
      // for each instance
      (*resSum) = 0;
      for (int i = 0; i < nObs; i++){
        if (DTDEBUG || LMDEBUG){
          cout << "instance " << i << " linear model predicted: " << Fitted(i+1)
               << " actual: " << instances[i]->output
               << " error: " << Residual(i+1) << endl;
        }
        (*resSum) += fabs(Residual(i+1));
      }


      // coeff
      nlmFeats = 0;
      for (int i = 0; i < nFeats; i++){
        if (DTDEBUG || LMDEBUG) cout << "Coeff " << i << " on feat: " << featIndices[i] << " is " << A(i+1) << endl;
        node->coefficients[featIndices[i]] = A(i+1);
        if (A(i+1) != 0)
          nlmFeats++;
      }
      if (DTDEBUG || LMDEBUG) cout << "constant is " << a << endl;
      node->constant = a;


    }

    CatchAll {
      // had an error trying the linear regression.
      // HACK TODO: for now, turn off one variable
      //cout << ntimes << " linear regression had exception" << endl;
      //<< BaseException::what() <<endl;

      // tried this already, stop now
      if (ntimes > 1 || nFeats < 2){
        //cout << "max regression" << endl;
        doRegression = false;
        break;
      }
      for (unsigned j = 0; j < featureMask.size(); j++){
        if (featureMask[j]){
          //cout << "remove feature " << j << endl;
          featureMask[j] = false;
          nFeats--;
          break;
        }
      }
      continue;
    }

    // it worked, dont need to do it again
    doRegression = false;

  }

  // return # features used
  return nlmFeats;

}

int M5Tree::fitSimpleLinearModel(tree_node *node,
                                 const std::vector<tree_experience*> &instances,
                                 std::vector<bool> featureMask,
                                 int nFeats, float* resSum){
  if(DTDEBUG || LMDEBUG) cout << "fitSimpleLinearModel, node=" << node->id
                              << ",nInstances:" << instances.size() << endl;
  
  // make sure there are enough coefficients for all the features
  if (node->coefficients.size() != instances[0]->input.size()){
    node->coefficients.resize(instances[0]->input.size(), 0);
  }

  // loop through all features, try simple single variable regression
  // keep track of error/coefficient of each
  int bestFeat = -1;
  float bestCoeff = -1;
  float bestError = 1000000;
  float bestConstant = -1;

  std::vector<float> xsum(instances[0]->input.size(), 0);
  std::vector<float> xysum(instances[0]->input.size(), 0);
  std::vector<float> x2sum(instances[0]->input.size(), 0);
  float ysum = 0;

  int nObs = (int)instances.size();
  for (int i = 0; i < nObs; i++){
    tree_experience *e = instances[i];
    if (LMDEBUG) cout << "Obs: " << i;
    
    // go through all features
    for (unsigned j = 0; j < instances[0]->input.size(); j++){
      if (!featureMask[j]) continue;
      if (LMDEBUG) cout << ", F" << j << ": " << e->input[j];
      xsum[j] += e->input[j];
      xysum[j] += (e->input[j] * e->output);
      x2sum[j] += (e->input[j] * e->input[j]);
    }
    ysum += e->output;
    if (LMDEBUG) cout << ", out: " << e->output << endl;
  }
  
  // now go through all features and calc coeff and constant
  for (unsigned j = 0; j < instances[0]->input.size(); j++){
    if (!featureMask[j]) continue;
    /*
      if (rng.uniform() < featPct){
      continue;
      }
    */
    float coeff = (xysum[j] - xsum[j]*ysum/nObs)/(x2sum[j]-(xsum[j]*xsum[j])/nObs);
    float constant = (ysum/nObs) - coeff*(xsum[j]/nObs);
    
    if (LMDEBUG) {
      cout << "Feat " << j << " coeff: " << coeff << ", constant: " 
           << constant << " mask: " << featureMask[j] << endl;
    }

    // now try to make predictions and see what error is
    float errorSum = 0;
    for (int i = 0; i < nObs; i++){
      tree_experience *e = instances[i];
      float pred = constant + coeff * e->input[j];
      float error = fabs(pred - e->output);
      if (LMDEBUG) cout << "Instance " << i << " error: " << error << endl;
      errorSum += error;
    }
    if (LMDEBUG) cout << "eSum: " << errorSum << endl;

    // check if this is the best
    if (errorSum < bestError){
      bestError = errorSum;
      bestFeat = j;
      bestConstant = constant;
      bestCoeff = coeff;
    }
  }
  
  if (LMDEBUG){
    cout << "SLM feat: " << bestFeat << " coeff: " 
         << bestCoeff << " constant: " 
         << bestConstant << " avgE: " << (bestError/nObs) << endl;
  }

  if (bestFeat < 0 || bestFeat > (int)instances[0]->input.size()){
    node->constant = 0.0;
    for (unsigned i = 0; i < node->coefficients.size(); i++){
      node->coefficients[i] = 0.0;
    }
    (*resSum) = 1000000;
    return 10;
  }

  // pick best feature somehow
  node->constant = bestConstant;
  for (unsigned i = 0; i < node->coefficients.size(); i++){
    if (i == (unsigned)bestFeat){
      node->coefficients[i] = bestCoeff;
      if (LMDEBUG) cout << "Set coefficient on feat " << i << " to " << bestCoeff << endl;
    }
    else {
      node->coefficients[i] = 0.0;
    }
  }

  (*resSum) = bestError;

  // only use 1 input feature this way
  return 1;
 
}




void M5Tree::implementSplit(tree_node* node, 
                            const std::vector<tree_experience*> &instances,
                            float bestSDR, int bestDim,
                            float bestVal, 
                            const std::vector<tree_experience*> &bestLeft,
                            const std::vector<tree_experience*> &bestRight,
                            bool changed){
  if (DTDEBUG) cout << "implementSplit node=" << node->id
                    << ",sdr=" << bestSDR
                    << ",dim=" << bestDim
                    << ",val=" << bestVal 
                    << ",chg=" << changed << endl;


  // see if this should still be a leaf node
  if (bestSDR < MIN_SDR){
    makeLeaf(node);
    float valSum = 0;
    for (unsigned i = 0; i < instances.size(); i++){
      valSum += instances[i]->output;
    }
    float avg = valSum / (float)(instances.size());
    node->constant = avg;
    if (SPLITDEBUG || STOCH_DEBUG){
      cout << "DT " << id << " Node " << node->id << " Poor sdr "
           << node->nInstances
           << " instances classified at leaf " << node->id
           << " with multiple outputs " << endl;
    }
    return;
  }

  // see if this split changed or not
  // assuming no changes above
  if (!changed && node->dim == bestDim && node->val == bestVal
       && !node->leaf && node->l != NULL && node->r != NULL){
    // same split as before.
    if (DTDEBUG || SPLITDEBUG) cout << "Same split as before" << endl;

    // see which leaf changed
    if (bestLeft.size() > (unsigned)node->l->nInstances){
      // redo left side
      if (DTDEBUG) cout << "Rebuild left side of tree" << endl;
      buildTree(node->l, bestLeft, changed);
    }

    if (bestRight.size() > (unsigned)node->r->nInstances){
      // redo right side
      if (DTDEBUG) cout << "Rebuild right side of tree" << endl;
      buildTree(node->r, bestRight, changed);
    }
    return;
  }


  // totally new
  // set the best split here
  node->leaf = false;
  node->dim = bestDim;
  node->val = bestVal;

  if (SPLITDEBUG) cout << "Best split was cut with val " << node->val
                       << " on dim " << node->dim
                       << " with sdr: " << bestSDR << endl;

  if (DTDEBUG) cout << "Left has " << bestLeft.size()
                    << ", right has " << bestRight.size() << endl;

  // make sure both instances
  if (bestLeft.size() == 0 || bestRight.size() == 0){
    cout << "ERROR: DT " << id << " node " << node->id << " has 0 instances: left: " << bestLeft.size()
         << " right: " << bestRight.size() << endl;
    cout << "Split was cut with val " << node->val
         << " on dim " << node->dim
         << " with sdr: " << bestSDR << endl;
    exit(-1);
  }


  // check if these already exist
  if (node->l == NULL){
    if (DTDEBUG) cout << "Init new left tree nodes " << endl;
    node->l = allocateNode();
  }
  if (node->r == NULL){
    if (DTDEBUG) cout << "Init new right tree nodes " << endl;
    node->r = allocateNode();
  }

  // recursively build the sub-trees to this one
  if (DTDEBUG) cout << "Building left tree for node " << node->id << endl;
  buildTree(node->l, bestLeft, true);
  if (DTDEBUG) cout << "Building right tree for node " << node->id << endl;
  buildTree(node->r, bestRight, true);

}


void M5Tree::getFeatsUsed(tree_node* node, std::vector<bool> *featsUsed){

  // if leaf, ones from linear model
  if (node->leaf){
    //cout << "coeff size: " << node->coefficients.size() << endl;
    for (unsigned i = 0; i < node->coefficients.size(); i++){
      if (node->coefficients[i] != 0){
        if (LMDEBUG || DTDEBUG) cout << "Leaf node, used coeff " << i << endl;
        (*featsUsed)[i] = true;
      }
    }
    return;
  }

  // otherwise see what split was used
  // and call for left and right sub-trees
  (*featsUsed)[node->dim] = true;
  if (LMDEBUG || DTDEBUG) cout << "Split node, used feat " << node->dim << endl;

  getFeatsUsed(node->l, featsUsed);
  getFeatsUsed(node->r, featsUsed);

  return;
}


void M5Tree::testPossibleSplits(const std::vector<tree_experience*> &instances,
                                float *bestSDR, int *bestDim,
                                float *bestVal, 
                                std::vector<tree_experience*> *bestLeft,
                                std::vector<tree_experience*> *bestRight) {
  if (DTDEBUG) cout << "testPossibleSplits" << endl;


  // calculate sd for the set
  float sd = calcSDforSet(instances);
  //if (DTDEBUG) cout << "I: " << I << endl;

  int nties = 0;

  // for each possible split, calc standard deviation reduction
  for (unsigned idim = 0; idim < instances[0]->input.size(); idim++){

    //float* sorted = sortOnDim(idim, instances);
    float minVal, maxVal;
    std::set<float> uniques = getUniques(idim, instances, minVal, maxVal);

    for (std::set<float>::iterator j = uniques.begin(); j != uniques.end(); j++){

      // skip max val, not a valid cut for either
      if ((*j) == maxVal)
        continue;

      // if this is a random forest, we eliminate some random number of splits
      // here (decision is taken from the random set that are left)
      if (rng.uniform() < featPct)
        continue;

      std::vector<tree_experience*> left;
      std::vector<tree_experience*> right;

      // splits that are cuts
      float splitval = (*j);
      float sdr = calcSDR(idim, splitval, instances, sd, left, right);

      if (SPLITDEBUG) cout << " CUT split val " << splitval
                           << " on dim: " << idim << " had sdr "
                           << sdr << endl;

      // see if this is the new best sdr
      compareSplits(sdr, idim, splitval, left, right, &nties,
                    bestSDR, bestDim, bestVal, bestLeft, bestRight);


    } // j loop
  }
}



void M5Tree::compareSplits(float sdr, int dim, float val, 
                           const std::vector<tree_experience*> &left, 
                           const std::vector<tree_experience*> &right,
                           int *nties, float *bestSDR, int *bestDim,
                           float *bestVal, 
                           std::vector<tree_experience*> *bestLeft,
                           std::vector<tree_experience*> *bestRight){
  if (DTDEBUG) cout << "compareSplits sdr=" << sdr << ",dim=" << dim
                    << ",val=" << val  <<endl;


  bool newBest = false;

  // if its a virtual tie, break it randomly
  if (fabs(*bestSDR - sdr) < SPLIT_MARGIN){
    //cout << "Split tie, making random decision" << endl;

    (*nties)++;
    float randomval = rng.uniform(0,1);
    float newsplitprob = (1.0 / (float)*nties);

    if (randomval < newsplitprob){
      newBest = true;
      if (SPLITDEBUG) cout << "   Tie on split. DT: " << id << " rand: " << randomval
                           << " splitProb: " << newsplitprob << ", selecting new split " << endl;
    }
    else
      if (SPLITDEBUG) cout << "   Tie on split. DT: " << id << " rand: " << randomval
                           << " splitProb: " << newsplitprob << ", staying with old split " << endl;
  }

  // if its clearly better, set this as the best split
  else if (sdr > *bestSDR){
    newBest = true;
    *nties = 1;
  }


  // set the split features
  if (newBest){
    *bestSDR = sdr;
    *bestDim = dim;
    *bestVal = val;
    *bestLeft = left;
    *bestRight = right;
    if (SPLITDEBUG){
      cout << "  New best sdr: " << *bestSDR
           << " with val " << *bestVal
           << " on dim " << *bestDim << endl;
    }
  } // newbest
}

float M5Tree::calcSDR(int dim, float val, 
                      const std::vector<tree_experience*> &instances,
                      float sd,
                      std::vector<tree_experience*> &left,
                      std::vector<tree_experience*> &right){
  if (DTDEBUG) cout << "calcSDR, dim=" << dim
                    << " val=" << val
                    << " sd=" << sd
                    << " nInstances= " << instances.size() << endl;

  left.clear();
  right.clear();

  // split into two sides
  for (unsigned i = 0; i < instances.size(); i++){
    if (DTDEBUG) cout << "calcSDR - Classify instance " << i 
                      << " on new split " << endl;

    if (passTest(dim, val, instances[i]->input)){
      left.push_back(instances[i]);
    }
    else{
      right.push_back(instances[i]);
    }
  }

  if (DTDEBUG) cout << "Left has " << left.size()
                    << ", right has " << right.size() << endl;

  // get sd for both sides
  float sdLeft = calcSDforSet(left);
  float sdRight = calcSDforSet(right);

  float leftRatio = (float)left.size() / (float)instances.size();
  float rightRatio = (float)right.size() / (float)instances.size();

  float sdr = sd - (leftRatio * sdLeft + rightRatio * sdRight);

  if (DTDEBUG){
    cout << "LeftSD: " << sdLeft
         << " RightSD: " << sdRight
         << " SD: " << sd
         << " SDR: " << sdr
         << endl;
  }

  return sdr;

}

float M5Tree::calcSDforSet(const std::vector<tree_experience*> &instances){
  if (DTDEBUG) cout << "calcSDforSet" << endl;

  int n = instances.size();

  if (n == 0)
    return 0;

  double sum = 0;
  double sumSqr = 0;

  // go through instances and calculate sums, sum of squares
  for (unsigned i = 0; i < instances.size(); i++){
    float val = instances[i]->output;
    sum += val;
    sumSqr += (val * val);
  }

  double mean = sum / (double)n;
  double variance = (sumSqr - sum*mean)/(double)n;
  float sd = sqrt(variance);

  return sd;

}

std::set<float> M5Tree::getUniques(int dim, const std::vector<tree_experience*> &instances, float& minVal, float& maxVal){
  if (DTDEBUG) cout << "getUniques,dim = " << dim;

  std::set<float> uniques;

  for (int i = 0; i < (int)instances.size(); i++){
    if (i == 0 || instances[i]->input[dim] < minVal)
      minVal = instances[i]->input[dim];
    if (i == 0 || instances[i]->input[dim] > maxVal)
      maxVal = instances[i]->input[dim];

    uniques.insert(instances[i]->input[dim]);
  }

  // lets not try more than 100 possible splits per dimension
  if (uniques.size() > 100){
    float rangeInc = (maxVal - minVal) / 100.0;
    uniques.clear();
    for (float i = minVal; i < maxVal; i+= rangeInc){
      uniques.insert(i);
    }
  }
  uniques.insert(maxVal);

  if (DTDEBUG) cout << " #: " << uniques.size() << endl;
  return uniques;
}


float* M5Tree::sortOnDim(int dim, const std::vector<tree_experience*> &instances){
  if (DTDEBUG) cout << "sortOnDim,dim = " << dim << endl;

  float* values = new float[instances.size()];

  for (int i = 0; i < (int)instances.size(); i++){
    //cout << "Instance " << i << endl;

    float val = instances[i]->input[dim];
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

  if (DTDEBUG){
    cout << "Sorted array: " << values[0];
    for (int i = 1; i < (int)instances.size(); i++){
      cout << ", " << values[i];
    }
    cout << endl;
  }

  return values;

}


void M5Tree::printTree(tree_node *t, int level){

  for (int i = 0; i < level; i++){
    cout << ".";
  }

  cout << "Node " << t->id  << " nInstances: " << t->nInstances ;

  // Leaf, print regression stuff
  if (t->leaf){
    cout << " Constant: " << t->constant << " coeff: ";
    for (unsigned j = 0; j < t->coefficients.size(); j++){
      if (!SIMPLE)
        cout << t->coefficients[j] << ", ";
      if (SIMPLE && t->coefficients[j] != 0)
        cout << " on feat " << j << ": " << t->coefficients[j];
    }
    cout << endl;
    return;
  }

  // otherwise, print split info, next nodes

  cout << " Type: CUT";
  cout << " Dim: " << t->dim << " Val: " << t->val;
  cout << " Left: " << t->l->id << " Right: " << t->r->id << endl;

  // print children
  if (t->dim != -1 && !t->leaf){
    printTree(t->l, level+1);
    printTree(t->r, level+1);
  }

}


// output a map of outcomes and their probabilities for this leaf node
void M5Tree::leafPrediction(tree_node* leaf, const std::vector<float> &input, std::map<float, float>* retval){
  if (DTDEBUG)
    cout << "Calculating output for leaf " << leaf->id << endl;

  float prediction = leaf->constant;
  if (DTDEBUG) cout << "leaf constant: " << leaf->constant << endl;

  // plus each coefficient
  for (unsigned i = 0; i < input.size(); i++){
    prediction += leaf->coefficients[i] * input[i];
    if (DTDEBUG) {
      cout << "feat " << i << " coeff: " << leaf->coefficients[i]
           << " on input " << input[i] << endl;
    }
  }

  if (DTDEBUG) cout << " prediction: " << prediction << endl;

  (*retval)[prediction] = 1.0;
}


void M5Tree::initNodes(){

  for (int i = 0; i < N_M5_NODES; i++){
    initTreeNode(&(allNodes[i]));
    freeNodes.push_back(i);
    if (NODEDEBUG) 
      cout << "init node " << i << " with id " << allNodes[i].id 
           << ", now " << freeNodes.size() << " free nodes." << endl;
  }

}

M5Tree::tree_node* M5Tree::allocateNode(){
  if (freeNodes.empty()){
    tree_node* newNode = new tree_node;
    initTreeNode(newNode);
    if (NODEDEBUG) 
      cout << "PROBLEM: No more pre-allocated nodes!!!" << endl
           << "return new node " << newNode->id 
           << ", now " << freeNodes.size() << " free nodes." << endl;
    return newNode;
  }

  int i = freeNodes.back();
  freeNodes.pop_back();
  if (NODEDEBUG) 
    cout << "allocate node " << i << " with id " << allNodes[i].id 
         << ", now " << freeNodes.size() << " free nodes." << endl;
  return &(allNodes[i]);
}

void M5Tree::deallocateNode(tree_node* node){
  if (node->id >= N_M5_NODES){
    if (NODEDEBUG) 
      cout << "dealloc extra node id " << node->id 
           << ", now " << freeNodes.size() << " free nodes." << endl;
    delete node;
    return;
  }

  freeNodes.push_back(node->id);
  if (NODEDEBUG) 
    cout << "dealloc node " << node->id 
         << ", now " << freeNodes.size() << " free nodes." << endl;
}
