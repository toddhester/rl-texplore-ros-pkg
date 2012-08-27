/** \file C45Tree.cc
    Implements the C4.5 decision tree class.
    This is an implementation of C4.5 decision trees described in:
    J. R. Quinlan, "Induction of decision trees," Machine Learning, vol 1. pp 81-106, 1986.
    \author Todd Hester
*/

#include "C45Tree.hh"



C45Tree::C45Tree(int id, int trainMode, int trainFreq, int m,
                 float featPct, Random rng):
  id(id), mode(trainMode), freq(trainFreq), M(m),
  featPct(featPct), ALLOW_ONLY_SPLITS(true), rng(rng)
{

  nnodes = 0;
  nOutput = 0;
  nExperiences = 0;
  hadError = false;
  maxnodes = N_C45_NODES;
  totalnodes = 0;

  // how close a split has to be to be randomly selected
  SPLIT_MARGIN = 0.0; //0.02; //5; //01; //0.05; //0.2; //0.05;

  MIN_GAIN_RATIO = 0.0001; //0.0004; //0.001; //0.0002; //0.001;

  DTDEBUG = false; //true;
  SPLITDEBUG = false; //true;
  STOCH_DEBUG = false; //true; //false; //true;
  INCDEBUG = false; //true; //false; //true;
  NODEDEBUG = false;
  COPYDEBUG = false;

  cout << "Created C4.5 decision tree " << id;
  if (DTDEBUG) {
    cout << " mode: " << mode << " freq: " << freq << endl;
  } else {
    cout << endl;
  }

  initNodes();
  initTree();

}

C45Tree::C45Tree(const C45Tree &t):
  id(t.id), mode(t.mode), freq(t.freq), M(t.M),
  featPct(t.featPct), ALLOW_ONLY_SPLITS(t.ALLOW_ONLY_SPLITS), rng(t.rng)
{
  COPYDEBUG = t.COPYDEBUG;
  if (COPYDEBUG) cout << "  C4.5 tree copy constructor id " << id << endl;
  nnodes = 0;
  nOutput = t.nOutput;
  nExperiences = t.nExperiences;
  hadError = t.hadError;
  maxnodes = t.maxnodes;
  totalnodes = 0;

  SPLIT_MARGIN = t.SPLIT_MARGIN;
  MIN_GAIN_RATIO = t.MIN_GAIN_RATIO;
  DTDEBUG = t.DTDEBUG;
  SPLITDEBUG = t.SPLITDEBUG;
  STOCH_DEBUG = t.STOCH_DEBUG;
  INCDEBUG = t.INCDEBUG;
  NODEDEBUG = t.NODEDEBUG;
  
  
  if (COPYDEBUG) cout << "   C4.5 copy nodes, experiences, root, etc" << endl;
  // copy all experiences
  for (int i = 0; i < N_C45_EXP; i++){
    allExp[i] = t.allExp[i];
  }
  if (COPYDEBUG) cout << "   C4.5 copied exp array" << endl;

  // set experience pointers
  experiences.resize(t.experiences.size());
  for (unsigned i = 0; i < t.experiences.size(); i++){
    experiences[i] = &(allExp[i]);
  }
  if (COPYDEBUG) cout << "   C4.5 set experience pointers" << endl;

  // now the tricky part, set the pointers inside the tree nodes correctly
  initNodes();

  if (COPYDEBUG) cout << "   C4.5 copy tree " << endl;
  root = allocateNode();
  lastNode = root;
  copyTree(root, t.root);
  if (COPYDEBUG) cout << "   C4.5 tree copy done" << endl;
   
  if (COPYDEBUG) {
    cout << endl << "New tree: " << endl;
    printTree(root, 0);
    cout << endl;
    cout << "  c4.5 copy done" << endl;
  }

}

C45Tree* C45Tree::getCopy(){
  
  C45Tree* copy = new C45Tree(*this);
  return copy;

}

void C45Tree::copyTree(tree_node* newNode, tree_node* origNode){

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


C45Tree::~C45Tree() {
  deleteTree(root);
  for (unsigned i = N_C45_EXP; i < experiences.size(); i++){
    delete experiences[i];
  }
  experiences.clear();
}

// here the target output will be a single value
bool C45Tree::trainInstance(classPair &instance){

  if (DTDEBUG) cout << "trainInstance" << endl;

  bool modelChanged = false;

  // simply add this instance to the set of experiences

  // take from static array until we run out
  tree_experience *e;
  if (nExperiences < N_C45_EXP){
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

  // mode 0: re-build every step
  if (mode == BUILD_EVERY || nExperiences <= 1){
    modelChanged = rebuildTree();
  }

  // mode 1: re-build on error only
  else if (mode == BUILD_ON_ERROR){

    // build on misclassification
    // check for misclassify
    // get leaf
    tree_node* leaf = traverseTree(root, e->input);
    // find probability for this output
    float count = (float)leaf->outputs[e->output];
    float outputProb = count / (float)leaf->nInstances;

    if (outputProb < 0.75){
      modelChanged = rebuildTree();
    }
  }

  // mode 2: re-build every FREQ steps
  else if (mode == BUILD_EVERY_N){
    // build every freq steps
    if (!modelChanged && (nExperiences % freq) == 0){
      modelChanged = rebuildTree();
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

  /*
  if (nExperiences % 50 == 0){
    cout << endl << "DT: " << id << endl;
    printTree(root, 0);
    cout << "Done printing tree" << endl;
  }
  */

  return modelChanged;

}


// here the target output will be a single value
bool C45Tree::trainInstances(std::vector<classPair> &instances){
  if (DTDEBUG) cout << "DT " << id << "  trainInstances: " << instances.size() << endl;

  bool modelChanged = false;

  bool doBuild = false;

  // loop through instances, possibly checking for errors
  for (unsigned a = 0; a < instances.size(); a++){
    classPair instance = instances[a];

    // simply add this instance to the set of experiences

    // take from static array until we run out
    tree_experience *e;
    if (nExperiences < N_C45_EXP){
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
      // get leaf
      tree_node* leaf = traverseTree(root, e->input);
      // find probability for this output
      float count = (float)leaf->outputs[e->output];
      float outputProb = count / (float)leaf->nInstances;

      if (outputProb < 0.75){
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
    modelChanged = rebuildTree();
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


bool C45Tree::rebuildTree(){
  return buildTree(root, experiences, false);
}


// TODO: here we want to return the probability of the output value being each of the possible values, in the stochastic case
void C45Tree::testInstance(const std::vector<float> &input, std::map<float, float>* retval){
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
  outputProbabilities(leaf, retval);

}

float C45Tree::getConf(const std::vector<float> &input){
  if (DTDEBUG) cout << "numVisits" << endl;

  // in case the tree is empty
  if (experiences.size() == 0){
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
void C45Tree::initTree(){
  if (DTDEBUG) cout << "initTree()" << endl;
  root = allocateNode();

  if (DTDEBUG) cout << "   root id = " << root->id << endl;

  // just to ensure the diff models are on different random values
  for (int i = 0; i < id; i++){
    rng.uniform(0, 1);
  }

}



// init a tree node
void C45Tree::initTreeNode(tree_node* node){
  if (DTDEBUG) cout << "initTreeNode()";

  node->id = nnodes++;
  if (DTDEBUG) cout << " id = " << node->id << endl;

  totalnodes++;
  if (totalnodes > maxnodes){
    maxnodes = totalnodes;
    if (DTDEBUG) cout << id << " C4.5 MAX nodes: " << maxnodes << endl;
  }

  // split criterion
  node->dim = -1;
  node->val = -1;
  node->type = -1;

  // current data
  node->nInstances = 0;
  node->outputs.clear();

  // next nodes in tree
  node->l = NULL;
  node->r = NULL;

  node->leaf = true;

}

void C45Tree::deleteTree(tree_node* node){
  if (DTDEBUG) cout << "deleteTree, node=" << node->id << endl;

  if (node==NULL)
    return;

  totalnodes--;

  node->nInstances = 0;
  node->outputs.clear();

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

  node->leaf = true;
  node->dim = -1;

}


C45Tree::tree_node* C45Tree::getCorrectChild(tree_node* node,
                                             const std::vector<float> &input){

  if (DTDEBUG) cout << "getCorrectChild, node=" << node->id << endl;

  if (passTest(node->dim, node->val, node->type, input))
    return node->l;
  else
    return node->r;

}

C45Tree::tree_node* C45Tree::traverseTree(tree_node* node,
                                          const std::vector<float> &input){

  if (DTDEBUG) cout << "traverseTree, node=" << node->id << endl;

  while (!node->leaf){
    node = getCorrectChild(node, input);
  }

  return node;
}


bool C45Tree::passTest(int dim, float val, bool type, const std::vector<float> &input){
  if (DTDEBUG) cout << "passTest, dim=" << dim << ",val=" << val << ",type=" << type
                    << ",input["<<dim<<"]=" << input[dim] <<endl;

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


bool C45Tree::buildTree(tree_node *node,
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

  // add each output to this node
  node->outputs.clear();
  for (unsigned i = 0; i < instances.size(); i++){
    node->outputs[instances[i]->output]++;
  }

  // see if they're all the same
  if (node->outputs.size() == 1){
    bool change = makeLeaf(node);
    if (DTDEBUG){
      cout << "All " << node->nInstances
           << " classified with output "
           << instances[0]->output << endl;
    }
    return change;
  }

  // if not, calculate gain ratio to determine best split
  else {

    if (SPLITDEBUG) cout << endl << "Creating new decision node" << endl;

    //node->leaf = false;
    //node->nInstances++;

    float bestGainRatio = -1.0;
    int bestDim = -1;
    float bestVal = -1;
    bool bestType = 0;
    std::vector<tree_experience*> bestLeft;
    std::vector<tree_experience*> bestRight;

    testPossibleSplits(instances, &bestGainRatio, &bestDim, &bestVal, &bestType, &bestLeft, &bestRight);

    return implementSplit(node, bestGainRatio, bestDim, bestVal, bestType, bestLeft, bestRight, changed);

  }

}


bool C45Tree::makeLeaf(tree_node* node){

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

  // changed from not leaf to leaf, or just init'd
  bool change = (!node->leaf || node->type < 0);

  node->leaf = true;
  node->type = 0;

  return change;
}

bool C45Tree::implementSplit(tree_node* node, float bestGainRatio, int bestDim,
                             float bestVal, bool bestType,
                             const std::vector<tree_experience*> &bestLeft, 
                             const std::vector<tree_experience*> &bestRight,
                             bool changed){
  if (DTDEBUG) cout << "implementSplit node=" << node->id << ",gainRatio=" << bestGainRatio
                    << ",dim=" << bestDim
                    << ",val=" << bestVal << ",type=" << bestType 
                    << ",chg=" << changed << endl;


  // see if this should still be a leaf node
  if (bestGainRatio < MIN_GAIN_RATIO){
    bool change = makeLeaf(node);
    if (SPLITDEBUG || STOCH_DEBUG){
      cout << "DT " << id << " Node " << node->id << " Poor gain ratio: "
	   << bestGainRatio << ", " << node->nInstances
           << " instances classified at leaf " << node->id
           << " with multiple outputs " << endl;
    }
    return change;
  }

  // see if this split changed or not
  // assuming no changes above
  if (!changed && node->dim == bestDim && node->val == bestVal
      && node->type == bestType && !node->leaf
      && node->l != NULL && node->r != NULL){
    // same split as before.
    if (DTDEBUG || SPLITDEBUG) cout << "Same split as before" << endl;
    bool changeL = false;
    bool changeR = false;

    // see which leaf changed
    if (bestLeft.size() > (unsigned)node->l->nInstances){
      // redo left side
      if (DTDEBUG) cout << "Rebuild left side of tree" << endl;
      changeL = buildTree(node->l, bestLeft, changed);
    }

    if (bestRight.size() > (unsigned)node->r->nInstances){
      // redo right side
      if (DTDEBUG) cout << "Rebuild right side of tree" << endl;
      changeR = buildTree(node->r, bestRight, changed);
    }

    // everything up to here is the same, check if there were changes below
    return (changeL || changeR);
  }

  // totally new
  // set the best split here
  node->leaf = false;
  node->dim = bestDim;
  node->val = bestVal;
  node->type = bestType;

  if (SPLITDEBUG) cout << "Best split was type " << node->type
                       << " with val " << node->val
                       << " on dim " << node->dim
                       << " with gainratio: " << bestGainRatio << endl;

  if (DTDEBUG) cout << "Left has " << bestLeft.size()
                    << ", right has " << bestRight.size() << endl;

  // make sure both instances
  if (bestLeft.size() == 0 || bestRight.size() == 0){
    cout << "ERROR: DT " << id << " node " << node->id << " has 0 instances: left: " << bestLeft.size()
         << " right: " << bestRight.size() << endl;
    cout << "Split was type " << node->type
         << " with val " << node->val
         << " on dim " << node->dim
         << " with gainratio: " << bestGainRatio << endl;
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

  // this one changed, or above changed, no reason to check change of lower parts
  return true;
  
}



void C45Tree::testPossibleSplits(const std::vector<tree_experience*> &instances, 
                                 float *bestGainRatio, int *bestDim,
                                 float *bestVal, bool *bestType, 
                                 std::vector<tree_experience*> *bestLeft,
                                 std::vector<tree_experience*> *bestRight) {
  if (DTDEBUG) cout << "testPossibleSplits" << endl;


  // pre-calculate some stuff for these splits (namely I, P, C)
  float I = calcIforSet(instances);
  //if (DTDEBUG) cout << "I: " << I << endl;

  int nties = 0;

  // for each possible split, calc gain ratio
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
      float gainRatio = calcGainRatio(idim, splitval, CUT, instances, I, left, right);

      if (SPLITDEBUG) cout << " CUT split val " << splitval
                           << " on dim: " << idim << " had gain ratio "
                           << gainRatio << endl;

      // see if this is the new best gain ratio
      compareSplits(gainRatio, idim, splitval, CUT, left, right, &nties,
                    bestGainRatio, bestDim, bestVal, bestType, bestLeft, bestRight);


      // no minval here, it would be the same as the cut split on minval
      if (ALLOW_ONLY_SPLITS && (*j) != minVal){
        // splits that are true only if this value is equal
        float splitval = (*j);

        float gainRatio = calcGainRatio(idim, splitval, ONLY, instances, I, left, right);

        if (SPLITDEBUG) cout << " ONLY split val " << splitval
                             << " on dim: " << idim << " had gain ratio "
                             << gainRatio << endl;

        // see if this is the new best gain ratio
        compareSplits(gainRatio, idim, splitval, ONLY, left, right, &nties,
                      bestGainRatio, bestDim, bestVal, bestType, bestLeft, bestRight);

      } // splits with only

    } // j loop
  }
}



void C45Tree::compareSplits(float gainRatio, int dim, float val, bool type,
                            const std::vector<tree_experience*> &left, 
                            const std::vector<tree_experience*> &right,
                            int *nties, float *bestGainRatio, int *bestDim,
                            float *bestVal, bool *bestType,
                            std::vector<tree_experience*> *bestLeft, std::vector<tree_experience*> *bestRight){
  if (DTDEBUG) cout << "compareSplits gainRatio=" << gainRatio << ",dim=" << dim
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
      if (SPLITDEBUG) cout << "   Tie on split. DT: " << id << " rand: " << randomval
                           << " splitProb: " << newsplitprob << ", selecting new split " << endl;
    }
    else
      if (SPLITDEBUG) cout << "   Tie on split. DT: " << id << " rand: " << randomval
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
    *bestLeft = left;
    *bestRight = right;
    if (SPLITDEBUG){
      cout << "  New best gain ratio: " << *bestGainRatio
           << ": type " << *bestType
           << " with val " << *bestVal
           << " on dim " << *bestDim << endl;
    }
  } // newbest
}

float C45Tree::calcGainRatio(int dim, float val, bool type,
                             const std::vector<tree_experience*> &instances,
                             float I,
                             std::vector<tree_experience*> &left,
                             std::vector<tree_experience*> &right){
  if (DTDEBUG) cout << "calcGainRatio, dim=" << dim
                    << " val=" << val
                    << " I=" << I
                    << " nInstances= " << instances.size() << endl;

  left.clear();
  right.clear();

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
  for (unsigned i = 0; i < instances.size(); i++){
    if (DTDEBUG) cout << "calcGainRatio - Classify instance " << i 
                      << " on new split " << endl;

    if (passTest(dim, val, type, instances[i]->input)){
      left.push_back(instances[i]);
    }
    else{
      right.push_back(instances[i]);
    }
  }

  if (DTDEBUG) cout << "Left has " << left.size()
                    << ", right has " << right.size() << endl;

  D[0] = (float)left.size() / (float)instances.size();
  D[1] = (float)right.size() / (float)instances.size();
  float leftInfo = calcIforSet(left);
  float rightInfo = calcIforSet(right);
  Info = D[0] * leftInfo + D[1] * rightInfo;
  Gain = I - Info;
  SplitInfo = calcIofP((float*)&D, 2);
  GainRatio = Gain / SplitInfo;

  if (DTDEBUG){
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

float C45Tree::calcIofP(float* P, int size){
  if (DTDEBUG) cout << "calcIofP, size=" << size << endl;
  float I = 0;
  for (int i = 0; i < size; i++){
    I -= P[i] * log(P[i]);
  }
  return I;
}

float C45Tree::calcIforSet(const std::vector<tree_experience*> &instances){
  if (DTDEBUG) cout << "calcIforSet" << endl;

  std::map<float, int> classes;

  // go through instances and figure count of each type
  for (unsigned i = 0; i < instances.size(); i++){
    // increment count for this value
    float val = instances[i]->output;
    classes[val]++;
  }

  // now calculate P
  float Pval;
  float I = 0;
  for (std::map<float, int>::iterator i = classes.begin(); i != classes.end(); i++){
    Pval = (float)(*i).second / (float)instances.size();
    // calc I of P
    I -= Pval * log(Pval);
  }

  return I;

}

std::set<float> C45Tree::getUniques(int dim, const std::vector<tree_experience*> &instances, float& minVal, float& maxVal){
  if (DTDEBUG) cout << "getUniques,dim = " << dim;

  std::set<float> uniques;

  for (int i = 0; i < (int)instances.size(); i++){
    if (i == 0 || instances[i]->input[dim] < minVal)
      minVal = instances[i]->input[dim];
    if (i == 0 || instances[i]->input[dim] > maxVal)
      maxVal = instances[i]->input[dim];

    uniques.insert(instances[i]->input[dim]);
  }

  if (DTDEBUG) cout << " #: " << uniques.size() << endl;
  return uniques;
}


float* C45Tree::sortOnDim(int dim, const std::vector<tree_experience*> &instances){
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


void C45Tree::printTree(tree_node *t, int level){

  for (int i = 0; i < level; i++){
    cout << ".";
  }

  cout << "Node " << t->id;
  if (t->type == C45Tree::CUT) cout << " Type: CUT"; 
  else                cout << " Type: ONLY";
  cout << " Dim: " << t->dim << " Val: " << t->val
       << " nInstances: " << t->nInstances ;
  
  if (t->leaf){
    cout << " Outputs: ";
    for (std::map<float, int>::iterator j = t->outputs.begin();
         j != t->outputs.end(); j++){
      cout << (*j).first << ": " << (*j).second << ", ";
    }
    cout << endl;
  }
  else
    cout << " Left: " << t->l->id << " Right: " << t->r->id << endl;


  // print children
  if (t->dim != -1 && !t->leaf){
    printTree(t->l, level+1);
    printTree(t->r, level+1);
  }

}





// output a map of outcomes and their probabilities for this leaf node
void C45Tree::outputProbabilities(tree_node* leaf, std::map<float, float>* retval){
  if (STOCH_DEBUG) cout << "Calculating output probs for leaf " << leaf->id << endl;

  // go through all output values at this leaf, turn into probabilities
  for (std::map<float, int>::iterator it = leaf->outputs.begin();
       it != leaf->outputs.end(); it++){

    float val = (*it).first;
    float count = (float)(*it).second;
    if (count > 0)
      (*retval)[val] = count / (float)leaf->nInstances;

    if (STOCH_DEBUG) 
      cout << "Output value " << val << " had count of " << count << " on "
           << leaf->nInstances <<" instances and prob of " 
           << (*retval)[val] << endl;
  }

}


void C45Tree::initNodes(){

  for (int i = 0; i < N_C45_NODES; i++){
    initTreeNode(&(allNodes[i]));
    freeNodes.push_back(i);
    if (NODEDEBUG) 
      cout << "init node " << i << " with id " << allNodes[i].id 
           << ", now " << freeNodes.size() << " free nodes." << endl;
  }

}

C45Tree::tree_node* C45Tree::allocateNode(){
  if (freeNodes.empty()){
    tree_node* newNode = new tree_node;
    initTreeNode(newNode);
    if (NODEDEBUG) 
      cout << id << " PROBLEM: No more pre-allocated nodes!!!" << endl
           << "return new node " << newNode->id 
           << ", now " << freeNodes.size() << " free nodes." << endl;
    return newNode;
  }

  int i = freeNodes.back();
  freeNodes.pop_back();
  if (NODEDEBUG) 
    cout << id << " allocate node " << i << " with id " << allNodes[i].id 
         << ", now " << freeNodes.size() << " free nodes." << endl;
  return &(allNodes[i]);
}

void C45Tree::deallocateNode(tree_node* node){

  if (node->id >= N_C45_NODES){
    if (NODEDEBUG) 
      cout << id << " dealloc extra node id " << node->id 
           << ", now " << freeNodes.size() << " free nodes." << endl;
    delete node;
    return;
  }

  freeNodes.push_back(node->id);
  if (NODEDEBUG) 
    cout << id << " dealloc node " << node->id 
         << ", now " << freeNodes.size() << " free nodes." << endl;
}

