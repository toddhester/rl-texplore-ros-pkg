#include "LinearSplitsTree.hh"

// LinearSplitsTree, from the following sources:

// Include stuff for newmat matrix libraries

#define WANT_MATH                    // include.h will get math fns
                                     // newmatap.h will get include.h
#include "../newmat/newmatap.h"      // need matrix applications
#ifdef use_namespace
using namespace NEWMAT;              // access NEWMAT namespace
#endif

// TODO:
//  - save regression from split testing rather than re-building it later


LinearSplitsTree::LinearSplitsTree(int id, int trainMode, int trainFreq, int m,
                                   float featPct, bool simple, float min_er,
                                   Random rng):
  id(id), mode(trainMode), 
  freq(trainFreq), M(m),
  featPct(featPct), SIMPLE(simple), 
  MIN_ER(min_er), rng(rng)
{

  nnodes = 0;
  nOutput = 0;
  nExperiences = 0;
  hadError = false;
  maxnodes = N_LS_NODES;
  totalnodes = 0;

  // how close a split has to be to be randomly selected
  SPLIT_MARGIN = 0.0; //0.02; //5; //01; //0.05; //0.2; //0.05;

  LMDEBUG = false;// true;
  DTDEBUG = false;//true;
  SPLITDEBUG = false; //true;
  STOCH_DEBUG = false; //true; //false; //true;
  INCDEBUG = false; //true; //false; //true;
  NODEDEBUG = false;
  COPYDEBUG = false; //true;

  cout << "Created linear splits decision tree " << id;
  if (SIMPLE) cout << " simple regression";
  else cout << " multivariate regrssion";
  if (DTDEBUG){
    cout << " mode: " << mode << " freq: " << freq << endl;
  }
  cout << " MIN_ER: " << MIN_ER << endl;


  initNodes();
  initTree();

}

LinearSplitsTree::LinearSplitsTree(const LinearSplitsTree& ls):
  id(ls.id), mode(ls.mode), 
  freq(ls.freq), M(ls.M),
  featPct(ls.featPct), SIMPLE(ls.SIMPLE), 
  MIN_ER(ls.MIN_ER), rng(ls.rng)
{
  COPYDEBUG = ls.COPYDEBUG;
  if (COPYDEBUG) cout << "LS copy " << id << endl;
  nnodes = 0;
  nOutput = ls.nOutput;
  nExperiences = ls.nExperiences;
  hadError = ls.hadError;
  totalnodes = 0;
  maxnodes = ls.maxnodes;
  SPLIT_MARGIN = ls.SPLIT_MARGIN; 
  LMDEBUG = ls.LMDEBUG;
  DTDEBUG = ls.DTDEBUG;
  SPLITDEBUG = ls.SPLITDEBUG;
  STOCH_DEBUG = ls.STOCH_DEBUG; 
  INCDEBUG = ls.INCDEBUG; 
  NODEDEBUG = ls.NODEDEBUG;

  if (COPYDEBUG) cout << "   LS copy nodes, experiences, root, etc" << endl;
  // copy all experiences
  for (int i = 0; i < N_LST_EXP; i++){
    allExp[i] = ls.allExp[i];
  }
  if (COPYDEBUG) cout << "   LS copied exp array" << endl;

  // set experience pointers
  experiences.resize(ls.experiences.size());
  for (unsigned i = 0; i < ls.experiences.size(); i++){
    experiences[i] = &(allExp[i]);
  }
  if (COPYDEBUG) cout << "   LS set experience pointers" << endl;

  // now the tricky part, set the pointers inside the tree nodes correctly
  initNodes();

  if (COPYDEBUG) cout << "   LS copy tree " << endl;
  root = allocateNode();
  lastNode = root;
  copyTree(root, ls.root);
  if (COPYDEBUG) cout << "   LS  tree copy done" << endl;
   
  if (COPYDEBUG) {
    cout << endl << "New tree: " << endl;
    printTree(root, 0);
    cout << endl;
    cout << "  LS copy done" << endl;
  }

}

void LinearSplitsTree::copyTree(tree_node* newNode, tree_node* origNode){

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

LinearSplitsTree* LinearSplitsTree::getCopy(){
  LinearSplitsTree* copy = new LinearSplitsTree(*this);
  return copy;
}

LinearSplitsTree::~LinearSplitsTree() {
  deleteTree(root);
  for (unsigned i = N_LST_EXP; i < experiences.size(); i++){
    delete experiences[i];
  }
  experiences.clear();
}

// here the target output will be a single value
bool LinearSplitsTree::trainInstance(classPair &instance){

  if (DTDEBUG) cout << id << " trainInstance" << endl;

  bool modelChanged = false;

  // simply add this instance to the set of experiences

  // take from static array until we run out
  tree_experience *e;
  if (nExperiences < N_LST_EXP){
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
    if (DTDEBUG || SPLITDEBUG) cout << "DT " << id << " tree re-built." << endl;

    if (DTDEBUG || SPLITDEBUG){
      cout << endl << "DT: " << id << endl;
      printTree(root, 0);
      cout << "Done printing tree" << endl;
    }
  }

  /*
  if (nExperiences % 100 == 0){
    cout << endl << "DT: " << id << endl;
    printTree(root, 0);
    cout << "Done printing tree" << endl;
    
    // look at error
    float errorSum = 0.0;
    for (int i = 0; i < nExperiences; i++){
      e = experiences[i];
      std::map<float, float> answer;
      testInstance(e->input, &answer);
      float val = answer.begin()->first;
      float error = fabs(val - e->output);
      errorSum += error;
    }
    float avgError = errorSum / (float)nExperiences;
    cout << "avgError: " << avgError << endl << endl;
  }
  */

  return modelChanged;

}


// here the target output will be a single value
bool LinearSplitsTree::trainInstances(std::vector<classPair> &instances){
  if (DTDEBUG) cout << "DT trainInstances: " << instances.size() << endl;

  bool modelChanged = false;

  bool doBuild = false;

  // loop through instances, possibly checking for errors
  for (unsigned a = 0; a < instances.size(); a++){
    classPair instance = instances[a];

    // simply add this instance to the set of experiences

    // take from static array until we run out
    tree_experience *e;
    if (nExperiences < N_LST_EXP){
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

    /*
    if (nExperiences % 100 == 0){
      cout << endl << "DT: " << id << endl;
      printTree(root, 0);
      cout << "Done printing tree" << endl;
      
      // look at error
      float errorSum = 0.0;
      for (int i = 0; i < nExperiences; i++){
        e = experiences[i];
        std::map<float, float> answer;
        testInstance(e->input, &answer);
        float val = answer.begin()->first;
        float error = fabs(val - e->output);
        errorSum += error;
      }
      float avgError = errorSum / (float)nExperiences;
      cout << "avgError: " << avgError << endl << endl;
    }
    */

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
    if (DTDEBUG || SPLITDEBUG) cout << "DT " << id << " tree re-built." << endl;

    if (DTDEBUG || SPLITDEBUG){
      cout << endl << "DT: " << id << endl;
      printTree(root, 0);
      cout << "Done printing tree" << endl;
    }
  }

  return modelChanged;

}


void LinearSplitsTree::rebuildTree(){
  //cout << "rebuild tree " << id << " on exp: " << nExperiences << endl;
  //deleteTree(root);

  // re-calculate avg error for root
  root->avgError = calcAvgErrorforSet(experiences);

  buildTree(root, experiences, false);
  //cout << "tree " << id << " rebuilt. " << endl;
}


// TODO: here we want to return the probability of the output value being each of the possible values, in the stochastic case
void LinearSplitsTree::testInstance(const std::vector<float> &input, std::map<float, float>* retval){
  if (DTDEBUG) cout << "testInstance on tree " << id << endl;

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

float LinearSplitsTree::getConf(const std::vector<float> &input){
  if (DTDEBUG) cout << "numVisits" << endl;

  // in case the tree is empty
  if (experiences.size() == 0){
    return 0;
  }

  if (lastNode == NULL)
    return 0;

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
void LinearSplitsTree::initTree(){
  if (DTDEBUG) cout << "initTree()" << endl;
  root = allocateNode();

  if (DTDEBUG) cout << "   root id = " << root->id << endl;

  // just to ensure the diff models are on different random values
  for (int i = 0; i < id; i++){
    rng.uniform(0, 1);
  }

}



// init a tree node
void LinearSplitsTree::initTreeNode(tree_node* node){
  if (DTDEBUG) cout << "initTreeNode()";

  node->id = nnodes++;
  if (DTDEBUG) cout << " id = " << node->id << endl;

  totalnodes++;
  if (totalnodes > maxnodes){
    maxnodes = totalnodes;
    if (DTDEBUG) cout << id << " LS MAX nodes: " << maxnodes << endl;
  }

  // split criterion
  node->dim = -1;
  node->val = -1;

  // current data
  node->nInstances = 0;

  // coefficients
  node->constant = 0.0;

  // coefficients will get resized later
  //  node->coefficients.resize(2,0);

  // next nodes in tree
  node->l = NULL;
  node->r = NULL;

  node->leaf = true;
  node->avgError = 10000;

}

/** delete current tree */
void LinearSplitsTree::deleteTree(tree_node* node){
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

  node->dim = -1;
  node->leaf = true;
  node->avgError = 10000;
  node->val = -1;
  node->constant = 0;
}


/** Get the correct child of this node based on the input */
LinearSplitsTree::tree_node* LinearSplitsTree::getCorrectChild(tree_node* node,
                                                               const std::vector<float> &input){

  if (DTDEBUG) cout << "getCorrectChild, node=" << node->id << endl;

  if (passTest(node->dim, node->val, input))
    return node->l;
  else
    return node->r;

}

/** Traverse the tree to the leaf for this input. */
LinearSplitsTree::tree_node* LinearSplitsTree::traverseTree(tree_node* node,
                                                            const std::vector<float> &input){

  if (DTDEBUG) cout << "traverseTree, node=" << node->id << endl;

  while (!node->leaf){
    node = getCorrectChild(node, input);
  }

  return node;
}


/** Decide if this passes the test */
bool LinearSplitsTree::passTest(int dim, float val, const std::vector<float> &input){
  if (DTDEBUG) cout << "passTest, dim=" << dim << ",val=" << val 
                    << ",input["<<dim<<"]=" << input[dim] <<endl;

  
  if (input[dim] > val)
    return false;
  else
    return true;

}


/** Build the tree from this node down using this set of experiences. */
void LinearSplitsTree::buildTree(tree_node *node,
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
    makeLeaf(node, instances);
    if (DTDEBUG){
      cout << "Tree " << id << " node " << node->id 
           << " All " << node->nInstances
           << " classified with output "
           << instances[0]->output << ", " << node->constant << endl;
    }
    return;
  }

  // check if linear model has no error
  if (node != root && node->avgError < MIN_ER){
    makeLeaf(node, instances);
    if (node->avgError < MIN_ER){
      if (DTDEBUG) {
      cout << "Tree " << id << " node " << node->id
           << " has low error " << node->avgError << " keeping as leaf" << endl;
      }
      return;
    }
  }
  
  // if not, calculate ER to determine best split
  if (SPLITDEBUG) cout << endl << "Creating new decision node " << id << "-" << node->id << endl;

  node->leaf = false;
  //node->nInstances++;

  float bestER = -1.0;
  int bestDim = -1;
  float bestVal = -1;
  std::vector<tree_experience*> bestLeft;
  std::vector<tree_experience*> bestRight;
  float leftError = 10000;
  float rightError = 10000;

  testPossibleSplits(node->avgError, instances, &bestER, &bestDim, &bestVal, &bestLeft, &bestRight, &leftError, &rightError);

  implementSplit(node, instances, bestER, bestDim, bestVal, bestLeft, bestRight, changed, leftError, rightError);

}


void LinearSplitsTree::makeLeaf(tree_node* node, const std::vector<tree_experience*> &instances){

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

  // fit linear model
  if (SIMPLE)
    node->avgError = fitSimpleLinearModel(instances, &(node->constant), &(node->coefficients));
  else 
    node->avgError = fitMultiLinearModel(instances, &(node->constant), &(node->coefficients));

  if (DTDEBUG || LMDEBUG){
    cout << "make leaf, fit linear model with constant " << node->constant 
         << "  error: " << node->avgError << endl;
  }

}


float LinearSplitsTree::fitMultiLinearModel(const std::vector<tree_experience*> &instances,
                                            float *bestConstant, 
                                            std::vector<float> *bestCoefficients){
  if(DTDEBUG || LMDEBUG) cout << "fitMultiLinearModel"
                              << ",nInstances:" << instances.size() << endl;

  // make sure there are enough coefficients for all the features
  if (bestCoefficients->size() != instances[0]->input.size()){
    bestCoefficients->resize(instances[0]->input.size(), 0);  
  }

  // in case of size 1
  if (instances.size() == 1){
    *bestConstant = instances[0]->output;
    for (unsigned i = 0; i < bestCoefficients->size(); i++){
      (*bestCoefficients)[i] = 0.0;
    }
    if (LMDEBUG || SPLITDEBUG){
      cout << "  Singleton constant: " 
           << *bestConstant << " avgE: " << 0 << endl;
    }
    return 0;
  }

  // loop through all features, try simple single variable regression
  // keep track of error/coefficient of each
  float bestError = 100000;
  float avgError = 100000;
  bool doRegression = true;
  int ntimes = 0;

  std::vector<bool> featureMask(bestCoefficients->size(), true);

  while (doRegression){
    //cout << id << " Attempt linear model " << ntimes << endl;
    ntimes++;

    int nObs = (int)instances.size();

    int nFeats = 0;
    for (unsigned i = 0; i < featureMask.size(); i++){
      if (featureMask[i]) nFeats++;
    }
    if (nFeats < 1)
      break;

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
        (*bestCoefficients)[j] = 0;
        if (!featureMask[j])
          continue;

        if (constants[j] && e->input[j] != instances[0]->input[j]){
          constants[j] = false;
        }

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

    if (foundProblem)
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

    // form sum of squares and product matrix
    //    [use << rather than = for copying Matrix into SymmetricMatrix]
    SymmetricMatrix SSQ;
    SSQ << XC.t() * XC;

    Try {

      ///////////////////////////
      // Cholesky Method
      LowerTriangularMatrix L = Cholesky(SSQ);

      // calculate estimate
      ColumnVector A = L.t().i() * (L.i() * (XC.t() * YC));
      ///////////////////////////
      
      //////////////////////////
      // Least Squares Method
      /*
      // calculate estimate
      //    [bracket last two terms to force this multiplication first]
      //    [ .i() means inverse, but inverse is not explicity calculated]
      ColumnVector A = SSQ.i() * (XC.t() * YC);
      //////////////////////////
      */
  
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
      bestError = 0;
      for (int i = 0; i < nObs; i++){
        if (DTDEBUG || LMDEBUG){
          cout << "instance " << i << " linear model predicted: " << Fitted(i+1)
               << " actual: " << instances[i]->output
               << " error: " << Residual(i+1) << endl;
        }
        bestError += fabs(Residual(i+1));
      }
      avgError = bestError / (float)nObs;

      // coeff
      for (int i = 0; i < nFeats; i++){
        if (DTDEBUG || LMDEBUG) cout << "Coeff " << i << " on feat: " << featIndices[i] << " is " << A(i+1) << endl;
        (*bestCoefficients)[featIndices[i]] = A(i+1);
      }
      if (DTDEBUG || LMDEBUG) cout << "constant is " << a << endl;
      *bestConstant = a;

    }

    CatchAll {
      // had an error trying the linear regression.
      // HACK TODO: for now, turn off one variable
      //cout << ntimes << " linear regression had exception" << endl;
      //<< BaseException::what() <<endl;

      /*
      for (int i = 0; i < nObs; i++){
        tree_experience *e = instances[i];
        cout << "Obs: " << i;
        
        // go through all features
        int featIndex = 1;
        for (unsigned j = 0; j < featureMask.size(); j++){
          node->coefficients[j] = 0;
          if (!featureMask[j])
            continue;
          
          
          cout << " Feat " << featIndex << " index " << j
               << " val " << X(i+1,featIndex) << ",";
          
          featIndex++;
        }
        
        cout << " out: " << e->output << endl;
      }
      */

      // tried this already, stop now
      if (ntimes > 2 || nFeats < 2){
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
    } // catch

    // it worked, dont need to do it again
    doRegression = false;

  }

  // return error
  return avgError;

}


float LinearSplitsTree::fitSimpleLinearModel(const std::vector<tree_experience*> &instances,
                                             float *bestConstant, std::vector<float> *bestCoefficients){
  if(DTDEBUG || LMDEBUG || SPLITDEBUG) cout << " fitSimpleLinearModel, "
                              << ",nInstances:" << instances.size() << endl;
  
  // make sure there are enough coefficients for all the features
  if (bestCoefficients->size() != instances[0]->input.size()){
    bestCoefficients->resize(instances[0]->input.size(), 0);  
  }

  // loop through all features, try simple single variable regression
  // keep track of error/coefficient of each
  int bestFeat = -1;
  float bestCoeff = -1;
  float bestError = 100000;
  *bestConstant = -1;

  // in case of size 1
  if (instances.size() == 1){
    bestFeat = 0;
    *bestConstant = instances[0]->output;
    for (unsigned i = 0; i < bestCoefficients->size(); i++){
      (*bestCoefficients)[i] = 0.0;
    }
    bestError = 0;
    if (LMDEBUG || SPLITDEBUG){
      cout << "  Singleton constant: " 
           << *bestConstant << " avgE: " << bestError << endl;
    }
    return 0;
  }

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
    float coeff = (xysum[j] - xsum[j]*ysum/nObs)/(x2sum[j]-(xsum[j]*xsum[j])/nObs);
    float constant = (ysum/nObs) - coeff*(xsum[j]/nObs);

    // so we don't get absurd coefficients that get rounded off earlier
    if (fabs(coeff) < 1e-5){
      coeff = 0.0;
    }
    if (fabs(constant) < 1e-5){
      constant = 0.0;
    }

    if (LMDEBUG) {
      cout << "Feat " << j << " coeff: " << coeff << ", constant: " 
           << constant  << endl;
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
    float avgError = errorSum / (float)nObs;
    if (LMDEBUG) cout << "avgError: " << avgError << endl;

    // check if this is the best
    if (avgError < bestError){
      bestError = avgError;
      bestFeat = j;
      *bestConstant = constant;
      bestCoeff = coeff;
    }
  }
  
  if (LMDEBUG || SPLITDEBUG){
    cout << "  LST feat: " << bestFeat << " coeff: " 
         << bestCoeff << " constant: " 
         << *bestConstant << " avgE: " << bestError << endl;
  }

  if (bestFeat < 0 || bestFeat > (int)instances[0]->input.size()){
    for (unsigned i = 0; i < bestCoefficients->size(); i++){
      (*bestCoefficients)[i] = 0.0;
    }
    *bestConstant = 0.0;
    return 100000;
  }

  // fill in coeff vector
  for (unsigned i = 0; i < bestCoefficients->size(); i++){
    if (i == (unsigned)bestFeat)
      (*bestCoefficients)[i] = bestCoeff;
    else
      (*bestCoefficients)[i] = 0.0; 
  }

  return bestError;
 
}




void LinearSplitsTree::implementSplit(tree_node* node, 
                                      const std::vector<tree_experience*> &instances,
                                      float bestER, int bestDim,
                                      float bestVal, 
                                      const std::vector<tree_experience*> &bestLeft,
                                      const std::vector<tree_experience*> &bestRight,
                                      bool changed, float leftError, float rightError){
  if (DTDEBUG) cout << "implementSplit node=" << node->id
                    << ",er=" << bestER
                    << ",dim=" << bestDim
                    << ",val=" << bestVal 
                    << ",chg=" << changed << endl;


  // see if this should still be a leaf node
  if (bestER < MIN_ER){
    makeLeaf(node, instances);

    if (SPLITDEBUG || STOCH_DEBUG){
      cout << " DT " << id << " Node " << node->id << " Poor er "
           << node->nInstances
           << " instances classified at leaf " << node->id
           << " with er " << bestER << " constant: " << node->constant << endl;
    }
    return;
  }

  //cout << id << " implement split with er " << bestER << endl;

  // see if this split changed or not
  // assuming no changes above
  if (!changed && node->dim == bestDim && node->val == bestVal
      && !node->leaf && node->l != NULL && node->r != NULL){
    // same split as before.
    if (DTDEBUG || SPLITDEBUG) cout << "Same split as before " << node->id << endl;
    node->l->avgError = leftError;
    node->r->avgError = rightError;


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
                       << " with er: " << bestER << endl;

  if (DTDEBUG) cout << "Left has " << bestLeft.size()
                    << ", right has " << bestRight.size() << endl;

  // make sure both instances
  if (bestLeft.size() == 0 || bestRight.size() == 0){
    cout << "ERROR: DT " << id << " node " << node->id << " has 0 instances: left: " << bestLeft.size()
         << " right: " << bestRight.size() << endl;
    cout << "Split was cut with val " << node->val
         << " on dim " << node->dim
         << " with er: " << bestER << endl;
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
  node->l->avgError = leftError;
  buildTree(node->l, bestLeft, true);
  if (DTDEBUG) cout << "Building right tree for node " << node->id << endl;
  node->r->avgError = rightError;
  buildTree(node->r, bestRight, true);

}


void LinearSplitsTree::testPossibleSplits(float avgError, const std::vector<tree_experience*> &instances,
                                          float *bestER, int *bestDim,
                                          float *bestVal, 
                                          std::vector<tree_experience*> *bestLeft,
                                          std::vector<tree_experience*> *bestRight,
                                          float *bestLeftError, float *bestRightError) {
  if (DTDEBUG || SPLITDEBUG) cout << "testPossibleSplits, error=" << avgError << endl;

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
      float leftError = 10000;
      float rightError = 10000;

      // splits that are cuts
      float splitval = (*j);
      float er = calcER(idim, splitval, instances, avgError, left, right, &leftError, &rightError);

      if (SPLITDEBUG){
        cout << id << " CUT split val " << splitval
             << " on dim: " << idim << " had er "
             << er << endl;
      }

      // see if this is the new best er
      compareSplits(er, idim, splitval, left, right, &nties, leftError, rightError,
                    bestER, bestDim, bestVal,bestLeft, bestRight, bestLeftError, bestRightError);


    } // j loop
  }
}



/** Decide if this split is better. */
void LinearSplitsTree::compareSplits(float er, int dim, float val, 
                                     const std::vector<tree_experience*> &left, 
                                     const std::vector<tree_experience*> &right,
                                     int *nties, float leftError, float rightError,
                                     float *bestER, int *bestDim,
                                     float *bestVal, 
                                     std::vector<tree_experience*> *bestLeft,
                                     std::vector<tree_experience*> *bestRight,
                                     float *bestLeftError, float *bestRightError){
  if (DTDEBUG) cout << "compareSplits er=" << er << ",dim=" << dim
                    << ",val=" << val  <<endl;


  bool newBest = false;

  // if its a virtual tie, break it randomly
  if (fabs(*bestER - er) < SPLIT_MARGIN){
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
  else if (er > *bestER){
    newBest = true;
    *nties = 1;
  }


  // set the split features
  if (newBest){
    *bestER = er;
    *bestDim = dim;
    *bestVal = val;
    *bestLeft = left;
    *bestRight = right;
    *bestLeftError = leftError;
    *bestRightError = rightError;
    if (SPLITDEBUG){
      cout << "  New best er: " << *bestER
           << " with val " << *bestVal
           << " on dim " << *bestDim << endl;
    }
  } // newbest
}

/** Calculate error reduction for this possible split. */
float LinearSplitsTree::calcER(int dim, float val,
                               const std::vector<tree_experience*> &instances,
                               float avgError,
                               std::vector<tree_experience*> &left,
                               std::vector<tree_experience*> &right,
                               float *leftError, float *rightError){
  if (DTDEBUG || SPLITDEBUG) cout << "calcER, dim=" << dim
                    << " val=" << val
                    << " err=" << avgError
                    << " nInstances= " << instances.size() << endl;

  left.clear();
  right.clear();

  // split into two sides
  for (unsigned i = 0; i < instances.size(); i++){
    if (DTDEBUG) cout << " calcER - Classify instance " << i << " on new split " << endl;

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
  *leftError = calcAvgErrorforSet(left);
  *rightError = calcAvgErrorforSet(right);

  float leftRatio = (float)left.size() / (float)instances.size();
  float rightRatio = (float)right.size() / (float)instances.size();
  float newError = (leftRatio * (*leftError) + rightRatio * (*rightError));

  float er = avgError - newError;

  if (DTDEBUG || SPLITDEBUG){
    cout << "LeftError: " << *leftError
         << " RightError: " << *rightError
         << " NewError: " << newError
         << " NodeError: " << avgError
         << " ER: " << er
         << endl;
  }

  return er;

}

/** Calculate std deviation for set. */
float LinearSplitsTree::calcAvgErrorforSet(const std::vector<tree_experience*> &instances){
  if (DTDEBUG) cout << "calcAvgErrorforSet" << endl;

  int n = instances.size();

  if (n == 0)
    return 0;

  // fit a linear model to instances
  // and figure out avg error of it
  float constant;
  float avgError = 0.0;
  std::vector<float> coeff;
  if (SIMPLE) 
    avgError = fitSimpleLinearModel(instances, &constant, &coeff);
  else 
    avgError = fitMultiLinearModel(instances, &constant, &coeff);

  return avgError;
}


/** Returns the unique elements at this index */
std::set<float> LinearSplitsTree::getUniques(int dim, const std::vector<tree_experience*> &instances, float& minVal, float& maxVal){
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


/** Returns a list of the attributes in this dimension sorted
    from lowest to highest. */
float* LinearSplitsTree::sortOnDim(int dim, const std::vector<tree_experience*> &instances){
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


/** Print the tree for debug purposes. */
void LinearSplitsTree::printTree(tree_node *t, int level){

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
void LinearSplitsTree::leafPrediction(tree_node* leaf, const std::vector<float> &input,
                                      std::map<float, float>* retval){
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


void LinearSplitsTree::initNodes(){

  for (int i = 0; i < N_LS_NODES; i++){
    initTreeNode(&(allNodes[i]));
    freeNodes.push_back(i);
    if (NODEDEBUG) 
      cout << "init node " << i << " with id " << allNodes[i].id 
           << ", now " << freeNodes.size() << " free nodes." << endl;
  }

}

LinearSplitsTree::tree_node* LinearSplitsTree::allocateNode(){
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

void LinearSplitsTree::deallocateNode(tree_node* node){
  if (node->id >= N_LS_NODES){
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
