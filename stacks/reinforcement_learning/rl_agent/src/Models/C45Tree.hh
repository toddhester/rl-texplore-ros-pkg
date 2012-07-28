/** \file C45Tree.hh
    Defines the C4.5 decision tree class.
    This is an implementation of C4.5 decision trees described in:
    J. R. Quinlan, "Induction of decision trees," Machine Learning, vol 1. pp 81-106, 1986.
    \author Todd Hester
*/


#ifndef _C45TREE_HH_
#define _C45TREE_HH_

#include <rl_common/Random.h>
#include <rl_common/core.hh>
#include <vector>
#include <set>
#include <map>

#define N_C45_EXP 200000
#define N_C45_NODES 2500

#define BUILD_EVERY 0
#define BUILD_ON_ERROR 1
#define BUILD_EVERY_N 2
#define BUILD_ON_TERMINAL 3
#define BUILD_ON_TERMINAL_AND_ERROR 4


/** C4.5 decision tree class. */
class C45Tree: public Classifier {

public:

  /** Default constructor 
      \param id id of the tree for debug
      \param trainMode build every step? only on errors? every freq steps?
      \param trainFreq frequency of model building if using latter mode
      \param m # of visits for a given state-action to be considered known
      \param featPct pct of features to remove from set used for each tree split
      \param rng Random Number Generator 
  */
  C45Tree(int id, int trainMode, int trainFreq, int m, 
	  float featPct, Random rng);

  /** Copy constructor */
  C45Tree(const C45Tree&);

  ~C45Tree();

  // structs to be defined
  struct tree_node;
  struct tree_experience;
  
  /** Make a copy of the subtree from origNode to newNode */
  void copyTree(tree_node* newNode, tree_node* origNode);

  virtual C45Tree* getCopy();

  /** Tree node struct. For decision nodes, it contains split information and pointers to child nodes. For leaf nodes, it contains all outputs that went into this leaf during trainiing. */
  struct tree_node {
    int id;

    // split criterion
    int dim;
    float val;
    bool type;

    // set of all outputs seen at this leaf/node
    std::map<float,int> outputs;
    int nInstances;

    // next nodes in tree
    tree_node *l;
    tree_node *r;

    bool leaf;
  };

  /** Experiences the tree is trained on. A vector of inputs and one float output to predict */
  struct tree_experience {
    std::vector<float> input;
    float output;
  };
  
  /** The types of splits. Split on ONLY meaning is input == x, or CUT meaning is input > x */
  enum splitTypes{
    ONLY,
    CUT
  };

  virtual bool trainInstance(classPair &instance);
  virtual bool trainInstances(std::vector<classPair> &instances);
  virtual void testInstance(const std::vector<float> &input, std::map<float, float>* retval);
  virtual float getConf(const std::vector<float> &input);

  /** Build the tree with the given instances from the given tree node */
  bool buildTree(tree_node* node, const std::vector<tree_experience*> &instances,  bool changed);

  // helper functions
  /** Initialize the tree */
  void initTree();

  /** Rebuild the tree */
  bool rebuildTree();

  /** Initialize the tree_node struct */
  void initTreeNode(tree_node* node);

  /** Traverse the tree to a leaf for the given input */
  tree_node* traverseTree(tree_node* node, const std::vector<float> &input);

  /** Get the correct child of this node for a given input */
  tree_node* getCorrectChild(tree_node* node, const std::vector<float> &input);

  /** Determine if the input passes the test defined by dim, val, type */
  bool passTest(int dim, float val, bool type, const std::vector<float> &input);

  /** Calculate the gain ratio for the given split of instances */
  float calcGainRatio(int dim, float val, bool type,
		       const std::vector<tree_experience*> &instances, float I,
		       std::vector<tree_experience*> &left,
		       std::vector<tree_experience*> &right);

  /** Returns an array of the values of features at the index dim, sorted from lowest to highest */
  float* sortOnDim(int dim, const std::vector<tree_experience*> &instances);

  /** Get all the unique values of the features on dimension dim */
  std::set<float> getUniques(int dim, const std::vector<tree_experience*> &instances, float & minVal, float& maxVal);

  /** Delete this tree node and all nodes below it in the tree. */
  void deleteTree(tree_node* node);

  /** Calculate I(P) */
  float calcIofP(float* P, int size);

  /**  Calculate I(P) for set. */
  float calcIforSet(const std::vector<tree_experience*> &instances);

  /** Print the tree for debug purposes. */
  void printTree(tree_node *t, int level);

  /** Test the possible splits for the given set of instances */
  void testPossibleSplits(const std::vector<tree_experience*> &instances, float *bestGainRatio, int *bestDim, 
                          float *bestVal, bool *bestType,
                          std::vector<tree_experience*> *bestLeft, std::vector<tree_experience*> *bestRight);

  /** Implement the given split at the given node */
  bool implementSplit(tree_node* node, float bestGainRatio, int bestDim,
                      float bestVal, bool bestType, 
                      const std::vector<tree_experience*> &left, 
                      const std::vector<tree_experience*> &right, bool changed);

  /** Compare the current split to determine if it is the best split. */
  void compareSplits(float gainRatio, int dim, float val, bool type, 
                     const std::vector<tree_experience*> &left, 
                     const std::vector<tree_experience*> &right,
                     int *nties, float *bestGainRatio, int *bestDim, 
                     float *bestVal, bool *bestType,
                     std::vector<tree_experience*> *bestLeft, 
                     std::vector<tree_experience*> *bestRight);

  /** Get the probability distribution for the given leaf node. */
  void outputProbabilities(tree_node *t, std::map<float, float>* retval);

  /** Make the given node into a leaf node. */
  bool makeLeaf(tree_node* node);

  /** Allocate a new node from our pre-allocated store of tree nodes */
  tree_node* allocateNode();

  /** Return tree node back to store of nodes */
  void deallocateNode(tree_node* node);

  /** Initialize our store of tree nodes */
  void initNodes();


  bool INCDEBUG;
  bool DTDEBUG;
  bool SPLITDEBUG;
  bool STOCH_DEBUG;
  int nExperiences;
  bool NODEDEBUG;
  bool COPYDEBUG;

  float SPLIT_MARGIN;
  float MIN_GAIN_RATIO; 

private:

  const int id;
  
  const int mode;
  const int freq;
  const int M;
  const float featPct; 
  const bool ALLOW_ONLY_SPLITS;

  Random rng;

  int nOutput;
  int nnodes;
  bool hadError;
  int maxnodes;
  int totalnodes;

  /** Vector of all experiences used to train the tree */
  std::vector<tree_experience*> experiences;

  /** Pre-allocated array of experiences to be filled during training. */
  tree_experience allExp[N_C45_EXP];

  /** Pre-allocated array of tree nodes to be used for tree */
  tree_node allNodes[N_C45_NODES];
  std::vector<int> freeNodes;

  // TREE
  /** Pointer to root node of tree. */
  tree_node* root;
  /** Pointer to last node of tree used (leaf used in last prediction made). */
  tree_node* lastNode;

};


#endif
  
