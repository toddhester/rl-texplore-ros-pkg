#ifndef _EXPFILE_HH_
#define _EXPFILE_HH_

#include "../Common/Random.h"
#include "../Common/core.hh"

#include <vector>


class ExperienceFile {
public:
  /** Standard constructor
      */
  ExperienceFile();
  
  ~ExperienceFile();

  void initFile(const char* filename, int nfeats);
  void saveExperience(experience e);
  void printExperience(experience e);
  std::vector<experience> loadExperiences(const char* filename);
  void closeFile();

  ofstream vectorFile;
  int expNum;

};

#endif
