#include "ExperienceFile.hh"
#include <algorithm>

//#include <time.h>
#include <sys/time.h>


ExperienceFile::ExperienceFile()
{
  expNum = 0;
}

ExperienceFile::~ExperienceFile() {
  if (vectorFile.is_open())
    vectorFile.close();
}

void ExperienceFile::initFile(const char* filename, int nfeats){
  vectorFile.open(filename, ios::out | ios::binary);

  // first part, save the vector size
  vectorFile.write((char*)&nfeats, sizeof(int));
}

void ExperienceFile::closeFile(){
  if (vectorFile.is_open())
    vectorFile.close();
}

void ExperienceFile::saveExperience(experience e){
  if (!vectorFile.is_open())
    return;

  /*
  if (expNum == 50){
    vectorFile.close();
    return;
  }
  */

  vectorFile.write((char*)&(e.s[0]), e.s.size()*sizeof(float));
  vectorFile.write((char*)&(e.next[0]), e.next.size()*sizeof(float));
  vectorFile.write((char*)&e.act, sizeof(int));
  vectorFile.write((char*)&e.reward, sizeof(float));
  vectorFile.write((char*)&e.terminal, sizeof(bool));

  //cout << "Experience " << expNum << endl;
  expNum++;
  //printExperience(e);
}


std::vector<experience> ExperienceFile::loadExperiences(const char* filename){
  ifstream inFile (filename, ios::in | ios::binary);
  
  int numFeats;
  inFile.read((char*)&numFeats, sizeof(int));

  std::vector<experience> seeds;

  // while file is not empty
  while(!inFile.eof()){
    experience e;
    e.s.resize(numFeats);
    e.next.resize(numFeats);

    inFile.read((char*)&(e.s[0]), e.s.size()*sizeof(float));
    if (inFile.eof()) break;
    inFile.read((char*)&(e.next[0]), e.next.size()*sizeof(float));
    if (inFile.eof()) break;
    inFile.read((char*)&e.act, sizeof(int));
    inFile.read((char*)&e.reward, sizeof(float));
    inFile.read((char*)&e.terminal, sizeof(bool));

    //cout << "Experience " << seeds.size() << endl;
    //printExperience(e);

    seeds.push_back(e);
  }

  inFile.close();

  return seeds;
}

void ExperienceFile::printExperience(experience e){

  cout << "State s: ";
  for(unsigned i = 0; i < e.s.size(); i++){
    cout << e.s[i] << ", ";
  }
  cout << endl << " Next: ";
  for(unsigned i = 0; i < e.next.size(); i++){
    cout << e.next[i] << ", ";
  }
  cout << endl;
  cout << "action: " << e.act << " reward: " << e.reward << endl;
  
}
