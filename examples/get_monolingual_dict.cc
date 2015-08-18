#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace cnn;

unsigned int VOCAB_SIZE = 0;

cnn::Dict d;

int kSOS;
int kEOS;



int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);

  if (argc != 2) {
    cerr << "argc: " << argc << endl;
    cerr << "Usage: " << argv[0] << " corpus.txt" << endl;
  }



  kSOS = d.Convert("<s>");
  kEOS = d.Convert("</s>");
  vector<vector<int>> training, dev;
  string line;
  int tlc = 0;
  int ttoks = 0;
  cerr << "Reading training data from " << argv[1] << "...\n";
  {
    ifstream in(argv[1]);
    assert(in);
    while(getline(in, line)) {
      ++tlc;
      training.push_back(ReadSentence(line, &d));
      ttoks += training.back().size();
      if (training.back().front() != kSOS && training.back().back() != kEOS) {
        cerr << "Training sentence in " << argv[1] << ":" << tlc << " didn't start or end with <s>, </s>\n";
        abort();
      }
    }
    cerr << tlc << " lines, " << ttoks << " tokens, " << d.size() << " types\n";
  }
  d.Freeze(); // no new word types allowed
  VOCAB_SIZE = d.size();


  ostringstream os;
  os << "dict"
    << "_" << argv[1]
    << "_" << VOCAB_SIZE
    << ".params";
  const string fname = os.str();
  cerr << "Parameters will be written to: " << fname << endl;

  ofstream out(fname);
  boost::archive::text_oarchive oa(out);
  oa << d;

}
