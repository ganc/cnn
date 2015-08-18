
#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
# include "cnn/expr.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>


using namespace std;
using namespace cnn;

typedef RNNPointer NetPointer;
typedef tuple<vector<int>, float, NetPointer, vector<float>> hypoth_type;
typedef unsigned int uint;

cnn::Dict sd, td; //Translation Source and Target dictionary
int skSOS; //Source Start of Sentence
int skEOS; //Source End of Sentence
int tkSOS; //Target Start of Sentence
int tkEOS; //Target End of Sentence

//These are from argv, line up with model read in
unsigned LAYERS = 0;
unsigned INPUT_DIM = 0;
unsigned HIDDEN_DIM = 0;


unsigned INPUT_VOCAB_SIZE = 0;
unsigned OUTPUT_VOCAB_SIZE = 0;


unsigned REVERSED = 0;


string seq2str(vector<int> seq, cnn::Dict d, bool no_wrapping) {
  string str;
  vector<int>::iterator it, it_end;
  if (no_wrapping) {
      it = seq.begin() + 1;
      it_end = seq.end() - 1;
  } else {
      it = seq.begin();
      it_end = seq.end();
  }

  for (; it != it_end; ++it) {
    str.append(d.Convert(*it));
  }
  return str;
}

template <class Builder>
struct EncoderDecoderForward {

  LookupParameters* p_c;
  LookupParameters* p_ec;  // map input to embedding (used in fwd and rev models)
  Parameters* p_ie2h;
  Parameters* p_bie;
  Parameters* p_h2oe;
  Parameters* p_boe;
  Parameters* p_R;
  Parameters* p_bias;
  Builder dec_builder;
  Builder rev_enc_builder;
  Builder fwd_enc_builder;

  explicit EncoderDecoderForward(Model& model) :
    dec_builder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model),
    rev_enc_builder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model),
    fwd_enc_builder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model) {


      p_ie2h = model.add_parameters({long(HIDDEN_DIM * LAYERS * 1.5), long(HIDDEN_DIM * LAYERS * 2)});
      p_bie = model.add_parameters({long(HIDDEN_DIM * LAYERS * 1.5)});
      p_h2oe = model.add_parameters({long(HIDDEN_DIM * LAYERS), long(HIDDEN_DIM * LAYERS * 1.5)});
      p_boe = model.add_parameters({long(HIDDEN_DIM * LAYERS)});
      p_ec = model.add_lookup_parameters(INPUT_VOCAB_SIZE, {INPUT_DIM}); 
      p_c = model.add_lookup_parameters(OUTPUT_VOCAB_SIZE, {INPUT_DIM}); 
      p_R = model.add_parameters({OUTPUT_VOCAB_SIZE, HIDDEN_DIM});
      p_bias = model.add_parameters({OUTPUT_VOCAB_SIZE});
    }
  
    float GetSeqScore (const vector<int>& insent, const vector<int>& osent, ComputationGraph& cg) {

    float score = 0;

    //If no reference sentence, no score calculating
    if (osent.size() == 0) 
        return 0;

    // forward encoder
    fwd_enc_builder.new_graph(cg);
    fwd_enc_builder.start_new_sequence();
    for (unsigned t = 0; t < insent.size(); ++t) {
     // if (IN_DEV)
     //     cerr << " | " << t ;
      Expression i_x_t = lookup(cg,p_ec,insent[t]);
      fwd_enc_builder.add_input(i_x_t);
    }
    // backward encoder
    rev_enc_builder.new_graph(cg);
    rev_enc_builder.start_new_sequence();
    for (int t = insent.size() - 1; t >= 0; --t) {
      Expression i_x_t = lookup(cg, p_ec, insent[t]);
      rev_enc_builder.add_input(i_x_t);
    }

    // encoder -> decoder transformation
    vector<Expression> to;
    for (auto h_l : fwd_enc_builder.final_h()) to.push_back(h_l);
    for (auto h_l : rev_enc_builder.final_h()) to.push_back(h_l);

    Expression i_combined = concatenate(to);
    Expression i_ie2h = parameter(cg, p_ie2h);
    Expression i_bie = parameter(cg, p_bie);
    Expression i_t = i_bie + i_ie2h * i_combined;
    cg.incremental_forward();
    Expression i_h = rectify(i_t);
    Expression i_h2oe = parameter(cg,p_h2oe);
    Expression i_boe = parameter(cg,p_boe);
    Expression i_nc = i_boe + i_h2oe * i_h;

    vector<Expression> oein1, oein2, oein;
    for (unsigned int i = 0; i < LAYERS; ++i) {
      oein1.push_back(pickrange(i_nc, i * HIDDEN_DIM, (i + 1) * HIDDEN_DIM));
      oein2.push_back(tanh(oein1[i]));
    }
    for (unsigned int i = 0; i < LAYERS; ++i) oein.push_back(oein1[i]);
    for (unsigned int i = 0; i < LAYERS; ++i) oein.push_back(oein2[i]);

    dec_builder.new_graph(cg);
    dec_builder.start_new_sequence(oein);

    // decoder
    Expression i_R = parameter(cg,p_R);
    Expression i_bias = parameter(cg,p_bias);

    const unsigned oslen = osent.size() - 1;
    for (unsigned t = 0; t < oslen; ++t) {
      Expression i_x_t = lookup(cg, p_c, osent[t]);
      Expression i_y_t = dec_builder.add_input(i_x_t);
      Expression i_r_t = i_bias + i_R * i_y_t;
      Expression i_ydist = log_softmax(i_r_t);
    
      vector<float> ydist = as_vector(cg.incremental_forward());
      score += ydist[osent[t+1]];
    }
    return score;
  }
};


int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);
  
  if (argc != 9) {
    cerr << "Usage: " << argv[0] << " test.txt model.params LAYERS INPUT_DIM HIDDEN_DIM INPUT_VOCAB_SIZE OUTPUT_VOCAB_SIZE REVERSED(src|||tgt = 0, tgt|||src = 1\n";
    return 1;
  }

  if (argc == 9) {
    LAYERS = stoi(argv[3]);
    INPUT_DIM = stoi(argv[4]);
    HIDDEN_DIM = stoi(argv[5]);

    INPUT_VOCAB_SIZE = stoi(argv[6]);
    OUTPUT_VOCAB_SIZE = stoi(argv[7]);

    REVERSED = stoi(argv[8]);
  }

  
  Model model;
  EncoderDecoderForward<LSTMBuilder> lm(model);

  cerr << "Reading model from " << argv[2] << "...\n";
  string fname = argv[2];
  ifstream in(fname);
  boost::archive::text_iarchive ia(in);
  ia >> model;
  ia >> sd;
  ia >> td;
  sd.Freeze(); //no new word types allowed
  td.Freeze();
  INPUT_VOCAB_SIZE = sd.size(); //set vocab sizes
  OUTPUT_VOCAB_SIZE = td.size();
 
  vector<vector<int>> test, ref;
  string line;
  int tlc = 0;
  int ttoks = 0;
  cerr << "Reading data from " << argv[1] << "...\n";
  {
    ifstream in(argv[1]);
    assert(in);
    while(getline(in, line)) {
      ++tlc;
      vector<int> test_sent, ref_sent;
      if (!REVERSED) {
        ReadSentencePair(line, &test_sent, &sd, &ref_sent, &td);
      } else {
        ReadSentencePair(line, &ref_sent, &td, &test_sent, &sd);
      }
      // cerr << "Src Sentence len: " << test_sent.size() << "tgt Sentence len: " << ref_sent.size() << "\n";
      test.push_back(test_sent);
      ref.push_back(ref_sent);

      ttoks += test.back().size();
      if ((!REVERSED && test.back().front() != skSOS && ref.back().back() != tkEOS) 
              || (REVERSED && test.back().back() != skEOS && ref.back().front() != tkSOS)) {
        cerr << "Training sentence in " << argv[1] << ":" << tlc << " didn't start or end with <s>, </s>\n";
        abort();
      }
    }
    cerr << tlc << " lines, " << ttoks << " tokens " << endl;
 
  }

  for (uint i = 0; i < test.size(); i++) {
    
      // build graph for this instance
      ComputationGraph cg;
      float score = 0;
      auto& isent = test[i];
      auto& osent = ref[i];
      if (i % 50 == 0) {
          cerr << "##"  << seq2str(isent, sd, true) << "\n" ;
          cerr << "Percent done: " << ((float)i)/((float)test.size()) << endl;
      }
      score = lm.GetSeqScore(isent, osent, cg);

      cout << seq2str(isent, sd, true) << " ||| " << seq2str(osent, td, true) << " ||| " << score << endl;
      }

  cerr << "Done." << endl;

} 
