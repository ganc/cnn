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
using namespace cnn::expr;

//parameters
unsigned LAYERS = 2;
unsigned INPUT_DIM = 64;
unsigned HIDDEN_DIM = 256;
unsigned INPUT_VOCAB_SIZE = 0;
unsigned OUTPUT_VOCAB_SIZE = 0;
unsigned REVERSED = 0;
string WHICH_TRAINER = "simpleSGD"; //fallthrough is SimpleSGD


cnn::Dict sd, td;
int skSOS;
int skEOS;
int tkSOS;
int tkEOS;

template <class Builder>
struct EncoderDecoder {
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
  explicit EncoderDecoder(Model& model) :
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

  // build graph and return Expression for total loss
  Expression BuildGraph(const vector<int>& insent, const vector<int>& osent, ComputationGraph& cg) {
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
    vector<Expression> errs;

    const unsigned oslen = osent.size() - 1;
    for (unsigned t = 0; t < oslen; ++t) {
      Expression i_x_t = lookup(cg, p_c, osent[t]);
      Expression i_y_t = dec_builder.add_input(i_x_t);
      Expression i_r_t = i_bias + i_R * i_y_t;
      Expression i_ydist = log_softmax(i_r_t);
      errs.push_back(pick(i_ydist,osent[t+1]));
    }
    Expression i_nerr = sum(errs);
    return -i_nerr;
  }
};

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);
  if (argc != 3 && argc != 4 && argc != 8 && argc != 9) {
    cerr << "argc: " << argc;
    cerr << "Usage: " << argv[0] << " corpus.txt dev.txt [LAYERS INPUT_DIM HIDDEN_DIM REVERSED?(if tgt|||src => 1)] trainer(simpleSGD, adam, momentumSGD)] or  [LAYERS INPUT_DIM HIDDEN_DIM REVERSED?(if tgt|||src => 1)] trainer(simpleSGD, adam, momentumSGD) model.params ] \n";
    return 1;
  }
  if (argc == 8 or argc == 9) {
    LAYERS = stoi(argv[3]);
    INPUT_DIM = stoi(argv[4]);
    HIDDEN_DIM = stoi(argv[5]);
    REVERSED = stoi(argv[6]);
    WHICH_TRAINER = argv[7];
  }
  time_t     start_now = time(0);
  struct tm  tstruct;
  char       buf[80];
  tstruct = *localtime(&start_now);
  strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

  cerr << "\n[time=" << string(buf) << "]\n";

  skSOS = sd.Convert("<s>");
  skEOS = sd.Convert("</s>");
  tkSOS = td.Convert("<s>");
  tkEOS = td.Convert("</s>");
  vector<vector<int>> src_training, src_dev, tgt_training, tgt_dev;
  string line;
  int tlc = 0;
  int sttoks = 0;
  int tttoks = 0;
  cerr << "Reading training data from " << argv[1] << "...\n";
  {
    ifstream in(argv[1]);
    assert(in);
    while(getline(in, line)) {
      ++tlc;
      vector<int> src_training_sent, tgt_training_sent;
      if (!REVERSED) {
        ReadSentencePair(line, &src_training_sent, &sd, &tgt_training_sent, &td);
      } else {
        ReadSentencePair(line, &tgt_training_sent, &td, &src_training_sent, &sd);
      }
      // cerr << "Src Sentence len: " << src_training_sent.size() << "tgt Sentence len: " << tgt_training_sent.size() << "\n";
      src_training.push_back(src_training_sent);
      tgt_training.push_back(tgt_training_sent);

      sttoks += src_training.back().size();
      tttoks += tgt_training.back().size();
      if ((!REVERSED && src_training.back().front() != skSOS && tgt_training.back().back() != tkEOS) 
              || (REVERSED && src_training.back().back() != skEOS && tgt_training.back().front() != tkSOS)) {
        cerr << "Training sentence in " << argv[1] << ":" << tlc << " didn't start or end with <s>, </s>\n";
        cerr << "Source " << sd.Convert("<s>") << " and " << sd.Convert("</s>")  << " Target " << td.Convert("<s>")  << " and " << td.Convert("</s>") << ". Instead is " << sd.Convert(src_training.back().front()) << " and " << sd.Convert(src_training.back().back()) << " and " << td.Convert(tgt_training.back().front()) << " and " << td.Convert(tgt_training.back().back());
        abort();
      }
    }
    cerr << tlc << " lines, " << sttoks << " src tokens, " << sd.size() << "src types\n" << tttoks << " tgt tokens, " << td.size() << "tgt types\n";
    cerr << src_training.size() << " srcsize, " << tgt_training.size() << " tgtsize\n";
  }
  sd.Freeze(); // no new word types allowed
  td.Freeze();
  INPUT_VOCAB_SIZE = sd.size();
  OUTPUT_VOCAB_SIZE = td.size();

  int dlc = 0;
  int sdtoks = 0;
  int tdtoks = 0;
  cerr << "Reading dev data from " << argv[2] << "...\n";
  {
    ifstream in(argv[2]);
    assert(in);
    while(getline(in, line)) {
      ++dlc;
      vector<int> src_dev_sent, tgt_dev_sent;
      if (!REVERSED) {
        ReadSentencePair(line, &src_dev_sent, &sd, &tgt_dev_sent, &td);
      } else {
        ReadSentencePair(line, &tgt_dev_sent, &td, &src_dev_sent, &sd);
      }
      // cerr << "Src Sentence len: " << src_dev_sent.size() << "tgt Sentence len: " << tgt_dev_sent.size() << "\n";
      src_dev.push_back(src_dev_sent);
      tgt_dev.push_back(tgt_dev_sent);
      sdtoks += src_dev.back().size();
      tdtoks += tgt_dev.back().size();
      if ((!REVERSED && src_training.back().front() != skSOS && tgt_training.back().back() != tkEOS) 
              || (REVERSED && src_training.back().back() != skEOS && tgt_training.back().front() != tkSOS)) {
        cerr << "Dev sentence in " << argv[2] << ":" << tlc << " didn't start or end with <s>, </s>\n";
        abort();
      }
    }
    cerr << dlc << " lines, " << sdtoks << " src tokens, " << tdtoks << " tgt tokens\n";
    cerr << src_dev.size() << " srcsize, " << tgt_dev.size() << " tgtsize\n";
  }

  ostringstream os;
  os << "bilm"
    << '_' << LAYERS
    << '_' << INPUT_DIM
    << '_' << HIDDEN_DIM
    << '_' << REVERSED
    << '_' << WHICH_TRAINER 
    << '_' << INPUT_VOCAB_SIZE
    << '_' << OUTPUT_VOCAB_SIZE
    << '_' << argv[1]
    << '_' << argv[2]
    << "-pid" << getpid() << ".params";
  const string fname = os.str();
  cerr << "Parameters will be written to: " << fname << endl;
  double best = 9e+99;

  Model model;
  Trainer* sgd = nullptr;

  if (WHICH_TRAINER == "momentumSGD")
    sgd = new MomentumSGDTrainer(&model);
  else if (WHICH_TRAINER == "adam")
    sgd = new AdamTrainer(&model);
  else if (WHICH_TRAINER == "adadelta")
    sgd = new AdadeltaTrainer(&model);
  else if (WHICH_TRAINER == "adagrad")
    sgd = new AdagradTrainer(&model);
  else if (WHICH_TRAINER == "simpleSGD")
    sgd = new SimpleSGDTrainer(&model);
  else { 
    cerr << "invalid trainer";
    exit(1);
  }


  //RNNBuilder rnn(LAYERS, INPUT_DIM, HIDDEN_DIM, &model);
  //EncoderDecoder<SimpleRNNBuilder> lm(model);
  EncoderDecoder<LSTMBuilder> lm(model);
  if (argc == 9) {
    string fname = argv[8];
    ifstream in(fname);
    boost::archive::text_iarchive ia(in);
    ia >> model;
    ia >> sd;
    ia >> td;
  }

  unsigned report_every_i = 250;
  unsigned dev_every_i_reports = 100;
  unsigned si = src_training.size();
  vector<unsigned> order(src_training.size());
  for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
  bool first = true;
  int report = 0;
  unsigned lines = 0;
  while(1) {
    Timer iteration("completed in");
    double loss = 0;
    unsigned chars = 0;
    for (unsigned i = 0; i < report_every_i; ++i) {
      if (si == src_training.size()) {
        si = 0;
        if (first) { first = false; } else { sgd->update_epoch(); }
        cerr << "**SHUFFLE\n";
        random_shuffle(order.begin(), order.end());
          double dloss = 0;
          int dchars = 0;
          for (uint i=0 ; i < src_dev.size(); i++) {
            auto& isent = src_dev[i];
            auto& osent = tgt_dev[i];
            ComputationGraph cg;
            lm.BuildGraph(isent, osent, cg);
            dloss += as_scalar(cg.forward());
            dchars += isent.size() - 1;
          }
          if (dloss < best) {
            best = dloss;
            ofstream out(fname);
            boost::archive::text_oarchive oa(out);
            oa << model;
            oa << sd;
            oa << td;
          }
          time_t     now = time(0);
          struct tm  tstruct;
          char       buf[80];
          tstruct = *localtime(&now);
          strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

          cerr << "\n***DEV [epoch=" << (lines / (double)src_training.size()) 
               << " time=" << string(buf)
               << "] E=" << (dloss / dchars) << " ppl=" << exp(dloss / dchars) << endl;

      }

      // build graph for this instance
      ComputationGraph cg;
      auto& isent = src_training[order[si]];
      auto& osent = tgt_training[order[si]];
      chars += isent.size() - 1;
      ++si;
      lm.BuildGraph(isent, osent, cg);
      //cg.PrintGraphviz();
      loss += as_scalar(cg.forward());
      cg.backward();
      sgd->update();
      ++lines;
    }
    sgd->status();
    cerr << "epoch=" << (lines / (double)src_training.size()) << " E=" << (loss / chars) << " ppl=" << exp(loss / chars) << ' ';

#if 0
    lm.RandomSample();
#endif

    // show score on dev data?
    report++;
    if (report % dev_every_i_reports == 0) {
      double dloss = 0;
      int dchars = 0;
      for (uint i=0 ; i < src_dev.size(); i++) {
        auto& isent = src_dev[i];
        auto& osent = tgt_dev[i];
        ComputationGraph cg;
        lm.BuildGraph(isent, osent, cg);
        dloss += as_scalar(cg.forward());
        dchars += isent.size() - 1;
      }
      if (dloss < best) {
        best = dloss;
        ofstream out(fname);
        boost::archive::text_oarchive oa(out);
        oa << model;
        oa << sd;
        oa << td;
      }
      time_t     now = time(0);
      struct tm  tstruct;
      char       buf[80];
      tstruct = *localtime(&now);
      strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

      cerr << "\n***DEV [epoch=" << (lines / (double)src_training.size()) 
           << " time=" << string(buf)
           << "] E=" << (dloss / dchars) << " ppl=" << exp(dloss / dchars) << ' ';

    }
  }
  delete sgd;
}
