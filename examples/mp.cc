#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/gpu-ops.h"
#include "cnn/expr.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/algorithm/string.hpp>

#include <sys/types.h>
#include <sys/wait.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <sstream>
#include <random>

using namespace std;
using namespace cnn;
using namespace cnn::expr;

const unsigned num_children = 10;
struct Workload {
  pid_t pid;
  unsigned start;
  unsigned end;
  int pipe[2];
};

vector<pair<cnn::real, cnn::real>> ReadData(string filename) {
  vector<pair<cnn::real, cnn::real>> data;
  ifstream fs(filename);
  string line;
  while (getline(fs, line)) {
    vector<string> parts;
    boost::split(parts, line, boost::is_any_of("\t"));
    data.push_back(make_pair(atof(parts[0].c_str()), atof(parts[1].c_str())));
  }
  return data;
}

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);

  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " data.txt" << endl;
    cerr << "Where data.txt contains tab-delimited pairs of floats." << endl;
    return 1;
  }
  vector<pair<cnn::real, cnn::real>> data = ReadData(argv[1]);

  Model model;
  AdamTrainer sgd(&model, 0.0);

  ComputationGraph cg;

  Parameters* m_param = model.add_parameters({1, 1});
  Parameters* b_param = model.add_parameters({1});
  Expression m = parameter(cg, m_param);
  Expression b = parameter(cg, b_param);

  cnn::real x_value, y_value;
  Expression x = input(cg, &x_value);
  Expression y_star = input(cg, &y_value);
  Expression y = m * x + b;
  Expression loss = squared_distance(y, y_star);

  // train the parameters
  for (unsigned iter = 0; true; ++iter) {
    random_shuffle(data.begin(), data.end());
    Workload workloads[num_children];
    pid_t pid;
    unsigned cid = 0;
    for (cid = 0; cid < num_children; cid++) {
      unsigned start = (unsigned)(1.0 * cid / num_children * data.size() + 0.5);
      unsigned end = (unsigned)(1.0 * (cid + 1) / num_children * data.size() + 0.5);
      workloads[cid].start = start;
      workloads[cid].end = end;
      pipe(workloads[cid].pipe);
      pid = fork();
      if (pid == 0 || pid == -1) {
        break;
      }
      workloads[cid].pid = pid;
      //cerr << "work load for child " << cid << " (pid=" << workloads[cid].pid << ") is from " << workloads[cid].start << " to " << workloads[cid].end << endl;
    }

    if (pid == -1) {
      cerr << "Fork failed. Exiting ...";
      return 1;
    }
    else if (pid == 0) {
      assert (cid >= 0 && cid < num_children);
      unsigned start = workloads[cid].start;
      unsigned end = workloads[cid].end; 
      assert (start < end);
      assert (end <= data.size());

      cnn::real m_start = as_scalar(m.value());
      cnn::real b_start = as_scalar(b.value());

      // Actually do the training thing
      cnn::real loss = 0;
      for (auto it = data.begin() + start; it != data.begin() + end; ++it) {
        auto p = *it;
        x_value = get<0>(p);
        y_value = get<1>(p);
        loss += as_scalar(cg.forward());
        cg.backward();
        sgd.update(1.0);
      }
      loss /= data.size();

      cnn::real m_end = as_scalar(m.value());
      cnn::real b_end = as_scalar(b.value());
      //cerr << "pid = " << getpid() << "\t" << "iter = " << iter << "\t" << "loss = " << loss << "\tm = " << m_end << "/" << m_start << "\tb = " << b_end << "/" << b_start << endl;
      //cerr << iter << "\t" << "loss = " << loss << "\tm = " << m_end << "\tb = " << b_end << endl;

      write(workloads[cid].pipe[1], (char*)&m_end, sizeof(cnn::real));
      write(workloads[cid].pipe[1], (char*)&b_end, sizeof(cnn::real));
      write(workloads[cid].pipe[1], (char*)&loss, sizeof(cnn::real));
      return 0;
    }
    else {
      // Send the start and end indices ot the child
      char buf[256];
      vector<cnn::real> m_values;
      vector<cnn::real> b_values;
      vector<cnn::real> loss_values;
      for(unsigned cid = 0; cid < num_children; ++cid) {
        // Read the new m and b values from the child
        cnn::real m, b, loss;
        read(workloads[cid].pipe[0], (char*)&m, sizeof(cnn::real));
        read(workloads[cid].pipe[0], (char*)&b, sizeof(cnn::real));
        read(workloads[cid].pipe[0], (char*)&loss, sizeof(cnn::real));
        m_values.push_back(m);
        b_values.push_back(b);
        loss_values.push_back(loss);
        // Wait until the child dies
        wait(NULL); 
      }
      cnn::real m = 0.0;
      cnn::real b = 0.0;
      cnn::real loss = 0.0;
      for (unsigned i = 0; i < m_values.size(); ++i) {
        m += m_values[i];
        b += b_values[i];
        loss += loss_values[i];
      }
      m /= m_values.size();
      b /= b_values.size();

      // Update parameters to use the new m and b values
      TensorTools::SetElements(m_param->values, {m});
      TensorTools::SetElements(b_param->values, {b});
      sgd.update_epoch(); 
      cerr << iter << "\t" << "loss = " << loss << "\tm = " << m << "\tb = " << b << endl;
    }
  }
}

