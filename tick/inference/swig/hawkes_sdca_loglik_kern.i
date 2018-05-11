
// License: BSD 3 clause


%include std_shared_ptr.i
%shared_ptr(HawkesSDCALoglikKern);

%{
#include "hawkes_sdca_loglik_kern.h"
%}

class HawkesSDCALoglikKern : public ModelHawkesList {
 public:
  HawkesSDCALoglikKern(ArrayDouble &decay, double l_l2sq,
                         int max_n_threads = 1, double tol = 0.,
                         int seed = -1, RandType rand_type = RandType::unif);

  HawkesSDCALoglikKern(double decay, double l_l2sq,
                       int max_n_threads = 1, double tol = 0.,
                       int seed = -1, RandType rand_type = RandType::unif);

  void compute_weights();

  void solve();

  SArrayDoublePtr get_decays() const;
  void set_decays(const ArrayDouble &decays);

  SArrayDoublePtr get_iterate();
  SArrayDoublePtr get_dual_iterate();
  double get_l_l2sq() const;
  void set_l_l2sq(const double l_l2sq);
  double get_max_dual() const;
  void set_max_dual(const double l_l2sq);

  double loss(const ArrayDouble &coeffs) override;
  double current_dual_objective();
  void set_starting_iterate(ArrayDouble & dual_iterate);
  SArrayDoublePtr get_dual_unscaled_features_init_i(ulong i);
  double get_dual_init_i_scalar_i(ulong i, ArrayDouble &kappa_i);
  SArrayDoublePtr get_dual_init();
  double dual_objective(ArrayDouble &dual);
  double dual_objective_dim_i(const ulong i, ArrayDouble &dual) const;
};
