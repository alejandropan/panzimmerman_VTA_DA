/*
  Sequential Actor-Critic Model (dual REINFORCE) — Stan implementation
  =====================================================================
  Non-centered parameterization at both mouse and session levels.
  bias_lick removed: absorbed into pi_lick_init, which already captures
  baseline lick log-odds. The two were near-collinear (both shift the
  lick logistic intercept), causing identifiability problems.

  KEY CHANGE from centered version:
    Instead of:  mouse_param ~ normal(pop_mean, pop_sd)
    We now use:  mouse_z ~ normal(0, 1)
                 mouse_param = pop_mean + pop_sd * mouse_z

    And instead of: ses_param ~ normal(mouse_param, ses_sd)
    We now use:     ses_z ~ normal(0, 1)
                    ses_param = mouse_param + ses_sd * ses_z

  This eliminates Neal's funnel: the SD parameters are now decoupled
  from the location parameters in the geometry HMC explores.

  ADDITION 1: V_choice_init — hierarchical initial value for the choice
  critic, mirroring V_cs_init (population → mouse → session).

  ADDITION 2: V_cs_init and V_choice_init are now bounded to [0, 1].
  The hierarchy lives on the logit scale (so the Gaussian non-centered
  parameterization is preserved), and the actual init values used in the
  forward pass are obtained via inv_logit. This prevents V_cs > 1 (which
  is impossible given the laser delivers a value of 1) and prevents
  V_choice from inheriting an out-of-bounds initialization. All
  parameters whose name ends in `_l` live on the logit scale; their
  unprefixed counterparts in transformed parameters are in [0, 1].
*/

data {
  int NS;
  int NSESS;
  int NT;

  int NSxNSESS;
  array[NSxNSESS] int NT_all;
  array[NSxNSESS] int sub_idx;
  array[NSxNSESS] int sess_idx;

  array[NS, NSESS, NT] int o;
  array[NS, NSESS, NT] int c;
  array[NS, NSESS, NT] int l;
}

parameters {

  // ------------------------------------------------------------------ //
  //  Population means (alphas on logit scale; rest unbounded)           //
  //  V_cs_init / V_choice_init are now LOGIT-SCALE (suffix _l).         //
  // ------------------------------------------------------------------ //
  real alpha_v_cs_m;
  real alpha_v_choice_m;
  real alpha_loss_m;
  real alpha_lick_m;
  real V_cs_init_m_l;
  real V_choice_init_m_l;

  real beta_win_m;
  real beta_loss_m;
  real beta_lick_m;
  real pi_lick_init_m;
  real side_m;

  // ------------------------------------------------------------------ //
  //  Population SDs                                                      //
  // ------------------------------------------------------------------ //
  real<lower=0> alpha_v_cs_sd;
  real<lower=0> alpha_v_choice_sd;
  real<lower=0> alpha_loss_sd;
  real<lower=0> alpha_lick_sd;
  real<lower=0> V_cs_init_sd;       // SD on logit scale
  real<lower=0> V_choice_init_sd;   // SD on logit scale

  real<lower=0> beta_win_sd;
  real<lower=0> beta_loss_sd;
  real<lower=0> beta_lick_sd;
  real<lower=0> pi_lick_init_sd;
  real<lower=0> side_sd;

  // ------------------------------------------------------------------ //
  //  Mouse-level Z-scores (non-centered)                                 //
  // ------------------------------------------------------------------ //
  array[NS] real alpha_v_cs_mouse_z;
  array[NS] real alpha_v_choice_mouse_z;
  array[NS] real alpha_loss_mouse_z;
  array[NS] real alpha_lick_mouse_z;
  array[NS] real V_cs_init_mouse_z;
  array[NS] real V_choice_init_mouse_z;

  array[NS] real beta_win_mouse_z;
  array[NS] real beta_loss_mouse_z;
  array[NS] real beta_lick_mouse_z;
  array[NS] real pi_lick_init_mouse_z;
  array[NS] real side_mouse_z;

  // ------------------------------------------------------------------ //
  //  Session-level SDs (choice-side + lick offsets only)                 //
  // ------------------------------------------------------------------ //
  real<lower=0> alpha_loss_ses_sd;
  real<lower=0> V_cs_init_ses_sd;       // SD on logit scale
  real<lower=0> V_choice_init_ses_sd;   // SD on logit scale

  real<lower=0> beta_win_ses_sd;
  real<lower=0> beta_loss_ses_sd;

  real<lower=0> pi_lick_init_ses_sd;
  real<lower=0> side_ses_sd;

  // ------------------------------------------------------------------ //
  //  Session-level Z-scores (non-centered)                               //
  // ------------------------------------------------------------------ //
  array[NSxNSESS] real alpha_loss_ses_z;
  array[NSxNSESS] real V_cs_init_ses_z;
  array[NSxNSESS] real V_choice_init_ses_z;

  array[NSxNSESS] real beta_win_ses_z;
  array[NSxNSESS] real beta_loss_ses_z;

  array[NSxNSESS] real pi_lick_init_ses_z;
  array[NSxNSESS] real side_ses_z;
}

transformed parameters {

  // ------------------------------------------------------------------ //
  //  Reconstruct mouse-level parameters from Z-scores                   //
  //  V_cs_init and V_choice_init: build on logit scale, inv_logit at    //
  //  the end. The `_l` versions (logit scale) are kept around so the    //
  //  session-level hierarchy can also live on the logit scale, which    //
  //  is what makes the Gaussian non-centered parameterization correct.  //
  // ------------------------------------------------------------------ //
  array[NS] real alpha_v_cs_mouse;
  array[NS] real alpha_v_choice_mouse;
  array[NS] real alpha_loss_mouse;
  array[NS] real alpha_lick_mouse;
  array[NS] real V_cs_init_mouse_l;       // logit scale (used by session level)
  array[NS] real<lower=0, upper=1> V_cs_init_mouse;       // [0,1] (for monitoring)
  array[NS] real V_choice_init_mouse_l;   // logit scale
  array[NS] real<lower=0, upper=1> V_choice_init_mouse;   // [0,1]

  array[NS] real beta_win_mouse;
  array[NS] real beta_loss_mouse;
  array[NS] real beta_lick_mouse;
  array[NS] real pi_lick_init_mouse;
  array[NS] real side_mouse;

  for (mouse in 1:NS) {
    alpha_v_cs_mouse[mouse]     = alpha_v_cs_m     + alpha_v_cs_sd     * alpha_v_cs_mouse_z[mouse];
    alpha_v_choice_mouse[mouse] = alpha_v_choice_m + alpha_v_choice_sd * alpha_v_choice_mouse_z[mouse];
    alpha_loss_mouse[mouse]     = alpha_loss_m     + alpha_loss_sd     * alpha_loss_mouse_z[mouse];
    alpha_lick_mouse[mouse]     = alpha_lick_m     + alpha_lick_sd     * alpha_lick_mouse_z[mouse];

    V_cs_init_mouse_l[mouse]      = V_cs_init_m_l     + V_cs_init_sd     * V_cs_init_mouse_z[mouse];
    V_cs_init_mouse[mouse]        = inv_logit(V_cs_init_mouse_l[mouse]);
    V_choice_init_mouse_l[mouse]  = V_choice_init_m_l + V_choice_init_sd * V_choice_init_mouse_z[mouse];
    V_choice_init_mouse[mouse]    = inv_logit(V_choice_init_mouse_l[mouse]);

    beta_win_mouse[mouse]       = beta_win_m       + beta_win_sd       * beta_win_mouse_z[mouse];
    beta_loss_mouse[mouse]      = beta_loss_m      + beta_loss_sd      * beta_loss_mouse_z[mouse];
    beta_lick_mouse[mouse]      = beta_lick_m      + beta_lick_sd      * beta_lick_mouse_z[mouse];
    pi_lick_init_mouse[mouse]   = pi_lick_init_m   + pi_lick_init_sd   * pi_lick_init_mouse_z[mouse];
    side_mouse[mouse]           = side_m           + side_sd           * side_mouse_z[mouse];
  }

  // ------------------------------------------------------------------ //
  //  Reconstruct session-level parameters from Z-scores                 //
  // ------------------------------------------------------------------ //
  array[NSxNSESS] real alpha_loss_ses;
  array[NSxNSESS] real<lower=0, upper=1> V_cs_init_ses;       // bounded
  array[NSxNSESS] real<lower=0, upper=1> V_choice_init_ses;   // bounded

  array[NSxNSESS] real beta_win_ses;
  array[NSxNSESS] real beta_loss_ses;

  array[NSxNSESS] real pi_lick_init_ses;
  array[NSxNSESS] real side_ses;

  for (ms_i in 1:NSxNSESS) {
    alpha_loss_ses[ms_i]     = alpha_loss_mouse[sub_idx[ms_i]]     + alpha_loss_ses_sd     * alpha_loss_ses_z[ms_i];

    V_cs_init_ses[ms_i] = inv_logit(
      V_cs_init_mouse_l[sub_idx[ms_i]] + V_cs_init_ses_sd * V_cs_init_ses_z[ms_i]
    );
    V_choice_init_ses[ms_i] = inv_logit(
      V_choice_init_mouse_l[sub_idx[ms_i]] + V_choice_init_ses_sd * V_choice_init_ses_z[ms_i]
    );

    beta_win_ses[ms_i]       = beta_win_mouse[sub_idx[ms_i]]       + beta_win_ses_sd       * beta_win_ses_z[ms_i];
    beta_loss_ses[ms_i]      = beta_loss_mouse[sub_idx[ms_i]]      + beta_loss_ses_sd      * beta_loss_ses_z[ms_i];

    pi_lick_init_ses[ms_i]   = pi_lick_init_mouse[sub_idx[ms_i]]   + pi_lick_init_ses_sd   * pi_lick_init_ses_z[ms_i];
    side_ses[ms_i]           = side_mouse[sub_idx[ms_i]]           + side_ses_sd           * side_ses_z[ms_i];
  }
}

model {

  // ------------------------------------------------------------------ //
  //  Priors — population means (unchanged)                              //
  //  V_cs_init_m_l ~ normal(0, 1.5) on logit scale corresponds to a     //
  //  prior on V_cs_init in [0,1] that is centered at 0.5 with most      //
  //  mass spread across the unit interval (logit ±2 ~ [0.12, 0.88]).    //
  // ------------------------------------------------------------------ //
  alpha_v_cs_m       ~ normal(-3.0, 0.5);
  alpha_v_choice_m   ~ normal(-3.0, 0.5);
  alpha_loss_m       ~ normal(-2.0, 0.5);
  alpha_lick_m       ~ normal(-3.0, 0.5);
  V_cs_init_m_l      ~ normal(0, 1.5);
  V_choice_init_m_l  ~ normal(0, 1.5);

  beta_win_m       ~ normal(0, 2);
  beta_loss_m      ~ normal(0, 2);
  beta_lick_m      ~ normal(0, 2);

  pi_lick_init_m   ~ normal(0, 1);
  side_m           ~ normal(0, 1);

  // ------------------------------------------------------------------ //
  //  Priors — population SDs (unchanged)                                //
  // ------------------------------------------------------------------ //
  alpha_v_cs_sd     ~ normal(0, 0.3);
  alpha_v_choice_sd ~ normal(0, 0.3);
  alpha_loss_sd     ~ normal(0, 0.3);
  alpha_lick_sd     ~ normal(0, 0.3);
  V_cs_init_sd      ~ normal(0, 0.5);   // logit-scale SD
  V_choice_init_sd  ~ normal(0, 0.5);   // logit-scale SD

  beta_win_sd       ~ normal(0, 0.5);
  beta_loss_sd      ~ normal(0, 0.5);
  beta_lick_sd      ~ normal(0, 0.5);

  pi_lick_init_sd   ~ normal(0, 0.5);
  side_sd           ~ normal(0, 0.5);

  // ------------------------------------------------------------------ //
  //  Priors — session-level SDs (unchanged)                             //
  // ------------------------------------------------------------------ //
  alpha_loss_ses_sd     ~ normal(0, 0.3);
  V_cs_init_ses_sd      ~ normal(0, 0.5);   // logit-scale SD
  V_choice_init_ses_sd  ~ normal(0, 0.5);   // logit-scale SD

  beta_win_ses_sd       ~ normal(0, 0.5);
  beta_loss_ses_sd      ~ normal(0, 0.5);

  pi_lick_init_ses_sd   ~ normal(0, 1);
  side_ses_sd           ~ normal(0, 0.5);

  // ------------------------------------------------------------------ //
  //  Z-score priors — mouse level (replaces centered mouse priors)      //
  // ------------------------------------------------------------------ //
  alpha_v_cs_mouse_z     ~ normal(0, 1);
  alpha_v_choice_mouse_z ~ normal(0, 1);
  alpha_loss_mouse_z     ~ normal(0, 1);
  alpha_lick_mouse_z     ~ normal(0, 1);
  V_cs_init_mouse_z      ~ normal(0, 1);
  V_choice_init_mouse_z  ~ normal(0, 1);

  beta_win_mouse_z       ~ normal(0, 1);
  beta_loss_mouse_z      ~ normal(0, 1);
  beta_lick_mouse_z      ~ normal(0, 1);
  pi_lick_init_mouse_z   ~ normal(0, 1);
  side_mouse_z           ~ normal(0, 1);

  // ------------------------------------------------------------------ //
  //  Z-score priors — session level (replaces centered session priors)  //
  // ------------------------------------------------------------------ //
  alpha_loss_ses_z     ~ normal(0, 1);
  V_cs_init_ses_z      ~ normal(0, 1);
  V_choice_init_ses_z  ~ normal(0, 1);

  beta_win_ses_z       ~ normal(0, 1);
  beta_loss_ses_z      ~ normal(0, 1);

  pi_lick_init_ses_z   ~ normal(0, 1);
  side_ses_z           ~ normal(0, 1);

  // ------------------------------------------------------------------ //
  //  Likelihood (unchanged — uses reconstructed parameters from         //
  //  transformed parameters block)                                      //
  // ------------------------------------------------------------------ //
  for (ms_i in 1:NSxNSESS) {

    real V_cs;
    real V_choice;
    real pi_choice;
    real pi_lick;
    real delta_cs;
    real delta_laser;
    real p_lick;
    real p_chosen;
    int  choice;
    int  action_sign;

    real a_v_cs;
    real a_v_choice;
    real a_loss;
    real a_lick;

    a_v_cs  = inv_logit(alpha_v_cs_mouse[sub_idx[ms_i]]);
    a_lick  = inv_logit(alpha_lick_mouse[sub_idx[ms_i]]);
    a_v_choice =  inv_logit(alpha_v_choice_mouse[sub_idx[ms_i]]);

    a_loss     = inv_logit(alpha_loss_ses[ms_i]);

    V_cs      = V_cs_init_ses[ms_i];
    V_choice  = V_choice_init_ses[ms_i];
    pi_choice = 0.0;
    pi_lick   = pi_lick_init_ses[ms_i];

    for (t in 1:NT_all[ms_i]) {

      choice      = c[sub_idx[ms_i], sess_idx[ms_i], t];
      action_sign = 2 * choice - 1;

      c[sub_idx[ms_i], sess_idx[ms_i], t] ~ bernoulli_logit(
        side_ses[ms_i] + pi_choice
      );

      if (choice == 1) {
        p_chosen = inv_logit(side_ses[ms_i] + pi_choice);
      } else {
        p_chosen = 1.0 - inv_logit(side_ses[ms_i] + pi_choice);
      }

      if (o[sub_idx[ms_i], sess_idx[ms_i], t] == 1) {

        delta_cs  = V_cs - V_choice;
        V_choice += a_v_choice * delta_cs;

        pi_choice = pi_choice
                    + action_sign * beta_win_ses[ms_i] * delta_cs * (1.0 - p_chosen);

        p_lick = inv_logit(pi_lick);

        l[sub_idx[ms_i], sess_idx[ms_i], t] ~ bernoulli_logit(
          pi_lick
        );

        if (l[sub_idx[ms_i], sess_idx[ms_i], t] == 1) {
          delta_laser = 1.0 - V_cs;
          V_cs       += a_v_cs * delta_laser;
          pi_lick     = (1.0 - a_lick) * pi_lick
                        + beta_lick_mouse[sub_idx[ms_i]] * delta_laser * (1.0 - p_lick);
        } else {
          delta_laser = 0.0 - V_cs;
          V_cs       += a_v_cs * delta_laser;
          pi_lick     = (1.0 - a_lick) * pi_lick;
        }

      } else {

        delta_cs  = 0.0 - V_choice;
        V_choice += a_v_choice * delta_cs;

        pi_choice = (1.0 - a_loss) * pi_choice
                    + action_sign * beta_loss_ses[ms_i] * delta_cs * (1.0 - p_chosen);
      }
    }
  }
}

generated quantities {
  array[NSxNSESS, NT] real log_lik_c;
  array[NSxNSESS, NT] real log_lik_l;

  for (ms_i in 1:NSxNSESS) {

    real V_cs;
    real V_choice;
    real pi_choice;
    real pi_lick;
    real delta_cs;
    real delta_laser;
    real p_lick;
    real p_chosen;
    int  choice;
    int  action_sign;

    real a_v_cs;
    real a_v_choice;
    real a_loss;
    real a_lick;

    a_v_cs  = inv_logit(alpha_v_cs_mouse[sub_idx[ms_i]]);
    a_lick  = inv_logit(alpha_lick_mouse[sub_idx[ms_i]]);
    a_v_choice = inv_logit(alpha_v_choice_mouse[sub_idx[ms_i]]);
    
    a_loss     = inv_logit(alpha_loss_ses[ms_i]);

    V_cs      = V_cs_init_ses[ms_i];
    V_choice  = V_choice_init_ses[ms_i];
    pi_choice = 0.0;
    pi_lick   = pi_lick_init_ses[ms_i];

    for (t in 1:NT) {
      log_lik_c[ms_i, t] = negative_infinity();
      log_lik_l[ms_i, t] = negative_infinity();
    }

    for (t in 1:NT_all[ms_i]) {

      choice      = c[sub_idx[ms_i], sess_idx[ms_i], t];
      action_sign = 2 * choice - 1;

      log_lik_c[ms_i, t] = bernoulli_logit_lpmf(
        c[sub_idx[ms_i], sess_idx[ms_i], t] | side_ses[ms_i] + pi_choice
      );

      if (choice == 1) {
        p_chosen = inv_logit(side_ses[ms_i] + pi_choice);
      } else {
        p_chosen = 1.0 - inv_logit(side_ses[ms_i] + pi_choice);
      }

      if (o[sub_idx[ms_i], sess_idx[ms_i], t] == 1) {

        delta_cs  = V_cs - V_choice;
        V_choice += a_v_choice * delta_cs;

        pi_choice = pi_choice
                    + action_sign * beta_win_ses[ms_i] * delta_cs * (1.0 - p_chosen);

        p_lick = inv_logit(pi_lick);

        log_lik_l[ms_i, t] = bernoulli_logit_lpmf(
          l[sub_idx[ms_i], sess_idx[ms_i], t] | pi_lick
        );

        if (l[sub_idx[ms_i], sess_idx[ms_i], t] == 1) {
          delta_laser = 1.0 - V_cs;
          V_cs       += a_v_cs * delta_laser;
          pi_lick     = (1.0 - a_lick) * pi_lick
                        + beta_lick_mouse[sub_idx[ms_i]] * delta_laser * (1.0 - p_lick);
        } else {
          delta_laser = 0.0 - V_cs;
          V_cs       += a_v_cs * delta_laser;
          pi_lick     = (1.0 - a_lick) * pi_lick;
        }

      } else {

        // log_lik_l[ms_i, t] stays negative_infinity() — lick not
        // observed on no-reward trials, excluded from LOO automatically

        delta_cs  = 0.0 - V_choice;
        V_choice += a_v_choice * delta_cs;

        pi_choice = (1.0 - a_loss) * pi_choice
                    + action_sign * beta_loss_ses[ms_i] * delta_cs * (1.0 - p_chosen);
      }
    }
  }
}
