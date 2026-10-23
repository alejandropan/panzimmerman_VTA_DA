data {
        int NS;
        int NSESS;
        int NT;
        int NSxNSESS;
        array[NSxNSESS] int NT_all;
        array[NSxNSESS] int sub_idx;
        array[NSxNSESS] int sess_idx;
        array[NS,NSESS,NT] int o;
        array[NS,NSESS,NT] int c;
}

transformed data {
        // Calculate total number of trials
        int total_trials = 0;
        for (i in 1:NSxNSESS) {
                total_trials += NT_all[i];
        }
}

parameters {
        // population-level
        real betam;
        real staym;
        real<lower=0, upper=1> alpham;
        real<lower=0, upper=1> alphastaym;
        real<lower=0, upper=1> alphaforgettingm;
        real etam_raw;
        real sidem;

        real<lower=0> betasd;
        real<lower=0> staysd;
        real<lower=0> alphasd;
        real<lower=0> alphastaysd;
        real<lower=0> alphaforgettingsd;
        real<lower=0> etasd;
        real<lower=0> sidesd;

        // mouse-level - NON-CENTERED for stay and alphas
        array[NS] real beta_mouse_raw;      // non-centered
        array[NS] real stay_mouse_raw;      // non-centered
        array[NS] real alpha_mouse_raw;     // non-centered
        array[NS] real alphastay_mouse_raw; // non-centered
        array[NS] real<lower=0, upper=1> alphaforgetting_mouse;
        array[NS] real eta_mouse_raw;
        array[NS] real side_mouse_raw;      // non-centered

        // session-level - NON-CENTERED for stay and alphastay
        real<lower=0> sidesess_sd;
        real<lower=0> alpha_ses_sd;
        real<lower=0> alphaforgetting_ses_sd;
        real<lower=0> eta_ses_sd;
        real<lower=0> alphastay_ses_sd;
        real<lower=0> stay_ses_sd;
        real<lower=0> beta_ses_sd;
        
        array[NSxNSESS] real sides_raw;          // non-centered
        array[NSxNSESS] real alpha_ses_raw;      // non-centered
        array[NSxNSESS] real<lower=0, upper=1> alphaforgetting_ses;
        array[NSxNSESS] real eta_ses_raw;
        array[NSxNSESS] real alphastay_ses_raw;  // non-centered
        array[NSxNSESS] real stay_ses_raw;       // non-centered
        array[NSxNSESS] real beta_ses_raw;       // non-centered
}

transformed parameters {
        // eta transformations (logit scale)
        real<lower=0, upper=1> etam = inv_logit(etam_raw);
        array[NS] real<lower=0, upper=1> eta_mouse;
        array[NSxNSESS] real<lower=0, upper=1> eta_ses;
        
        // mouse-level non-centered transformations
        array[NS] real beta_mouse;
        array[NS] real stay_mouse;
        array[NS] real<lower=0, upper=1> alpha_mouse;
        array[NS] real<lower=0, upper=1> alphastay_mouse;
        array[NS] real side_mouse;
        
        // session-level non-centered transformations
        array[NSxNSESS] real sides;
        array[NSxNSESS] real<lower=0, upper=1> alpha_ses;
        array[NSxNSESS] real<lower=0, upper=1> alphastay_ses;
        array[NSxNSESS] real stay_ses;
        array[NSxNSESS] real beta_ses;
        
        // Transform eta
        for (mouse in 1:NS) {
            eta_mouse[mouse] = inv_logit(eta_mouse_raw[mouse]);
        }
        for (ms_i in 1:NSxNSESS) {
            eta_ses[ms_i] = inv_logit(eta_ses_raw[ms_i]);
        }
        
        // Non-centered transformations for mouse level
        for (mouse in 1:NS) {
            beta_mouse[mouse] = betam + betasd * beta_mouse_raw[mouse];
            stay_mouse[mouse] = staym + staysd * stay_mouse_raw[mouse];
            side_mouse[mouse] = sidem + sidesd * side_mouse_raw[mouse];
            
            // For bounded parameters, transform on logit scale
            alpha_mouse[mouse] = inv_logit(logit(alpham) + alphasd * alpha_mouse_raw[mouse]);
            alphastay_mouse[mouse] = inv_logit(logit(alphastaym) + alphastaysd * alphastay_mouse_raw[mouse]);
        }
        
        // Non-centered transformations for session level
        for (ms_i in 1:NSxNSESS) {
            beta_ses[ms_i] = beta_mouse[sub_idx[ms_i]] + beta_ses_sd * beta_ses_raw[ms_i];
            stay_ses[ms_i] = stay_mouse[sub_idx[ms_i]] + stay_ses_sd * stay_ses_raw[ms_i];
            sides[ms_i] = side_mouse[sub_idx[ms_i]] + sidesess_sd * sides_raw[ms_i];
            
            // For bounded parameters, transform on logit scale
            alpha_ses[ms_i] = inv_logit(logit(alpha_mouse[sub_idx[ms_i]]) + alpha_ses_sd * alpha_ses_raw[ms_i]);
            alphastay_ses[ms_i] = inv_logit(logit(alphastay_mouse[sub_idx[ms_i]]) + alphastay_ses_sd * alphastay_ses_raw[ms_i]);
        }
}

model {
        // population priors
        alpham ~ normal(0.5, 1);
        alphastaym ~ normal(0.5, 1);
        alphaforgettingm ~ normal(0.5, 1);
        etam_raw ~ normal(-2, 1.5);
        
        betam ~ normal(0, 1);
        staym ~ normal(0, 1);
        sidem ~ normal(0, 1);

        alphasd ~ normal(0, 1);
        alphastaysd ~ normal(0, 1);
        betasd ~ normal(0, 1);
        staysd ~ normal(0, 1);
        sidesd ~ normal(0, 1);
        alphaforgettingsd ~ normal(0, 1);
        etasd ~ normal(0, 1);

        // session-level variance priors
        alpha_ses_sd ~ normal(0, 0.5);
        alphastay_ses_sd ~ normal(0, 0.5);
        beta_ses_sd ~ normal(0, 0.5);
        stay_ses_sd ~ normal(0, 0.5);
        sidesess_sd ~ normal(0, 0.5);
        alphaforgetting_ses_sd ~ normal(0, 0.5);
        eta_ses_sd ~ normal(0, 0.5);

        // mouse-level priors - NON-CENTERED (standard normal)
        for (mouse in 1:NS) {
            beta_mouse_raw[mouse] ~ normal(0, 1);
            stay_mouse_raw[mouse] ~ normal(0, 1);
            alpha_mouse_raw[mouse] ~ normal(0, 1);
            alphastay_mouse_raw[mouse] ~ normal(0, 1);
            side_mouse_raw[mouse] ~ normal(0, 1);
            
            alphaforgetting_mouse[mouse] ~ normal(alphaforgettingm, alphaforgettingsd);
            eta_mouse_raw[mouse] ~ normal(etam_raw, etasd);
        }

        // session-level priors + likelihood
        for (ms_i in 1:NSxNSESS) {
                vector[2] q;
                vector[2] qstay;
                real var_alpha;
                real delta;
                int choice;
                int nonchoice;

                // NON-CENTERED priors (standard normal)
                beta_ses_raw[ms_i] ~ normal(0, 1);
                stay_ses_raw[ms_i] ~ normal(0, 1);
                alpha_ses_raw[ms_i] ~ normal(0, 1);
                alphastay_ses_raw[ms_i] ~ normal(0, 1);
                sides_raw[ms_i] ~ normal(0, 1);
                
                alphaforgetting_ses[ms_i] ~ normal(alphaforgetting_mouse[sub_idx[ms_i]], alphaforgetting_ses_sd);
                eta_ses_raw[ms_i] ~ normal(eta_mouse_raw[sub_idx[ms_i]], eta_ses_sd);

                for (i in 1:2) {
                        q[i] = 0;
                        qstay[i] = 0;
                        var_alpha = 0.5;
                }

                for (t in 1:NT_all[ms_i]) {
                        c[sub_idx[ms_i], sess_idx[ms_i], t] ~ bernoulli_logit(
                            sides[ms_i]
                            + beta_ses[ms_i] * (q[2] - q[1])
                            + stay_ses[ms_i] * (qstay[2] - qstay[1])
                        );
                        choice = c[sub_idx[ms_i], sess_idx[ms_i], t];
                        nonchoice = !choice;
                        delta = o[sub_idx[ms_i], sess_idx[ms_i], t] - q[choice+1];
                        q[choice+1] = q[choice+1] + alpha_ses[ms_i] * var_alpha * delta;
                        q[nonchoice+1] = q[nonchoice+1] - q[nonchoice+1] * alphaforgetting_ses[ms_i];
                        qstay = qstay * (1 - alphastay_ses[ms_i]);
                        qstay[choice+1] = qstay[choice+1] + alphastay_ses[ms_i];
                        var_alpha = eta_ses[ms_i] * abs(delta) + (1 - eta_ses[ms_i]) * var_alpha;
                }
        }
}

generated quantities {
        // Use maximum possible trials (NS * NSESS * NT)
        vector[NS*NSESS*NT] log_lik;
        
        // Global trial counter
        int trial_idx = 1;
        
        // Loop through sessions (same structure as model block)
        for (ms_i in 1:NSxNSESS) {
                vector[2] q;
                vector[2] qstay;
                real var_alpha;  // Variable learning rate
                real delta;
                int choice;
                int nonchoice;
                
                // Initialize Q-values, qstay, and var_alpha
                for (i in 1:2) {
                        q[i] = 0;
                        qstay[i] = 0;
                        var_alpha = 0.5;  // Initial variable alpha
                }
                
                // Loop through trials in this session
                for (t in 1:NT_all[ms_i]) {
                        // Compute choice probability (same as model block)
                        real choice_logit = sides[ms_i] + 
                                          beta_ses[ms_i] * (q[2] - q[1]) + 
                                          stay_ses[ms_i] * (qstay[2] - qstay[1]);
                        
                        // Get actual choice
                        choice = c[sub_idx[ms_i], sess_idx[ms_i], t];
                        nonchoice = !choice;
                        
                        // Store log-likelihood for this trial
                        log_lik[trial_idx] = bernoulli_logit_lpmf(choice | choice_logit);
                        
                        // Compute prediction error
                        delta = o[sub_idx[ms_i], sess_idx[ms_i], t] - q[choice+1];
                        
                        // Update Q-values with VARIABLE learning rate (KEY DIFFERENCE from standard model)
                        q[choice+1] = q[choice+1] + alpha_ses[ms_i] * var_alpha * delta;
                        q[nonchoice+1] = q[nonchoice+1] - q[nonchoice+1] * alphaforgetting_ses[ms_i];
                        
                        // Update qstay
                        qstay = qstay * (1 - alphastay_ses[ms_i]);
                        qstay[choice+1] = qstay[choice+1] + alphastay_ses[ms_i];
                        
                        // Update variable learning rate for NEXT trial
                        var_alpha = eta_ses[ms_i] * abs(delta) + (1 - eta_ses[ms_i]) * var_alpha;
                        
                        // Move to next trial
                        trial_idx += 1;
                }
        }
}
