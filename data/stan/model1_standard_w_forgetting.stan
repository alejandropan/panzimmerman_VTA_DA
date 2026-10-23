data {
        int NS; // number of subjects
        int NSESS; // max number of sessions
        int NT; // max number of trials

        int NSxNSESS; // flattening the subjects and sessions
        array[NSxNSESS] int NT_all; // a vector with different trial lengths per subject+per session
        array[NSxNSESS] int sub_idx;    // Mapping parameter to subject
        array[NSxNSESS] int sess_idx;    // Mapping parameter to session

        array[NS,NSESS,NT] int o;
        array[NS,NSESS,NT] int c;
}

transformed data {
        // Calculate total number of trials across all sessions
        int total_trials = 0;
        for (i in 1:NSxNSESS) {
                total_trials += NT_all[i];
        }
}

parameters {
        // normal intercept model
        real betam;
        real staym;
        real<lower=0, upper=1> alpham;
        real<lower=0, upper=1> alphastaym;
        real<lower=0, upper=1> alphaforgettingm;
        real sidem;

        real<lower=0> betasd;
        real<lower=0> staysd;
        real<lower=0> alphasd;
        real<lower=0> alphastaysd;
        real<lower=0> alphaforgettingsd;
        real<lower=0> sidesd;

        array[NS] real beta_mouse;
        array[NS] real stay_mouse;
        array[NS] real<lower=0, upper=1> alpha_mouse;
        array[NS] real<lower=0, upper=1> alphastay_mouse;
        array[NS] real<lower=0, upper=1> alphaforgetting_mouse;
        array[NS] real side_mouse;

        real<lower=0> sidesess_sd;
        array[NSxNSESS] real sides;
        real<lower=0> alpha_ses_sd;
        array[NSxNSESS] real<lower=0, upper=1> alpha_ses;
        real<lower=0> alphaforgetting_ses_sd;
        array[NSxNSESS] real<lower=0, upper=1> alphaforgetting_ses;
        real<lower=0> alphastay_ses_sd;
        array[NSxNSESS] real<lower=0, upper=1> alphastay_ses;
        real<lower=0> stay_ses_sd;
        array[NSxNSESS] real stay_ses;
        real<lower=0> beta_ses_sd;
        array[NSxNSESS] real beta_ses;

}


model {
        // for each var_mouse
        alpham ~ normal(0.5,1);
        alphastaym ~ normal(0.5,1);
        betam ~ normal(0,1);
        staym ~ normal(0,1);
        sidem ~ normal(0,1);
        alphaforgettingm ~ normal(0.5,1);
        alphasd ~ normal(0,1);
        alphastaysd ~ normal(0,1);
        betasd ~ normal(0,1);
        staysd ~ normal(0,1);
        sidesd ~ normal(0,1);
        alphaforgettingsd ~ normal(0,1);

        // pulling out mouse so there's no if statement in for loop below
        // more efficient for stan
        for (mouse in 1:NS){
            alpha_mouse[mouse] ~ normal(alpham,alphasd);
            alphastay_mouse[mouse] ~ normal(alphastaym,alphastaysd);

            beta_mouse[mouse] ~ normal(betam,betasd);
            stay_mouse[mouse] ~ normal(staym,staysd);
            side_mouse[mouse] ~ normal(sidem, sidesd);
            alphaforgetting_mouse[mouse] ~ normal(alphaforgettingm, alphaforgettingsd);

            }

        for (ms_i in 1:NSxNSESS) {
                vector[2] q;
                vector[2] qstay;

                int choice;
                int nonchoice;

                alpha_ses[ms_i] ~ normal(alpha_mouse[sub_idx[ms_i]], alpha_ses_sd);
                alphastay_ses[ms_i] ~ normal(alphastay_mouse[sub_idx[ms_i]], alphastay_ses_sd);
                beta_ses[ms_i] ~ normal(beta_mouse[sub_idx[ms_i]], beta_ses_sd);
                stay_ses[ms_i] ~ normal(stay_mouse[sub_idx[ms_i]], stay_ses_sd);
                sides[ms_i] ~ normal(side_mouse[sub_idx[ms_i]], sidesess_sd);
                alphaforgetting_ses[ms_i] ~ normal(alphaforgetting_mouse[sub_idx[ms_i]], alphaforgetting_ses_sd);

                for (i in 1:2) {
                        q[i]= 0;
                        qstay[i]= 0;
                }

                for (t in 1:NT_all[ms_i]) {
                        c[sub_idx[ms_i], sess_idx[ms_i], t] ~ bernoulli_logit(sides[ms_i]  + beta_ses[ms_i]  * (q[2] - q[1]) + stay_ses[ms_i] * (qstay[2] - qstay[1]));
                        choice = c[sub_idx[ms_i], sess_idx[ms_i], t];
                        nonchoice = !choice;
                        q[choice+1] = (1-alpha_ses[ms_i]) * q[choice+1] + alpha_ses[ms_i] * o[sub_idx[ms_i], sess_idx[ms_i],t]; // +1 Because of not 0 indexing
                        q[nonchoice+1] = q[nonchoice+1] - q[nonchoice+1]*alphaforgetting_ses[ms_i]; // +1 Because of not 0 indexing
                        qstay = qstay * (1 - alphastay_ses[ms_i]);
                        qstay[choice+1] = qstay[choice+1] + alphastay_ses[ms_i];
                }

        }

}

generated quantities {
        // Vector to store log-likelihood for each trial
        vector[total_trials] log_lik;
        
        // Global trial counter
        int trial_idx = 1;
        
        // Loop through sessions (same structure as model block)
        for (ms_i in 1:NSxNSESS) {
                vector[2] q;
                vector[2] qstay;
                
                int choice;
                int nonchoice;
                
                // Initialize Q-values and qstay
                for (i in 1:2) {
                        q[i] = 0;
                        qstay[i] = 0;
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
                        
                        // Update Q-values (same as model block)
                        q[choice+1] = (1-alpha_ses[ms_i]) * q[choice+1] + alpha_ses[ms_i] * o[sub_idx[ms_i], sess_idx[ms_i], t];
                        q[nonchoice+1] = q[nonchoice+1] - q[nonchoice+1] * alphaforgetting_ses[ms_i];
                        qstay = qstay * (1 - alphastay_ses[ms_i]);
                        qstay[choice+1] = qstay[choice+1] + alphastay_ses[ms_i];
                        
                        // Move to next trial
                        trial_idx += 1;
                }
        }
}
