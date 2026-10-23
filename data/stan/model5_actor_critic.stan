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



parameters {
        // normal intercept model
        real betam;
        real<lower=0, upper=1>  alpham;
        real<lower=0, upper=1>  alphaforgettingm;
        real<lower=0, upper=1>  alpharewm;    
        real sidem;
        real dmwaterm;


        real<lower=0> betasd;
        real<lower=0> dmwatersd;

        real<lower=0> alphasd;
        real<lower=0> alphaforgettingsd;
        real<lower=0> alpharewsd;
        real<lower=0> sidesd;

        array[NS] real beta_mouse;
        array[NS] real dmwater_mouse;
        array[NS] real<lower=0, upper=1> alpha_mouse;
        array[NS] real<lower=0, upper=1> alphaforgetting_mouse;
        array[NS] real<lower=0, upper=1> alpharew_mouse;
        array[NS] real side_mouse;

        real<lower=0> sidesess_sd;
        array[NSxNSESS] real sides;
        real<lower=0> alpha_ses_sd;
        array[NSxNSESS] real<lower=0, upper=1> alpha_ses;
        real<lower=0> alphaforgetting_ses_sd;
        array[NSxNSESS] real<lower=0, upper=1> alphaforgetting_ses;
        real<lower=0> alpharew_ses_sd;
        array[NSxNSESS] real<lower=0, upper=1> alpharew_ses;
        real<lower=0> beta_ses_sd;
        array[NSxNSESS] real beta_ses;
        real<lower=0> dmwater_ses_sd;
        array[NSxNSESS] real dmwater_ses;
}


model {
        // for each var_mouse
        alpham ~ normal(0.5,1);
        alphaforgettingm  ~ normal(0.5,1);
        alpharewm  ~ normal(0.5,1);
        betam ~ normal(0,1);
        sidem ~ normal(0,1);
        dmwaterm ~ normal(0,1);
        alphasd ~ normal(0,1);
        alphaforgettingsd ~ normal(0,1);
        alpharewsd ~ normal(0,1);
        betasd ~ normal(0,1);
        sidesd ~ normal(0,1);
        dmwatersd ~ normal(0,1);

        // pulling out mouse so there's no if statement in for loop below
        // more efficient for stan
        for (mouse in 1:NS){
            alpha_mouse[mouse] ~ normal(alpham,alphasd);
            alphaforgetting_mouse[mouse] ~ normal(alphaforgettingm,alphaforgettingsd);
            alpharew_mouse[mouse] ~ normal(alpharewm,alpharewsd);

            beta_mouse[mouse] ~ normal(betam,betasd);
            side_mouse[mouse] ~ normal(sidem, sidesd);
            dmwater_mouse[mouse]~ normal(dmwaterm, dmwatersd);
            }

        for (ms_i in 1:NSxNSESS) {
                real alpha;
                real alphaforgetting;
                real alpharew;
                real betawater;
                real betawaterloss;
                real qwater;
                real rpe;
                real qrew;
                int choice;

                alphaforgetting_ses[ms_i] ~ normal(alphaforgetting_mouse[sub_idx[ms_i]], alphaforgetting_ses_sd);
                alpharew_ses[ms_i] ~ normal(alpharew_mouse[sub_idx[ms_i]], alpharew_ses_sd);
                alpha_ses[ms_i] ~ normal(alpha_mouse[sub_idx[ms_i]], alpha_ses_sd);
                beta_ses[ms_i] ~ normal(beta_mouse[sub_idx[ms_i]], beta_ses_sd);
                dmwater_ses[ms_i] ~ normal(dmwater_mouse[sub_idx[ms_i]], dmwater_ses_sd);
                sides[ms_i] ~ normal(side_mouse[sub_idx[ms_i]], sidesess_sd);
                alpha = alpha_ses[ms_i];
                alphaforgetting = alphaforgetting_ses[ms_i];
                alpharew = alpharew_ses[ms_i];
                qwater = 0;
                qrew= 0;
                betawater = beta_ses[ms_i];
                betawaterloss = dmwater_ses[ms_i];

                for (t in 1:NT_all[ms_i]) {
                  c[sub_idx[ms_i], sess_idx[ms_i], t] ~ bernoulli_logit(sides[ms_i] + qwater);
                  rpe = o[sub_idx[ms_i], sess_idx[ms_i], t] - qrew;
                  qrew = qrew + (alpharew * rpe);     
                  choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] - 1; // 1 / -1 to facilitate REINFORCE
                  if (o[sub_idx[ms_i], sess_idx[ms_i], t] == 1)
                    qwater = (1 - alpha) * qwater + (choice * betawater * (1-inv_logit(choice*(sides[ms_i] + qwater))) * rpe);
                  if (o[sub_idx[ms_i], sess_idx[ms_i], t] == 0)// r is -1
                    qwater = (1 - alphaforgetting) * qwater + (choice * betawaterloss * (1-inv_logit(choice*(sides[ms_i] + qwater))) * rpe);
                }
        }
}

generated quantities {
        // Use maximum possible trials
        vector[NS*NSESS*NT] log_lik;
        
        // Global trial counter
        int trial_idx = 1;
        
        // Loop through sessions (same structure as model block)
        for (ms_i in 1:NSxNSESS) {
                real alpha;
                real alphaforgetting;
                real alpharew;
                real betawater;
                real betawaterloss;
                real qwater;
                real rpe;
                real qrew;
                int choice;
                
                // Initialize parameters (same as model block)
                alpha = alpha_ses[ms_i];
                alphaforgetting = alphaforgetting_ses[ms_i];
                alpharew = alpharew_ses[ms_i];
                qwater = 0;
                qrew = 0;
                betawater = beta_ses[ms_i];
                betawaterloss = dmwater_ses[ms_i];
                
                // Loop through trials in this session
                for (t in 1:NT_all[ms_i]) {
                        // Compute choice probability (same as model block)
                        real choice_logit = sides[ms_i] + qwater;
                        
                        // Get actual choice (0 or 1)
                        int c_val = c[sub_idx[ms_i], sess_idx[ms_i], t];
                        
                        // Store log-likelihood for this trial
                        log_lik[trial_idx] = bernoulli_logit_lpmf(c_val | choice_logit);
                        
                        // Compute reward prediction error
                        rpe = o[sub_idx[ms_i], sess_idx[ms_i], t] - qrew;
                        
                        // Update qrew (reward expectation)
                        qrew = qrew + (alpharew * rpe);
                        
                        // Convert choice to 1/-1 for REINFORCE update
                        choice = 2 * c_val - 1;
                        
                        // Update qwater based on outcome, modulated by RPE
                        if (o[sub_idx[ms_i], sess_idx[ms_i], t] == 1) {
                                // Reward received
                                qwater = (1 - alpha) * qwater + 
                                        (choice * betawater * (1 - inv_logit(choice * (sides[ms_i] + qwater))) * rpe);
                        }
                        if (o[sub_idx[ms_i], sess_idx[ms_i], t] == 0) {
                                // No reward (water loss)
                                qwater = (1 - alphaforgetting) * qwater + 
                                        (choice * betawaterloss * (1 - inv_logit(choice * (sides[ms_i] + qwater))) * rpe);
                        }
                        
                        // Move to next trial
                        trial_idx += 1;
                }
        }
}
