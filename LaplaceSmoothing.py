import numpy as np 

class  LaplaceSmoothing:

    def __init__(self):
        """
			Attributes:
				likelihoods: Likelihood of each feature per class
				class_priors: Prior probabilities of classes 
				pred_priors: Prior probabilities of features 
				features: All features of dataset

        """
        self.features = list
        self.likelihoods = {}
        self.class_priors = {}
        self.pred_priors = {}
        self.seen_feature_values = {}  # Store unique values of features seen during training

        
        self.X_train = np.array
        self.y_train = np.array
        self.train_size = int
        self.num_feats = int
  
        self.num_levels = int
    
    def fit(self, X, y, num_levels):

        self.features = list(X.columns)
        self.X_train = X
        self.y_train = y
        self.train_size = X.shape[0]
        self.num_feats = X.shape[1] 
        self.num_levels = num_levels
        
        for feature in self.features:
            self.likelihoods[feature] = {}
            self.pred_priors[feature] = {}
            self.seen_feature_values[feature] = []  
            
            for feat_val in np.unique(self.X_train[feature]):
                self.pred_priors[feature].update({feat_val: 0})
                self.seen_feature_values[feature].append(feat_val)  # Track seen values
 
            
                for outcome in np.unique(self.y_train):
                    self.likelihoods[feature].update({str(feat_val) + '_' + str(outcome): 0})
                    self.class_priors.update({outcome: 0})  
        
        self._calc_class_prior()
        self._calc_likelihoods()
        self._calc_predictor_prior()

    def _calc_class_prior(self):
        """ P(c) - Prior Class Probability """

        for outcome in np.unique(self.y_train):
            outcome_count = sum(self.y_train == outcome)
            self.class_priors[outcome] = outcome_count / self.train_size

    def _calc_likelihoods(self, a=1):
        """ P(x|c) - Likelihood """

        for i,feature in enumerate(self.features):
            v = self.num_levels[i]

            for outcome in np.unique(self.y_train):
                outcome_count = sum(self.y_train == outcome)
                feat_likelihood = self.X_train[feature][self.y_train[self.y_train == outcome].index.values.tolist()].value_counts().to_dict()
                
                for feat_val in np.unique(self.X_train[feature]):
                    count = feat_likelihood.get(feat_val, 0)
                    self.likelihoods[feature][str(feat_val) + '_' + str(outcome)] = (count + a) / (outcome_count + a * v)
                    
    def _calc_predictor_prior(self):    
        """ P(x) """
        for feature in self.features:
            feat_vals = self.X_train[feature].value_counts().to_dict()

            for feat_val, count in feat_vals.items():
                self.pred_priors[feature][feat_val] = count/self.train_size


    def predict(self, X, a=1):
        """ Calculates Posterior probability P(c|x) """
        results = []
        X = np.array(X)

        for query in X:
            skip_pattern = False
            probs_outcome = {}
            for outcome in np.unique(self.y_train):
                prior = self.class_priors[outcome]
                likelihood = 1
                
                for i, (feat, feat_val) in enumerate(zip(self.features, query)):
                    if feat_val not in self.seen_feature_values[feat]:  # Check if unseen feature value 
                        print(f"Warning: Skipping pattern due to unseen feature value '{feat_val}' for feature '{feat}'.")
                        skip_pattern = True
                        break
                    
                    likelihood *= self.likelihoods[feat].get(str(feat_val) + '_' + str(outcome), a / (self.train_size + a * self.num_levels[i]))

                if skip_pattern:
                    break
                
                posterior = (likelihood * prior)
                probs_outcome[outcome] = posterior
            
            if not skip_pattern:    
                result = max(probs_outcome, key = lambda x: probs_outcome[x])
                results.append(result)

        return np.array(results)