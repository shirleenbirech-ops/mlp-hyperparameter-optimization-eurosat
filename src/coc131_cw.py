import numpy as np
from PIL import Image
import os
import glob

# Please write the optimal hyperparameter values you obtain in the global variable 'optimal_hyperparm' below. This
# variable should contain the values when I look at your submission. I should not have to run your code to populate this
# variable.
optimal_hyperparam = {'activation': 'relu', 'solver':'adam', 'learning_rate_init': 0.0001,'batch_size': 16, 'alpha': 0.1, 'hidden_layer_sizes': (300, 150)}




class COC131:




    def __init__(self):
        self.x = None
        self.y = None
        self.q3_logs={}
        self.optimal_hyperparam = {
            'activation': 'relu',
            'solver': 'adam',
            'learning_rate_init': 0.0001,
            'batch_size': 16,
            'alpha': 0.1,
            'hidden_layer_sizes': (300, 150)
        }

   
        





    def q1(self, filename=None):
        """
        This function should be used to load the data. To speed-up processing in later steps, lower resolution of the
        image to 32*32. The folder names in the root directory of the dataset are the class names. After loading the
        dataset, you should save it into an instance variable self.x (for samples) and self.y (for labels). Both self.x
        and self.y should be numpy arrays of dtype float.

        :param filename: this is the name of an actual random image in the dataset. You don't need this to load the
        dataset. This is used by me for testing your implementation.
        :return res1: a one-dimensional numpy array containing the flattened low-resolution image in file 'filename'.
        Flatten the image in the row major order. The dtype for the array should be float.
        :return res2: a string containing the class name for the image in file 'filename'. This string should be same as
        one of the folder names in the originally shared dataset.


        SB: This function assumes that the datatset is stored in the 'EUROSTATRBG' folder, and image
        """


        new_imagesize = (32, 32)

        base_dir = os.path.dirname(__file__)
        data = os.path.join(base_dir, 'EuroSAT_RGB')

        image_path = glob.glob(os.path.join(data, '*', '*.jpg'))
        image_label = [os.path.basename(os.path.dirname(p)) for p in image_path]




        def load_image(path) :
            img = Image.open(path).convert('RGB')
            return np.array(img.resize(new_imagesize), dtype=float)
        
        images = [load_image(path) for path in image_path]
        self.x = np.array(images, dtype=float)
        self.y = np.array(image_label)




        res1 = np.zeros(1)
        res2 = ''
        if filename:
            for path in image_path:
                if path.endswith(filename):
                    print(f"Found match: {path}")  
                    img = load_image(path)
                    res1 = img.flatten()
                    res2 = os.path.basename(os.path.dirname(path))
                    break

       
        return res1, res2
    
    def _standardize_data(self, data, std=2.5):

        """
        SB: Helper function to standardize input data to mean=0 and given standard deviation (default=2.5).
        The function expects a 2d arrray where rows represent samples and columns represent features. Each feature is then
        standardised to have a mean 0 and unit variance using StsndardScaler, then rescaled to the standard deviation by multiplying 
        the standardised values by std.

        Parameters:
        :param data: 2D NumPy array of shape (n_samples, n_features)
        :param std: desired standard deviation for the output
        :return scaler: fitted StandardScaler object, with the mean and variance information
        :return scaled_data: transformed data with desired std
        """
        from sklearn.preprocessing import StandardScaler

        #Intialise a standard scaler object
        scaler = StandardScaler()

        #Fit the scaler to the data and transform it to have a mean of 0 and std of 1
        standardized_data = scaler.fit_transform(data)

        #Rescale the data to have the desired std
        scaled_data = standardized_data * std

        return scaler, scaled_data


    def q2(self, inp):
        """
        This function should compute the standardized data from a given 'inp' data. The function should work for a
        dataset with any number of features.

        :param inp: an array from which the standardized data is to be computed.
        :return res2: a numpy array containing the standardized data with standard deviation of 2.5. The array should
        have the same dimensions as the original data
        :return res1: sklearn object used for standardization.

        SB: This function calls the '_standardize_data'method to standardize the features of the dataset.
        First the input is reshaped to (n_samples, n_features) to ensure that it is compatible with StandardScaler,
        applies the scaling, and the reshapes the standardized data back into its original dimensions

        """
        #Storing original to be restored later
        original_shape = inp.shape


        #Flatten the input for standardization
        reshaped_input = inp.reshape(inp.shape[0], -1)

        #Standardize and rescale the flattened the input data
        res1, flattened_output = self._standardize_data(reshaped_input, std=2.5)

        #Reshape the standardized data back to its original shape
        res2 = flattened_output.reshape(original_shape)

        return res2, res1
    

    def tune_single_param(self, param_name, values, fixed_config, test_size=0.3, pre_split_data=None):

        """
        SB:
    Tunes a single hyperparameter by training an MLPClassifier with different values and 
    selecting the one that gives the highest test accuracy.

    Parameters:
    - param_name (str): The name of the hyperparameter to tune 
    - values (list): A list of values to test for the specified hyperparameter.
    - fixed_config (dict): Dictionary of fixed hyperparameters to use during tuning.
    - test_size (float): Proportion of the dataset to use for testing (default is 0.3).
    - pre_split_data (tuple, optional): Optional tuple containing pre-split (X_train, X_test, y_train, y_test).

    Returns:
    - dict: The updated hyperparameter configuration with the best-performing value for the tuned parameter.
    """
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import train_test_split
        import numpy as np
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        import copy

    # Standardizing the data
        X,y = self.preprocess()

        if pre_split_data:
            X_train, X_test, y_train, y_test = pre_split_data
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

        max_epochs = 30
        best_val = None
        best_acc = -np.inf
        test_acc_log = {}
        #Tuning Loop 
        for i, val in enumerate(values, 1):
            config = copy.deepcopy(fixed_config)  
            config[param_name] = val

            print(f"  [{i}/{len(values)}] Testing {param_name} = {val}")
            # Ignore the convergence warnings for short test runs
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    model = MLPClassifier(**config, max_iter=max_epochs, random_state=42)
                    model.fit(X_train, y_train)
                acc = model.score(X_test, y_test)
                print(f"    --> Accuracy: {acc:.4f}")

                #Updates the log with the best value of the tuned hypeparameter
                test_acc_log[str(val)] = acc
                if acc > best_acc:
                    best_acc = acc
                    best_val = val
            except Exception as e:
                test_acc_log[str(val)] = None
                print(f"     Skipping {val} due to error: {e}")


        #Save the log and update the configuration with the best value found

        config = copy.deepcopy(fixed_config)
        config[param_name] = best_val
        self.q3_logs[param_name] = test_acc_log
        print(f"\n Best {param_name}: {best_val} (Accuracy: {best_acc:.4f})")
        return config


    
    def train_final_model(self, config, test_size=0.3, pre_split_data=None, max_epochs=60, patience=5):
        """
    SB:  Trains an MLPClassifier using manual epoch-by-epoch updates with early stopping.

        Uses warm_start=True and max_iter=1 to track loss, training accuracy, and test accuracy 
        across epochs. Stops early if test accuracy does not improve after a set number of epochs.

        Parameters:
        - config (dict): Model hyperparameters.
        - test_size (float): Proportion of data for testing (default 0.3).
        - pre_split_data (tuple): Optional (X_train, X_test, y_train, y_test).
        - max_epochs (int): Total number of training epochs.
        - patience (int): Early stopping threshold.

        Returns:
        - model: Trained MLPClassifier.
        - losses: Loss values per epoch.
        - train_accs: Training accuracy per epoch.
        - test_accs: Test accuracy per epoch.
    """
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import train_test_split
        import numpy as np

        #Data prep 
        X,y = self.preprocess()
    # Use pre split data if available, else split the dataset
        if pre_split_data:
            X_train, X_test, y_train, y_test = pre_split_data
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=42
            )

        self.X_test = X_test
        self.y_test = y_test    

        #Setting up the Model 
        config = config.copy()
        config.update({
            'warm_start': True,
            'max_iter': 1 
        })
        model = MLPClassifier(**config, random_state=42)


        #Training loop with early stopping 
        losses, train_accs, test_accs = [], [], []
        best_test_acc = -np.inf
        epochs_no_improve = 0
        for epoch in range(max_epochs):
            model.fit(X_train, y_train)

            #Track metrics
            losses.append(model.loss_)
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            train_accs.append(train_acc)
            test_accs.append(test_acc)


            #Early stopping based on test accuracy
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

            
    

        return model, np.array(losses), np.array(train_accs), np.array(test_accs)





    def plot_param_result(self, param_name, plot_type='bar'):
        """
        Plots the effect of a single hyperparameter on test accuracy.
        
        Parameters:
        - param_name: str, name of the hyperparameter to plot
        - plot_type: 'bar', 'line', or 'scatter'
        """
        import matplotlib.pyplot as plt

        log = self.q3_logs.get(param_name)
        if not log:
            print(f"No results found for {param_name}")
            return

        values = list(log.keys())
        accs = [log[k] if log[k] is not None else 0 for k in values]

        plt.figure(figsize=(8, 4))
        if plot_type == 'bar':
            plt.bar(values, accs, color='pink')
        elif plot_type == 'line':
            plt.plot(values, accs, marker='o', color='teal')
        elif plot_type == 'scatter':
            plt.scatter(values, accs, color='orange', s=60)
        else:
            print(f"Unknown plot type '{plot_type}'. Supported: bar, line, scatter.")
            return

        plt.title(f"Effect of '{param_name}' on Test Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel(param_name)
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

        


    def q3(self, test_size=None, pre_split_data=None, hyperparam=None):

        """
        This function should build a MLP Classifier using the dataset loaded in function 'q1' and evaluate model
        performance. You can assume that the function 'q1' has been called prior to calling this function. This function
        should support hyperparameter optimizations.

        :param test_size: the proportion of the dataset that should be reserved for testing. This should be a fraction
        between 0 and 1.
        :param pre_split_data: Can be used to provide data already split into training and testing.
        :param hyperparam: hyperparameter values to be tested during hyperparameter optimization.
        :return: The function should return 1 model object and 3 numpy arrays which contain the loss, training accuracy
        and testing accuracy after each training iteration for the best model you found.

         
        SB: Trains an MLPClassifier using the preprocessed dataset from `q1` and specified hyperparameters.

        Uses either internal train-test splitting (via `test_size`) or provided split data (`pre_split_data`).
        Calls `train_final_model()` to train the model and track loss, train accuracy, and test accuracy over epochs.

        Parameters:
            test_size (float, optional): Proportion of data for testing if no pre-split data is given.
            pre_split_data (tuple, optional): ((X_train, y_train), (X_test, y_test)) to bypass splitting.
            hyperparam (dict): Dictionary of hyperparameters for the MLP (e.g., solver, learning_rate_init, etc.).

        Returns:
            final_model (MLPClassifier): The trained model.
            losses (np.ndarray): Loss values over epochs.
            train_accs (np.ndarray): Training accuracy over epochs.
            test_accs (np.ndarray): Testing accuracy over epochs.
    
        """


        if hyperparam is None:
            raise ValueError("No hyperparameters provided. Run manual tuning functions first.")

        print("Training final model with manually tuned hyperparameters...")
        final_model, losses, train_accs, test_accs = self.train_final_model(
            config=hyperparam,
            test_size=test_size,
            pre_split_data=pre_split_data
        )
        self.optimal_hyperparam = hyperparam
        return final_model, losses, train_accs, test_accs



    def evaluate_param_effect(self, param_name, values, fixed_config, X_train, X_test, y_train, y_test):
        """
       
        Evaluates how different values of a single hyperparameter affect model performance and internal parameters.

        Trains an MLPClassifier for each value of the given hyperparameter while keeping all other 
        hyperparameters fixed. Records key performance metrics and parameter norms.

        Parameters:
        - param_name (str): The name of the hyperparameter to evaluate (e.g., 'alpha').
        - values (list): A list of values to test for the given parameter.
        - fixed_config (dict): A dictionary of other hyperparameter values to keep constant.
        - X_train, X_test, y_train, y_test (arrays): Pre-split training and test data.

        Returns:
        - results (dict): A dictionary containing:
            - param_name (list): The values tested.
            - train_acc (list): Training accuracy for each value.
            - test_acc (list): Test accuracy for each value.
            - loss (list): Final training loss for each value.
            - f1 (list): Weighted F1-score for each test run.
            - precision (list): Weighted precision score for each test run.
            - recall (list): Weighted recall score for each test run.
            - weight_norm (list): L2 norm of the learned weights.
            - bias_norm (list): L2 norm of the learned biases.
   
        """
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import precision_score, recall_score, f1_score
        import numpy as np

        results = {
            param_name: values,
            "train_acc": [],
            "test_acc": [],
            "loss": [],
            "weight_norm": [],
            "bias_norm": [],
            "recall": []
        }

        for val in values:
            print(f"Evaluating {param_name} = {val}")
            config = fixed_config.copy()
            config[param_name] = val

            clf = MLPClassifier(**config, max_iter=30, random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            results["train_acc"].append(clf.score(X_train, y_train))
            results["test_acc"].append(clf.score(X_test, y_test))
            results["loss"].append(clf.loss_)
            results["f1"].append(f1_score(y_test, y_pred, average='weighted'))
            results["precision"].append(precision_score(y_test, y_pred, average='weighted'))
            results["recall"].append(recall_score(y_test, y_pred, average='weighted'))

            # Norms
            weight_norm = np.sqrt(sum(np.sum(w**2) for w in clf.coefs_))
            bias_norm = np.sqrt(sum(np.sum(b**2) for b in clf.intercepts_))
            results["weight_norm"].append(weight_norm)
            results["bias_norm"].append(bias_norm)

        return results

        

    def q4(self):

        """
        This function should study the impact of alpha on the performance and parameters of the model. For each value of
        alpha in the list below, train a separate MLPClassifier from scratch. Other hyperparameters for the model can
        be set to the best values you found in 'q3'. You can assume that the function 'q1' has been called
        prior to calling this function.

        :return: res should be the data you visualized.

        SB:Evaluates the impact of different alpha values
        on model performance and internal parameters.

        For each alpha value, the model is trained from scratch using fixed optimal hyperparameters 
        (excluding alpha). It records test accuracy, training loss, precision, recall, F1 score, 
        and parameter norms weights and biases.

        Returns:
        - results (dict): A dictionary containing alpha values and their corresponding evaluation metrics 
        for plotting and analysis

        """

        from sklearn.neural_network import MLPClassifier
        
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import precision_score, recall_score, f1_score
        import numpy as np

        alpha_values = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 50, 100]
    
        # Data
        X,y=self.preprocess()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

        # Fixed config from q3 (minus alpha)
        base_config = {
            'activation': 'relu',
            'solver': 'sgd',
            'learning_rate_init': 0.0005,
            'batch_size': 16,
            'hidden_layer_sizes': (300, 150),
            
        }

        results = self.evaluate_param_effect("alpha", alpha_values, base_config, X_train, X_test, y_train, y_test)
        return results
    
    def preprocess(self):

        """
        Preprocesses the dataset by standardizing `self.x` and encoding `self.y`.

        Returns:
        - X: Standardized and reshaped feature data (2D array of shape (n_samples, n_features)).
        - y: Encoded labels as integers (1D array).
        """
        from sklearn.preprocessing import LabelEncoder

        # Standardize `self.x`
        X_std, _ = self.q2(self.x)
        X = X_std.reshape(X_std.shape[0], -1)  # Flatten images into 2D array

        # Encode `self.y`
        le = LabelEncoder()
        y = le.fit_transform(self.y)
        

        return X, y
    def q5(self):
        """
         This function should perform hypothesis testing to study the impact of using CV with and without Stratification
        on the performance of MLPClassifier. Set other model hyperparameters to the best values obtained in the previous
        questions. Use 5-fold cross validation for this question. You can assume that the function 'q1' has been called
        prior to calling this function.

        :return: The function should return 4 items - the final testing accuracy for both methods of CV, p-value of the
        test and a string representing the result of hypothesis testing. The string can have only two possible values -
        'Splitting method impacted performance' and 'Splitting method had no effect'.


         SB:
            Compares the effect of using Stratified vs Non-Stratified 5-Fold Cross-Validation 
            on the performance of an MLPClassifier using previously tuned hyperparameters.

            The function trains the model using both methods, collects test accuracy scores, 
            and performs a paired t-test to assess if the difference in performance is statistically significant.

            Returns:
            - res1 (float): Mean test accuracy using Stratified K-Fold CV.
            - res2 (float): Mean test accuracy using standard K-Fold CV.
            - res3 (float): p-value from paired t-test.
            - res4 (str): Hypothesis test result — either 
            'Splitting method impacted performance' or 'Splitting method had no effect'.
        """
    


        from sklearn.model_selection import KFold, StratifiedKFold
        from sklearn.neural_network import MLPClassifier
        from scipy.stats import ttest_rel

        import numpy as np


       

       #Prepare Data
        X,y = self.preprocess()

       
        model_kwargs = optimal_hyperparam.copy()

        
        model_kwargs["solver"] = "adam"         
        model_kwargs["max_iter"] = 20        
        model_kwargs["early_stopping"] = True   

       
        def train_and_score(X, y, train_idx, test_idx, model_kwargs):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            clf = MLPClassifier(**model_kwargs)
            clf.fit(X_train, y_train)
            return clf.score(X_test, y_test)

    
        print("Running Stratified K-Fold")
        stratified_scores = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print(f"Training Stratified {i+1}/5")
            stratified_scores.append(train_and_score(X, y, train_idx, test_idx, model_kwargs))

        
        print("Running Non-Stratified K-Fold")
        non_stratified_scores = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for i, (train_idx, test_idx) in enumerate(kf.split(X)):
            print(f"Training Non-Stratified {i+1}/5")
            non_stratified_scores.append(train_and_score(X, y, train_idx, test_idx, model_kwargs))

        
        stat, p_value = ttest_rel(stratified_scores, non_stratified_scores)
        result_str = (
            "Splitting method impacted performance"
            if p_value < 0.05 else
            "Splitting method had no effect"
        )

        
        self.q5_strat_scores = stratified_scores
        self.q5_nonstrat_scores = non_stratified_scores

        return (stratified_scores[-1]), (non_stratified_scores[-1]), p_value, result_str


    def q6(self):
        """

        SB: 
            Performs unsupervised learning using Locally Linear Embedding (LLE) and evaluates each configuration
            using the silhouette score to identify the best number of neighbors (k).

            Assumes that self.q1 has been called and that self.x and self.y are available.

            The function:
            - Standardizes and flattens the input image data.
            - Applies LLE with different neighbor values in parallel.
            - Computes 2D embeddings and silhouette scores for each configuration.
            - Selects and returns the embedding corresponding to the best silhouette score.

            Returns:
                dict: {
                    'best_k': int, the neighbor value with the highest silhouette score,
                    'best_embedding': np.ndarray, the 2D embedding corresponding to best_k,
                    'y': np.ndarray, the true labels (used only for silhouette score and visualization),
                    'silhouette_scores': dict mapping neighbor values to their silhouette scores
                }
            
                This function should perform unsupervised learning using LocallyLinearEmbedding in Sklearn. You can assume that
                the function 'q1' has been called prior to calling this function.

                :return: The function should return the data you visualize.
        """
        from sklearn.manifold import LocallyLinearEmbedding
        from sklearn.metrics import silhouette_score
        from joblib import Parallel, delayed
        import numpy as np

        # Prepare data
        X_std, _ = self.q2(self.x)
        X_flat = X_std.reshape(X_std.shape[0], -1)
        y = self.y

        neighbor_values = [5, 20, 50, 100, 200]

        def run_lle(k):
            print(f"Computing LLE for k = {k}")
            try:
                lle = LocallyLinearEmbedding(n_components=2, n_neighbors=k, random_state=42)
                X_emb = lle.fit_transform(X_flat)
                score = silhouette_score(X_emb, y)
                return k, X_emb, score
            except Exception as e:
                print(f"  Failed for k = {k}: {e}")
                return k, None, None

        # Run in parallel
        results = Parallel(n_jobs=-1)(delayed(run_lle)(k) for k in neighbor_values)

        #Run Sequentially
        #results = []
        #for k in neighbor_values:
           # result = run_lle(k)
            #results.append(result)

        # Organize results
        silhouette_scores = {}
        embeddings = {}

        for k, emb, score in results:
            silhouette_scores[k] = score
            if emb is not None:
                embeddings[k] = emb

        # Select best k
        valid_scores = {k: v for k, v in silhouette_scores.items() if v is not None}
        best_k = max(valid_scores, key=valid_scores.get)
        best_embedding = embeddings[best_k]

        print(f"\nBest k = {best_k} with silhouette score = {valid_scores[best_k]:.4f}")

        return {
            "best_k": best_k,
            "best_embedding": best_embedding,
            "y": y,
            "silhouette_scores": silhouette_scores
        }


    def tune_hidden_layer_sizes(self, fixed_config=None, test_size=0.3, pre_split_data=None):
        """
        Tunes the 'hidden_layer_sizes' hyperparameter by testing a range of layer configurations.
        Evaluates model performance for each configuration and selects the one with the highest test accuracy.

        Parameters:
        - fixed_config (dict): Base configuration with fixed hyperparameters.
        - test_size (float): Proportion of the data used for testing.
        - pre_split_data (tuple): Optional pre-split (X_train, X_test, y_train, y_test).

        Returns:
        - best_config (dict): Updated configuration with the optimal hidden layer sizes.
        """

        #Hidden layer sizes to be tested

        values = [(64,), (128,), (64, 64), (128, 128), (160, 160), (256, 128), (300, 150), (300, 270)]

        print("\n Tuning: hidden_layer_sizes")

        print("Values to be tested:")
        for i, v in enumerate(values, 1):
            print(f"  [{i}/{len(values)}] {v}")

        #Call the tuning function for the hyperparameter
        best_config = self.tune_single_param('hidden_layer_sizes', values, fixed_config, test_size, pre_split_data)
        #Adds the best value to the log
        best_val = best_config.get('hidden_layer_sizes')

        ##Log the results

        best_acc = self.q3_logs['hidden_layer_sizes'].get(str(best_val))
        print(f"\n Best hidden_layer_sizes: {best_val} with accuracy = {best_acc:.4f}\n")

        return best_config





    def tune_activation(self, fixed_config=None, test_size=0.3, pre_split_data=None):
        """
        Tunes the 'activation' function used in the hidden layers.
        Compares performance across ReLU, Tanh, and Logistic activations.

        Parameters:
        - fixed_config (dict): Base configuration with fixed hyperparameters.
        - test_size (float): Proportion of the data used for testing.
        - pre_split_data (tuple): Optional pre-split (X_train, X_test, y_train, y_test).

        Returns:
        - best_config (dict): Configuration with the best-performing activation function.
        """
        # Activation functions to be tested
        values = ['relu', 'tanh', 'logistic']

        print("\n Tuning: activation")
        #CAll the tuning function for the hyperparameter
        best_config = self.tune_single_param('activation', values, fixed_config, test_size, pre_split_data)

        return best_config
    
    def tune_solver(self, fixed_config=None, test_size=0.3, pre_split_data=None):
        """
        Tunes the 'solver' used for weight optimization during training.
        Evaluates performance using Adam, SGD, and L-BFGS solvers.

        Parameters:
        - fixed_config (dict): Base configuration with fixed hyperparameters.
        - test_size (float): Proportion of the data used for testing.
        - pre_split_data (tuple): Optional pre-split (X_train, X_test, y_train, y_test).

        Returns:
        - best_config (dict): Configuration with the best-performing solver.
        """

        values = ['adam', 'sgd', 'lbfgs'] #VAlues to be tested
        
        print("\n Tuning: solver")
        #Call the tuning function for the hyperparameter
        best_config = self.tune_single_param('solver', values, fixed_config, test_size, pre_split_data)
        self.plot_param_result('solver')
        return best_config
    
    def tune_batch_size(self, fixed_config=None, test_size=0.3, pre_split_data=None):
        """
        Tunes the 'batch_size' parameter, which controls how many samples are used per training update.
        Tests values like 16, 32, and 'auto' to find the most effective setting.

        Parameters:
        - fixed_config (dict): Base configuration with fixed hyperparameters.
        - test_size (float): Proportion of the data used for testing.
        - pre_split_data (tuple): Optional pre-split (X_train, X_test, y_train, y_test).

        Returns:
        - best_config (dict): Configuration with the optimal batch size.
        """
        
        values = [16, 32, 'auto'] #Values to be tested
        print("\n Tuning: batch_size")
        #Call the tuning function for the hyperparameter
        best_config = self.tune_single_param('batch_size', values, fixed_config, test_size, pre_split_data)
        return best_config
    
    def tune_alpha(self, fixed_config=None, test_size=0.3, pre_split_data=None):
        """
        Tunes the 'alpha' regularization strength to prevent overfitting.
        Evaluates test performance across multiple values to find the best balance.

        Parameters:
        - fixed_config (dict): Base configuration with fixed hyperparameters.
        - test_size (float): Proportion of the data used for testing.
        - pre_split_data (tuple): Optional pre-split (X_train, X_test, y_train, y_test).

        Returns:
        - best_config (dict): Configuration with the optimal alpha value.
        """
        values = [0.0001, 0.001, 0.01, 0.1,] #Values to be tested
        print("\n Tuning: alpha")
        #Call the tuning function for the hyperparameter
        best_config = self.tune_single_param('alpha', values, fixed_config, test_size, pre_split_data)
        return best_config 
    
    def tune_learning_rate(self, fixed_config=None, test_size=0.3, pre_split_data=None):

        """
        Tunes the 'learning_rate_init' parameter, which controls the step size during weight updates.
        Evaluates model stability and convergence across a range of common learning rates.

        Parameters:
        - fixed_config (dict): Base configuration with fixed hyperparameters.
        - test_size (float): Proportion of the data used for testing.
        - pre_split_data (tuple): Optional pre-split (X_train, X_test, y_train, y_test).

        Returns:
        - best_config (dict): Configuration with the optimal learning rate.
        """

        values = [0.1, 0.01, 0.005, 0.001, 0.0005,0.0001] #Values to be tested
        print("\n Tuning: learning_rate_init")
        best_config = self.tune_single_param('learning_rate_init', values, fixed_config, test_size, pre_split_data)
        self.plot_param_result('learning_rate_init')
        return best_config



 
        