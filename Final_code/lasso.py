import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
from lassoScratch import CustomLASSO 

mean_weights, mean_idx = [], []
cell_type = ["Excitatory neurons", "Others", "Oligodendrocytes", "Inhibitory neurons"]
for j in range(4):
    train_all = [] 
    test_all = [] 
    weights = []
    labels = pd.read_csv("metadata.csv")

    for i in range(1,6): 
    # train data 
        data = pd.read_csv(f"cv_splits/fold_{i}/train_labeled.csv")


        data = data.merge(labels[['Donor ID', 'Last CASI Score']], on='Donor ID', how='left')
        data["mean_expression"] = data["mean_expression"].apply(
            lambda x: [float(i) for i in x.strip("[]").split(",")]
        ) 

        data_type = data[data['cluster'] == j]
        #data_type2 = data[data['cluster'] == 1]
        #data_type3 = data[data['cluster'] == 2]
        #data_type4 = data[data['cluster'] == 3]

    # test data 
        data_t = pd.read_csv(f"cv_splits/fold_{i}/test_labeled.csv")
        data_t = data_t.merge(labels[['Donor ID', 'Last CASI Score']], on='Donor ID', how='left')
        data_t["mean_expression"] = data_t["mean_expression"].apply(
            lambda x: [float(i) for i in x.strip("[]").split(",")]
        )
        data_t_type = data_t[data_t['cluster'] == j]
        #data_t_type2 = data_t[data_t['cluster'] == 1]
        #data_t_type3 = data_t[data_t['cluster'] == 2]
        #data_t_type4 = data_t[data_t['cluster'] == 3]

        X_train = np.array(data_type["mean_expression"].tolist())                       # numpy array of features
        y_train = data_type["Last CASI Score"].values       # pandas Series of labels (with 4 classes)
        print(X_train.shape)

        X_test = np.array(data_t_type["mean_expression"].tolist())
        y_test = data_t_type["Last CASI Score"].values
        print(X_test.shape)

        # Normalize the training and test data 
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) 

        n_epochs = 100
        alpha_value = 0.5
        learning_rate = 1

        sgd_lasso = CustomLASSO(alpha_value, learning_rate, learning_rate_schedule='optimal',
                    random_state=42, warm_start=True)

        train_mse_history = []
        test_mse_history = []

        print(f"Starting training for {n_epochs} epochs...")
        for epoch in range(n_epochs):
            sgd_lasso.partial_fit(X_train_scaled, y_train) # No need to loop through y classes for regression

            y_train_pred = sgd_lasso.predict(X_train_scaled)
            train_mse = mean_squared_error(y_train, y_train_pred)
            train_mse_history.append(train_mse)

            y_test_pred = sgd_lasso.predict(X_test_scaled)
            test_mse = mean_squared_error(y_test, y_test_pred)
            test_mse_history.append(test_mse)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs} - Train MSE: {train_mse:.4f} - Test MSE: {test_mse:.4f}")

        print("Training finished.")
        final_coefficients = sgd_lasso.weights
        
        train_all.append(train_mse_history)
        test_all.append(test_mse_history)
        weights.append(final_coefficients)


    weights = np.array(weights)
    mean_weight = np.mean(weights, axis=0)
    sorted_indices = np.argsort(np.abs(mean_weight))[::-1]

    # Get the top 10 genes with largest absolute weights
    top10_indices = sorted_indices[:10]
    selected_weight = mean_weight[top10_indices]
    mean_weights.append(selected_weight)
    mean_idx.append(top10_indices)

    train_array = np.array(train_all)  # shape (5, n_epochs)
    test_array = np.array(test_all)    # shape (5, n_epochs)

    # Calculate mean and standard deviation across folds for each epoch
    train_mean = np.mean(train_array, axis=0)  # Mean across folds for each epoch
    train_std = np.std(train_array, axis=0)    # Std across folds for each epoch

    test_mean = np.mean(test_array, axis=0)
    test_std = np.std(test_array, axis=0)

    # Create x-axis for epochs
    epochs = np.arange(1, train_mean.shape[0] + 1)  # Dynamically set based on actual number of epochs


    plt.figure(figsize=(10, 6))

    # Plot training mean and standard deviation
    plt.plot(epochs, train_mean, label='Train Mean', color='blue')
    plt.fill_between(epochs, train_mean - train_std, train_mean + train_std, 
                    color='blue', alpha=0.2, label='Train ±1 Std Dev')

    # Plot testing mean and standard deviation
    plt.plot(epochs, test_mean, label='Test Mean', color='orange')
    plt.fill_between(epochs, test_mean - test_std, test_mean + test_std, 
                    color='orange', alpha=0.2, label='Test ±1 Std Dev')

    # Customize plot
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title(f"Loss of {cell_type[j]}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

   
    plt.savefig(f"cell{j}_metrics")
    plt.show()

# Save the gene weights of each cell type
pd.DataFrame(mean_weights).to_csv('mean_weights.csv', header=False, index=False)
pd.DataFrame(mean_idx).to_csv('mean_idx.csv', header=False, index=False)