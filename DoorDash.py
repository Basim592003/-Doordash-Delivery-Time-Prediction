#!/usr/bin/env python
# coding: utf-8

# # Doordash Delivery Time Prediction

# This project aims to predict the total delivery time for Doordash orders using multiple algorithms. Key steps include data preprocessing, feature engineering, handling multicollinearity, and scaling the data. 
# 
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r'D:\UNIVERSITY\DataScience\datasets\historical_data.csv'
data = pd.read_csv(file_path)

data.head()


# In[2]:


max_duration_row = data.loc[data['estimated_order_place_duration'].idxmax()] 
max_duration_row


# In[3]:


data.info()


# # Converting columns to datetime and making new columns and then one hot encoding, droping and fillanlly concating the resulting dataframe

# In[4]:


data['created_at'] = pd.to_datetime(data['created_at'])
data['actual_delivery_time'] = pd.to_datetime(data['actual_delivery_time'])


# In[5]:


from datetime import datetime
data['Totaltime'] = (data['actual_delivery_time'] - data['created_at']).dt.total_seconds()


# In[6]:


data['busyRatio'] = data['total_busy_dashers'] / data['total_onshift_dashers']


# In[7]:


data['estimated_non_prep_duration'] = data['estimated_store_to_consumer_driving_duration'] + data['estimated_order_place_duration']


# In[8]:


data['market_id'].nunique()


# In[9]:


data['store_id'].nunique()


# In[10]:


data['order_protocol'].nunique()


# In[11]:


mark = pd.get_dummies(data['market_id']).add_prefix('market_id_')

mark.head()


# In[12]:


order = pd.get_dummies(data['order_protocol']).add_prefix('order_protocol_')
order.head()


# In[13]:


storeUnique = data['store_id'].unique().tolist()

storeidandCategory = {
    store_id: data[data['store_id'] == store_id]['store_primary_category'].mode()
    for store_id in storeUnique
}


# In[14]:


def fill(store_id):
    try:
        """Return primary store category from the dictionary"""
        return storeidandCategory[store_id].values[0]
    except:
        return np.nan
# fill null values
data["nan_free_store_primary_category"] = data['store_id'].apply(fill)


# In[15]:


store_p_cat_dum = pd.get_dummies(data['nan_free_store_primary_category']).add_prefix('category_')


# In[16]:


train_df = data.drop(columns =
                    ["created_at", "market_id", "store_id", "store_primary_category", "actual_delivery_time"
, "nan_free_store_primary_category",'order_protocol'])

train_df.head()


# In[17]:


train_df = pd.concat([train_df,order,mark,store_p_cat_dum ],axis=1)
train_df = train_df.astype('float32')
train_df.head()


# In[18]:


np.where(np.any(~np.isfinite(train_df),axis=0) == True)
train_df.replace([np.inf, -np.inf], np.nan, inplace = True)
train_df.dropna(inplace=True)


# In[19]:


train_df.shape


# # Making a correlation heatmap just for lower triangle to peed up computing

# In[20]:


corr = train_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))


# In[21]:


f, ax = plt.subplots(figsize= (11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.3, center=0, square = True, linewidth =.5)
plt.show()


# # Dropping redundant pairs, getting most correlated values and dropping them and repeat

# In[22]:


def get_redundant_pairs(df):
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


# In[23]:


def get_top_abs_correlations(df,n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending =False)
    return au_corr[0:n]


# In[24]:


print(get_top_abs_correlations(train_df,20))


# In[25]:


train_df = data.drop(columns=["created_at", "market_id", "store_id", "store_primary_category", 
                                         "actual_delivery_time", "nan_free_store_primary_category", "order_protocol"])


# In[26]:


train_df = pd.concat([train_df,order,store_p_cat_dum ],axis=1)
train_df = train_df.drop(columns=["total_onshift_dashers", "total_busy_dashers", 
                                  "category_indonesian", "estimated_non_prep_duration"])


# In[27]:


train_df = train_df.astype('float32')
#np.where(np.any(~np.isfinite(train_df),axis=0) == True)
train_df.replace([np.inf, -np.inf], np.nan, inplace = True)
train_df.dropna(inplace=True)


# In[28]:


train_df.shape


# In[29]:


print(get_top_abs_correlations(train_df,20))


# In[30]:


train_df = data.drop(columns=["created_at", "market_id", "store_id", "store_primary_category", 
                                         "actual_delivery_time", "nan_free_store_primary_category", "order_protocol"])
train_df = pd.concat([train_df,store_p_cat_dum ],axis=1)
train_df = train_df.drop(columns=["total_onshift_dashers", "total_busy_dashers", 
                                  "category_indonesian", "estimated_non_prep_duration"])
train_df = train_df.astype('float32')
#np.where(np.any(~np.isfinite(train_df),axis=0) == True)
train_df.replace([np.inf, -np.inf], np.nan, inplace = True)
train_df.dropna(inplace=True)
print(get_top_abs_correlations(train_df,20))


# In[31]:


train_df["percent_distinct_item_of_total"] = train_df["num_distinct_items"] / train_df["total_items"]
train_df["avg_price_per_item"] = train_df["subtotal"] / train_df["total_items"]
train_df.drop(columns=["num_distinct_items", "subtotal"], inplace=True)
print("Top Absolute Correlations")
print(get_top_abs_correlations(train_df, 20))


# In[32]:


train_df["price_range_of_items"] = train_df["max_item_price"] - train_df["min_item_price"]
train_df.drop(columns=["max_item_price", "min_item_price"], inplace=True)
print("Top Absolute Correlations")
print(get_top_abs_correlations(train_df, 20))


# In[33]:


train_df.shape


# # Computing vif and removing high vif values

# In[34]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
def compute_vif(features) :
    vif_data = pd.DataFrame()
    vif_data["feature"] = features
    vif_data["VIF"] = [variance_inflation_factor(train_df[features].values, i) for i in range(len(features))]
    return vif_data.sort_values(by=['VIF']).reset_index(drop=True)

features = train_df.drop(columns=["Totaltime"]).columns.to_list()
vif_data = compute_vif(features)
vif_data


# In[51]:


multicollinearity = True
while multicollinearity:
    highest_vif_feature = vif_data['feature'].values.tolist()[-1]
    print("I will remove", highest_vif_feature)
    features.remove(highest_vif_feature)
    vif_data = compute_vif(features)
    multicollinearity = False if len(vif_data[vif_data['VIF'] > 5]) == 0 else True

selected_features = vif_data['feature'].values.tolist()
vif_data


# In[38]:


selected_features


# # Splitting data and calculating ginis importance

# In[52]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

X = train_df[selected_features]
y = train_df["Totaltime"]
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


# In[53]:


feature_names = [f"feature {i}" for i in range(X.shape[1])]
forest = RandomForestRegressor(random_state=42)
forest.fit(X_train, y_train)

feats = {}  
for feature, importance in zip(X.columns, forest.feature_importances_):
    feats[feature] = importance  

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})


# In[54]:


importances.sort_values(by='Gini-importance').plot(kind='bar', rot=90, figsize=(15, 12))
plt.show()


# In[55]:


importances.sort_values(by='Gini-importance').index.tolist()[-35:]


# In[56]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

X_Train = X_train.values
X_Train = np.asarray(X_Train)
X_std = StandardScaler().fit_transform(X_Train)
pca = PCA().fit(X_std)
plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()


# # Making a function to scale, calculate rmse and reggession algoritms to avoid repition and finally training

# In[57]:


def scale(scaler, X, y):
    X_scaler = scaler
    X_scaler.fit_transform(X)
    X_scaled = X_scaler.transform(X)
    
    y_scaler = scaler
    y_scaler.fit_transform(y.values.reshape(-1, 1))
    y_scaled = y_scaler.transform(y.values.reshape(-1, 1))
    
    return X_scaled, y_scaled, X_scaler, y_scaler


# In[58]:


from sklearn.preprocessing import MinMaxScaler
X_scaled, y_scaled, X_scaler, Y_scaler = scale(MinMaxScaler(), X, y)
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)


# In[59]:


from sklearn.metrics import mean_squared_error

def rmse_with_inv_transform(scaler, y_test, y_pred_scaled, model_name):
    """Convert the scaled error to actual error and calculate RMSE."""
    y_predict = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
    
    rmse_error = mean_squared_error(y_test, y_predict[:, 0], squared=False)
    
    print("Error = {:.2f} in {}".format(rmse_error, model_name))
    return rmse_error, y_predict


# In[60]:


from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from sklearn import linear_model


# In[61]:


y


# In[62]:


from sklearn.metrics import mean_squared_error

def make_regression(X_train, y_train, X_test, y_test, model, model_name, verbose=True):
    """Apply selected regression model to data and measure error"""
    model.fit(X_train, y_train)
    y_predict = model.predict(X_train)
    train_error = mean_squared_error(y_train, y_predict, squared=False)
    y_predict = model.predict(X_test)
    test_error = mean_squared_error(y_test, y_predict, squared=False)
    if verbose:
        print("Train error = '{}'".format(train_error) + " in " + model_name)
        print("Test error = '{}'".format(test_error) + " in " + model_name)
    trained_model = model

    return trained_model, y_predict, train_error, test_error



# In[ ]:


pred_dict = {
    "regression_model": [],
    "feature_set": [],
    "scaler_name": [],
    "RMSE": []
}

regression_models = {
    "Ridge": linear_model.Ridge(),
    "DecisionTree": tree.DecisionTreeRegressor(max_depth=6),
    "RandomForest": RandomForestRegressor(),
    "XGBoost": XGBRegressor(),
    "LGBM": LGBMRegressor(),
    "MLP": MLPRegressor(),
}

feature_sets = {
    "full_dataset": X.columns.to_list(),
}

scalers = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    #"NoScale": None
}


for feature_set_name in feature_sets.keys():
    feature_set = feature_sets[feature_set_name]
    for scaler_name in scalers.keys():
        print(f"-------scaled with {scaler_name}------- included columns are {feature_set_name}")
        print("")
        for model_name in regression_models.keys():
            if scaler_name == "NotScale":
                X = train_df[feature_set]
                y = train_df["actual_total_delivery_duration"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                make_regression(X_train, y_train, X_test, y_test, regression_models[model_name], model_name, verbose=True)
            else:
                X_scaled, y_scaled, X_scaler, y_scaler = scale(scalers[scaler_name], X[feature_set], y)
                X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
                    X_scaled, y_scaled, test_size=0.2, random_state=42)
                _, y_predict_scaled, _, _ = make_regression(X_train_scaled, y_train_scaled[:,0], X_test_scaled, y_test_scaled,regression_models[model_name], model_name)
                rmse_error, y_predict = rmse_with_inv_transform(y_scaler, y_test, y_predict_scaled, model_name)
            pred_dict["regression_model"].append(model_name)
            pred_dict["feature_set"].append(feature_set_name)
            pred_dict["scaler_name"].append(scaler_name)
            pred_dict["RMSE"].append(rmse_error)




# In[64]:


pred_df = pd.DataFrame(pred_dict)
pred_df.columns


# # High rmse values obtained, continue with more feature engineering and training, even with neural networks

# In[65]:


pred_df


# In[77]:


train_df['prep_time'] = train_df['Totaltime'] - train_df['estimated_store_to_consumer_driving_duration']


# In[78]:


feature_sets = {
    "selected_features_20": importances.sort_values(by='Gini-importance')[-40:].index.to_list(),
}

scalers = {
    "StandardScaler": StandardScaler(),

}


# In[80]:


for feature_set_name in feature_sets.keys():
    feature_set = feature_sets[feature_set_name]
    
    for scaler_name in scalers.keys():
        print(f"-------scaled with {scaler_name}------- included columns are {feature_set_name}")
        print("")

        for model_name in regression_models.keys():
            X = train_df[feature_set]
            y = train_df["prep_time"]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            train_indices = X_train.index
            test_indices = X_test.index

            X_scaled, y_scaled, X_scaler, y_scaler = scale(scalers[scaler_name], X, y)

            X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
                X_scaled, y_scaled, test_size=0.2, random_state=42
            )

            X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled, test_size=0.2)
            _, y_predict_scaled, _, _ = make_regression(X_train_scaled, y_train_scaled[:,0], X_test_scaled, y_test_scaled,regression_models[model_name], model_name)

            rmse_error, y_predict = rmse_with_inv_transform(
                y_scaler, y_test, y_predict_scaled, model_name
            )

            pred_dict["regression_model"].append(model_name)
            pred_dict["feature_set"].append(feature_set_name)
            pred_dict["scaler_name"].append(scaler_name)
            pred_dict["RMSE"].append(rmse_error)


# In[81]:


pred_values_dict = {
    "Totaltime": train_df["Totaltime"][test_indices].values.tolist(),
    "prep_duration_prediction": y_predict[:,0].tolist(),  # Ensure it's a 1D list of predictions
    "estimated_store_to_consumer_driving_duration": train_df["estimated_store_to_consumer_driving_duration"][test_indices].values.tolist(),
    "estimated_order_place_duration": train_df["estimated_order_place_duration"][test_indices].values.tolist(),
}


# In[82]:


values_df = pd.DataFrame(pred_values_dict)
values_df


# In[83]:


values_df['sum'] = values_df['prep_duration_prediction'] + values_df['estimated_store_to_consumer_driving_duration']
values_df


# In[84]:


mean_squared_error(values_df["Totaltime"], values_df["sum"], squared=False)


# In[85]:


X = values_df[["prep_duration_prediction", "estimated_store_to_consumer_driving_duration", "estimated_order_place_duration"]]
y = values_df["Totaltime"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


# In[86]:


from sklearn.metrics import mean_squared_error

for model_name in regression_models.keys():
    _, y_predict, _, _ = make_regression(
        X_train, y_train, X_test, y_test, regression_models[model_name], model_name, verbose=False
    )
    
    rmse_error = mean_squared_error(y_test, y_predict, squared=False)
    
    pred_dict["regression_model"].append(model_name)
    pred_dict["feature_set"].append(feature_set_name)
    pred_dict["scaler_name"].append(scaler_name)
    pred_dict["RMSE"].append(rmse_error) 

    print(f"RMSE of {model_name}: {rmse_error}")


# In[87]:


import keras
from keras.models import Sequential
from keras. layers import Dense
import tensorflow as tf
tf. random. set_seed (42)
def create_model(feature_set_size):
    model = Sequential()
    model.add(Dense(16, input_dim=feature_set_size, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='sgd', loss='mse',
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


# In[88]:


print(f"-----scaled with {scaler_name}----- included columns are {feature_set_name}")
print("")
model_name = "ANN"
scaler_name = "StandardScaler"
X = values_df[["prep_duration_prediction", "estimated_store_to_consumer_driving_duration", "estimated_order_place_duration"]]
y = values_df["Totaltime"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_scaled, y_scaled, X_scaler, y_scaler = scale(scalers[scaler_name], X, y)
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42)
print("feature_set_size:", X_train_scaled.shape[1])
model = create_model(feature_set_size=X_train_scaled.shape[1])
history = model.fit(X_train_scaled, y_train_scaled, epochs=100, batch_size=64, verbose=1)
y_pred = model.predict(X_test_scaled)
rmse_error = rmse_with_inv_transform(y_scaler, y_test, y_pred, model_name)
pred_dict["regression_model"].append(model_name)
pred_dict["feature_set"].append(feature_set_name)
pred_dict["scaler_name"].append(scaler_name)
pred_dict["RMSE"].append(rmse_error)


# In[89]:


plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel( 'Epoch')
plt.show()


# In[90]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_model(feature_set_size):
    model = Sequential()
    model.add(Dense(128, input_dim=feature_set_size, activation='relu'))
    model.add(Dropout(0.2))  
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse', metrics=['mse'])
    return model

model = create_model(feature_set_size=X_train_scaled.shape[1])
history = model.fit(X_train_scaled, y_train_scaled, epochs=200, batch_size=128, verbose=1, validation_split=0.2)

y_pred_scaled = model.predict(X_test_scaled)
rmse_error = rmse_with_inv_transform(y_scaler, y_test, y_pred_scaled, model_name)
print(f"RMSE for {model_name}: {rmse_error}")


# # Improved the models but still not feasible so we go back to where we calculate vif and only work with vif scores less than 5

# In[91]:


pred_df = pd.DataFrame.from_dict(pred_dict)
pred_df = pred_df[pred_df["RMSE"].apply(lambda x: isinstance(x, (float, int)))]

pred_dict = pred_df.to_dict(orient="list")
pred_df.sort_values(by='RMSE')


# In[98]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
def compute_vif2(features) :
    vif_data = pd.DataFrame()
    vif_data["feature"] = features
    vif_data["VIF"] = [variance_inflation_factor(values_df[features].values, i) for i in range(len(features))]
    return vif_data.sort_values(by=['VIF']).reset_index(drop=True)


# In[99]:


features = values_df.drop(columns=["Totaltime"]).columns.to_list()
vif_data = compute_vif2(features)
vif_data


# In[101]:


features = values_df.drop(columns=["Totaltime", "sum"]).columns.to_list()
vif_data = compute_vif2(features)
print(vif_data)


# In[104]:


correlation_matrix = values_df.drop(columns=["Totaltime", "sum"]).corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

plt.title("Feature Correlation Heatmap", fontsize=16)
plt.show()


# In[107]:


multicollinearity = True
while multicollinearity:
    highest_vif_feature = vif_data['feature'].values.tolist()[-1]
    print("I will remove", highest_vif_feature)
    features.remove(highest_vif_feature)
    vif_data = compute_vif(features)
    multicollinearity = False if len(vif_data[vif_data['VIF'] > 5]) == 0 else True

selected_features = vif_data['feature'].values.tolist()
vif_data


# In[111]:


selected_features = vif_data[vif_data['VIF'] < 5]['feature'].values.tolist()

filtered_values_df = train_df[selected_features + ["Totaltime"]].copy()


if "prep_duration_prediction" in filtered_values_df.columns and "estimated_store_to_consumer_driving_duration" in filtered_values_df.columns:
    filtered_values_df['sum'] = (
        filtered_values_df['prep_duration_prediction'] +
        filtered_values_df['estimated_store_to_consumer_driving_duration']
    )

filtered_values_df.head()


# In[115]:


correlation_matrix = filtered_values_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", fmt=".2f", linewidths=0.5)

plt.title("Feature Correlation Heatmap", fontsize=1)
plt.show()


# # And the final metrics are provided down below and looks like on average the model is off by 3 mins

# In[121]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X = filtered_values_df.drop(columns=["Totaltime"]).values
y = filtered_values_df["Totaltime"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge_model = Ridge(alpha=1.0) 
ridge_model.fit(X_train_scaled, y_train)

y_pred = ridge_model.predict(X_test_scaled)

# Metrics
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Ridge Regression Performance Metrics:")
print("RMSE:", rmse)  
print("MAE:", mae)
print("R^2 Score:", r2)

metrics = {
    "RMSE": rmse,
    "MAE": mae,
    "R^2": r2
}
metrics


# In[ ]:




