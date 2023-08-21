# RIM-interpret

RIM-interpret is a Python package designed to enhance the interpretability of machine learning models.

## Installation

Use pip to install RIM-interpret.

## Usage

RIM-interpret is compatible with most linear and tree-based regression models. In the future, we hope to expand the compatibility to inlcude more regression models and an option for classification tasks.

```python
import RIM_interpret

import sklearn
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split


#Import example dataset and convert to pandas df
data = datasets.load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
print(df.head())
#Predictors
X=df.drop("target", axis=1)
#Target
y=df["target"]

#Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)


#Train elastic net regression model
en_model = ElasticNet(alpha = 0.1, l1_ratio = 0.5)
fit_en = en_model.fit(X_train, y_train)
#Create dataframes to test
pfi_df_en = RIM_interpret.get_pfi(fit=fit_en, X_test=X_test, y_test=y_test)
shap_df_en = RIM_interpret.get_shap(fit=fit_en, X_test=X_test, model_type="Linear")
lime_df_en = RIM_interpret.get_lime(fit=fit_en, X_train=X_train, X_test=X_test)
inter_df_en = RIM_interpret.get_inter(fit=fit_en, X_train=X_train, X_test=X_test, y_test=y_test model_type="Linear")

```

## Contributing

Please create a GitHub issue for any bugs or questions (https://github.com/xloffree/RIM-interpret).

## License

[MIT](https://choosealicense.com/licenses/mit/)