from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

class RandomForest(object):
    """integrated class for RandomForestRegressor and RandomForestClassifier"""
    def __init__(self, params):
        """initialize two random forest models and job parameters"""
        self._unpack(params)
        self.reg    = RandomForestRegressor(criterion='mse', **self.params)
        self.clf    = RandomForestClassifier(
            criterion       ='gini',
            **self.params,
            class_weight    = self.cw)
        self.done   = False
        self.err    = None

    def _unpack(self, params):
        """Unpack params to set parameters for both random forests"""
        params      = dict(params)
        self.cw     = params.pop('class_weight')
        self.params = params

    def fit_predict(self, X_train, y_train, X, vt):
        """dynamically decide model and returns prediction"""
        rf = None
        y = None
        try:
            if vt == 1:
                rf = self.reg
            else:
                rf = self.clf
            rf.fit(X_train, y_train)
            y = rf.predict(X)
            self.done = True
        except Exception as e:
            print(e)
            self.err = e

        return y
