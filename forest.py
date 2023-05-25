from tree import DecisionTreeRegressor
import numpy as np
class RandomForestRegressor(DecisionTreeRegressor):
    def __init__(self, x, y , Ans = 1, rand = 1, *args, **kwargs):
        super().__init__(x, y, *args, **kwargs)
        self.trees = []
        self.rand = rand
        self.forest_size = Ans


    def fit(self, x, y):
        for i in range(self.forest_size):
            mask = np.random.random(y.shape[0])
            x_f, y_f = x[mask < self.rand], y[mask < self.rand] 
            regr = DecisionTreeRegressor(x_f, y_f, max_depth=self.max_depth)
            regr.fit(x_f, y_f)
            self.trees.append(regr)


    def predict(self, x):
        y_pred = np.zeros(x.shape[0], dtype=float)
        N = self.forest_size
        for tree in self.trees:
            y_pred += tree.predict(x)
        return y_pred/N